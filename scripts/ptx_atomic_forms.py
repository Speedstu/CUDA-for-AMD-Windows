#!/usr/bin/env python3
"""Hand-written PTX regression kernels for atomic instruction forms.

Background (issue #6): nvcc emits ``red.<space>.<op>.<type> [a], b`` for an
``atomicAdd()``/``atomicMin()``/... whose result is unused, and
``atom.<space>.<op>.<type> d, [a], b`` when it is used. The pinned upstream
ZLUDA PTX parser understands ``atom`` but has no ``red`` instruction, so a
kernel containing ``red`` is dropped from its module and
``cuModuleGetFunction`` returns CUDA_ERROR_NOT_FOUND (500) for it.

Every case here is one tiny module with one kernel, launched as one block of
256 threads. Thread ``t`` updates bin ``t & 3`` of a 4-element result, so each
bin receives 64 updates. The expected values come from a pure-Python CPU
reference (``cpu_reference``), not from any GPU backend:

    add 1 (u32)              -> [64, 64, 64, 64]
    add 0x1_0000_0001 (u64)  -> [0x40_0000_0040] * 4   (carry into the high word)
    add 1.0 (f32)            -> [64.0] * 4
    max t (s32)              -> [252, 253, 254, 255]
    min t - 300 (s32)        -> [-300, -299, -298, -297] (zero-initialised memory,
                                so an unsigned or skipped min leaves 0)

Each ``red`` case has an ``atom`` twin that differs in that one instruction
only. The twin proves the kernel shape and the expectation on a driver that
cannot run ``red`` yet, and tells "module loads" apart from "the device
computed the right result" once it can.

Result per case:
    pass          kernel ran and matched the CPU reference
    not_exported  module loaded, cuModuleGetFunction did not find the kernel
    rejected      cuModuleLoadData refused the module
    incorrect     kernel ran and the result differs, or launch/sync/copy failed

``summarize`` folds the cases into one verdict: ``incorrect`` if any case is
incorrect, otherwise ``unsupported`` if any kernel is missing or rejected,
otherwise ``pass``.

Run standalone (creates its own context):
    zluda.exe -- python scripts/ptx_atomic_forms.py
or through the harness as capability probe ``driver_ptx_atomics``.
``python scripts/ptx_atomic_forms.py --dump-ptx DIR`` writes the modules to
disk without touching a driver.
"""
from __future__ import annotations

import ctypes
import json
import struct
import sys
from dataclasses import dataclass
from typing import Any

THREADS = 256
BINS = 4

HEADER = """.version 7.0
.target sm_80
.address_size 64

"""

# element type -> (size in bytes, struct format, PTX type suffix)
_TYPES = {
    "u32": (4, "I"),
    "s32": (4, "i"),
    "u64": (8, "Q"),
    "f32": (4, "f"),
}


@dataclass(frozen=True)
class Case:
    name: str
    kernel: str
    instruction: str   # "atom", "red" or "control"
    space: str         # "global", "shared", "generic" (generic address of shared memory)
    op: str            # "add", "min", "max", "store"
    ctype: str         # key of _TYPES
    operand: str       # "register" or "immediate"
    ptx: str
    note: str = ""


def cpu_reference(case: Case) -> list[Any]:
    """What 256 threads leave in the four bins, computed on the CPU."""
    bins: list[Any] = [0.0 if case.ctype == "f32" else 0] * BINS
    for tid in range(THREADS):
        k = tid & 3
        if case.op == "store":
            bins[k] = 7
        elif case.op == "add" and case.ctype == "u32":
            bins[k] = (bins[k] + 1) & 0xFFFFFFFF
        elif case.op == "add" and case.ctype == "u64":
            bins[k] = (bins[k] + 0x100000001) & 0xFFFFFFFFFFFFFFFF
        elif case.op == "add" and case.ctype == "f32":
            bins[k] = bins[k] + 1.0
        elif case.op == "max" and case.ctype == "s32":
            bins[k] = max(bins[k], tid)
        elif case.op == "min" and case.ctype == "s32":
            bins[k] = min(bins[k], tid - 300)
        else:
            raise ValueError(f"no CPU reference for {case.op}.{case.ctype}")
    return bins


def _shared_kernel(name: str, body: str) -> str:
    """Threads 0..3 zero bins[tid]; every thread updates bins[tid & 3]; threads 0..3 copy out."""
    return HEADER + f""".visible .entry {name}(
    .param .u64 p_out
)
{{
    .reg .b32   %r<12>;
    .reg .b64   %rd<6>;
    .reg .pred  %p<4>;
    .shared .align 4 .b8 bins[16];

    mov.u32     %r1, %tid.x;
    setp.lt.u32 %p1, %r1, 4;
    shl.b32     %r2, %r1, 2;
    mov.u32     %r3, bins;
    add.s32     %r4, %r3, %r2;
    mov.u32     %r5, 0;
    @%p1 st.shared.u32 [%r4], %r5;
    bar.sync    0;
    and.b32     %r6, %r1, 3;
    shl.b32     %r6, %r6, 2;
    add.s32     %r7, %r3, %r6;
    mov.u32     %r8, 1;
    {body}
    bar.sync    0;
    @!%p1 bra   DONE;
    ld.shared.u32 %r10, [%r4];
    ld.param.u64 %rd1, [p_out];
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64     %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r10;
DONE:
    ret;
}}
"""


def _global_kernel(name: str, elem_size: int, body: str) -> str:
    """Every thread updates out[tid & 3] directly; the host zeroes out[] before the launch."""
    return HEADER + f""".visible .entry {name}(
    .param .u64 p_out
)
{{
    .reg .b32   %r<8>;
    .reg .b64   %rd<8>;
    .reg .f32   %f<4>;

    mov.u32     %r1, %tid.x;
    and.b32     %r2, %r1, 3;
    ld.param.u64 %rd1, [p_out];
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r2, {elem_size};
    add.s64     %rd4, %rd2, %rd3;
    {body}
    ret;
}}
"""


# 32-bit shared address in %r7 -> 64-bit generic address in %rd5
_TO_GENERIC = "cvt.u64.u32 %rd5, %r7;\n    cvta.shared.u64 %rd5, %rd5;\n    "
# ... and back to a 32-bit shared address in %r9 (the __cvta_generic_to_shared route of the llama histogram patch)
_TO_SHARED_AGAIN = _TO_GENERIC + "cvta.to.shared.u64 %rd5, %rd5;\n    cvt.u32.u64 %r9, %rd5;\n    "


def _build_cases() -> tuple[Case, ...]:
    cases: list[Case] = []

    def add(name, instruction, space, op, ctype, operand, ptx, note=""):
        cases.append(Case(name, "k_" + name, instruction, space, op, ctype, operand, ptx, note))

    add("control_shared_store", "control", "shared", "store", "u32", "register",
        _shared_kernel("k_control_shared_store", "mov.u32 %r9, 7;\n    st.shared.u32 [%r7], %r9;"),
        "no atomics: shared memory, bar.sync and the copy-out path")

    for operand, src in (("register", "%r8"), ("immediate", "1")):
        tag = "reg" if operand == "register" else "imm"
        # u32 add on global memory
        add(f"atom_global_add_u32_{tag}", "atom", "global", "add", "u32", operand,
            _global_kernel(f"k_atom_global_add_u32_{tag}", 4,
                           f"mov.u32 %r3, 1;\n    atom.global.add.u32 %r4, [%rd4], {'%r3' if operand == 'register' else '1'};"))
        add(f"red_global_add_u32_{tag}", "red", "global", "add", "u32", operand,
            _global_kernel(f"k_red_global_add_u32_{tag}", 4,
                           f"mov.u32 %r3, 1;\n    red.global.add.u32 [%rd4], {'%r3' if operand == 'register' else '1'};"))
        # u32 add on shared memory, 32-bit shared address
        add(f"atom_shared_add_u32_{tag}", "atom", "shared", "add", "u32", operand,
            _shared_kernel(f"k_atom_shared_add_u32_{tag}", f"atom.shared.add.u32 %r9, [%r7], {src};"))
        add(f"red_shared_add_u32_{tag}", "red", "shared", "add", "u32", operand,
            _shared_kernel(f"k_red_shared_add_u32_{tag}", f"red.shared.add.u32 [%r7], {src};"),
            "immediate form is what nvcc emitted for top_k_radix_histogram<256,8>" if operand == "immediate" else "")
        # u32 add on shared memory through a generic address (no state-space qualifier)
        add(f"atom_generic_add_u32_{tag}", "atom", "generic", "add", "u32", operand,
            _shared_kernel(f"k_atom_generic_add_u32_{tag}", _TO_GENERIC + f"atom.add.u32 %r11, [%rd5], {src};"))
        add(f"red_generic_add_u32_{tag}", "red", "generic", "add", "u32", operand,
            _shared_kernel(f"k_red_generic_add_u32_{tag}", _TO_GENERIC + f"red.add.u32 [%rd5], {src};"))

    # generic -> cvta.to.shared -> 32-bit shared address, immediate operand
    add("atom_shared_add_u32_cvta_to", "atom", "shared", "add", "u32", "immediate",
        _shared_kernel("k_atom_shared_add_u32_cvta_to", _TO_SHARED_AGAIN + "atom.shared.add.u32 %r11, [%r9], 1;"),
        "instruction sequence of GGML_CUDA_ZLUDA_ATOM_HISTOGRAM")
    add("red_shared_add_u32_cvta_to", "red", "shared", "add", "u32", "immediate",
        _shared_kernel("k_red_shared_add_u32_cvta_to", _TO_SHARED_AGAIN + "red.shared.add.u32 [%r9], 1;"))

    # other element types and reductions on global memory, register operand
    typed = (
        ("add", "u64", 8, "mov.u64 %rd5, 4294967297;", "%rd6", "%rd5"),
        ("add", "f32", 4, "mov.f32 %f1, 0f3F800000;", "%f2", "%f1"),
        ("max", "s32", 4, "", "%r4", "%r1"),
        ("min", "s32", 4, "sub.s32 %r3, %r1, 300;", "%r4", "%r3"),
    )
    for op, ctype, size, setup, dst, src in typed:
        prefix = (setup + "\n    ") if setup else ""
        add(f"atom_global_{op}_{ctype}", "atom", "global", op, ctype, "register",
            _global_kernel(f"k_atom_global_{op}_{ctype}", size, prefix + f"atom.global.{op}.{ctype} {dst}, [%rd4], {src};"))
        add(f"red_global_{op}_{ctype}", "red", "global", op, ctype, "register",
            _global_kernel(f"k_red_global_{op}_{ctype}", size, prefix + f"red.global.{op}.{ctype} [%rd4], {src};"))
    return tuple(cases)


CASES: tuple[Case, ...] = _build_cases()


def summarize(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Fold per-case results into one verdict; see the module docstring."""
    groups: dict[str, list[str]] = {"pass": [], "not_exported": [], "rejected": [], "incorrect": []}
    for name, rec in results.items():
        groups.setdefault(rec.get("status", "incorrect"), []).append(name)
    unknown = [k for k in groups if k not in ("pass", "not_exported", "rejected", "incorrect")]
    for key in unknown:
        groups["incorrect"].extend(groups.pop(key))
    if groups["incorrect"]:
        verdict = "incorrect"
    elif groups["not_exported"] or groups["rejected"]:
        verdict = "unsupported"
    else:
        verdict = "pass"
    return {
        "verdict": verdict,
        "total": len(results),
        "passed": len(groups["pass"]),
        "not_exported": sorted(groups["not_exported"]),
        "rejected": sorted(groups["rejected"]),
        "incorrect": sorted(groups["incorrect"]),
    }


def bind(cuda) -> None:
    """Declare the driver entry points used by run_cases on an already loaded nvcuda.dll."""
    c_int, c_uint, c_void_p, c_uint64, c_size_t = ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(c_void_p), c_void_p]
    cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(c_void_p), c_void_p, ctypes.c_char_p]
    cuda.cuModuleUnload.argtypes = [c_void_p]
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(c_uint64), c_size_t]
    cuda.cuMemFree_v2.argtypes = [c_uint64]
    cuda.cuMemcpyHtoD_v2.argtypes = [c_uint64, c_void_p, c_size_t]
    cuda.cuMemcpyDtoH_v2.argtypes = [c_void_p, c_uint64, c_size_t]
    cuda.cuCtxSynchronize.argtypes = []
    cuda.cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
                                    c_void_p, c_void_p, c_void_p]
    for fn in (cuda.cuModuleLoadData, cuda.cuModuleGetFunction, cuda.cuModuleUnload, cuda.cuMemAlloc_v2,
               cuda.cuMemFree_v2, cuda.cuMemcpyHtoD_v2, cuda.cuMemcpyDtoH_v2, cuda.cuCtxSynchronize,
               cuda.cuLaunchKernel):
        fn.restype = c_int


def run_case(cuda, case: Case) -> dict[str, Any]:
    """Load, launch and check one case. Requires a current CUDA context and bind(cuda)."""
    size, fmt = _TYPES[case.ctype]
    nbytes = size * BINS
    expected = cpu_reference(case)
    rec: dict[str, Any] = {"instruction": case.instruction, "space": case.space, "op": case.op,
                           "type": case.ctype, "operand": case.operand, "expected": expected}
    image = ctypes.create_string_buffer(case.ptx.encode("ascii") + b"\0")
    module = ctypes.c_void_p()
    device_out = ctypes.c_uint64()
    try:
        rec["module_load"] = int(cuda.cuModuleLoadData(ctypes.byref(module), ctypes.cast(image, ctypes.c_void_p)))
        if rec["module_load"] != 0:
            rec["status"] = "rejected"
            return rec
        func = ctypes.c_void_p()
        rec["get_function"] = int(cuda.cuModuleGetFunction(ctypes.byref(func), module, case.kernel.encode("ascii")))
        if rec["get_function"] != 0:
            rec["status"] = "not_exported"
            return rec
        rec["status"] = "incorrect"  # until the device result says otherwise
        rec["alloc"] = int(cuda.cuMemAlloc_v2(ctypes.byref(device_out), nbytes))
        if rec["alloc"] != 0:
            return rec
        zeros = ctypes.create_string_buffer(nbytes)
        rec["clear"] = int(cuda.cuMemcpyHtoD_v2(device_out.value, ctypes.cast(zeros, ctypes.c_void_p), nbytes))
        if rec["clear"] != 0:
            return rec
        ptr_arg = ctypes.c_uint64(device_out.value)
        params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(ptr_arg), ctypes.c_void_p))
        rec["launch"] = int(cuda.cuLaunchKernel(func, 1, 1, 1, THREADS, 1, 1, 0, None,
                                                ctypes.cast(params, ctypes.c_void_p), None))
        if rec["launch"] != 0:
            return rec
        rec["sync"] = int(cuda.cuCtxSynchronize())
        if rec["sync"] != 0:
            return rec
        host = ctypes.create_string_buffer(nbytes)
        rec["copy_back"] = int(cuda.cuMemcpyDtoH_v2(ctypes.cast(host, ctypes.c_void_p), device_out.value, nbytes))
        if rec["copy_back"] != 0:
            return rec
        got = list(struct.unpack("<%d%s" % (BINS, fmt), host.raw))
        rec["result"] = got
        if got == expected:
            rec["status"] = "pass"
        return rec
    finally:
        if device_out.value:
            try:
                cuda.cuMemFree_v2(device_out.value)
            except Exception:
                pass
        if module.value:
            try:
                cuda.cuModuleUnload(module)
            except Exception:
                pass


def run_cases(cuda) -> dict[str, Any]:
    """Run every case on an nvcuda.dll that already has a current context."""
    bind(cuda)
    results = {case.name: run_case(cuda, case) for case in CASES}
    return {"cases": results, "summary": summarize(results)}


def _standalone() -> int:
    if not hasattr(ctypes, "WinDLL"):
        print("ptx_atomic_forms: Windows only", file=sys.stderr)
        return 4
    cuda = ctypes.WinDLL("nvcuda.dll")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    for call, code in (("cuInit", cuda.cuInit(0)),):
        if code != 0:
            print(f"{call} returned CUDA error {code}", file=sys.stderr)
            return 2
    dev = ctypes.c_int()
    ctx = ctypes.c_void_p()
    if cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0 or cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value) != 0:
        print("could not create a CUDA context", file=sys.stderr)
        return 2
    try:
        report = run_cases(cuda)
    finally:
        cuda.cuCtxDestroy_v2(ctx)
    try:
        from loaded_driver_identity import loaded_module_identity
        report["loaded_nvcuda"] = loaded_module_identity()
    except Exception:
        pass
    for name, rec in report["cases"].items():
        print("%-34s %-13s %s" % (name, rec["status"], rec.get("result", "")))
    print("PTX_ATOMIC_FORMS:" + json.dumps(report, sort_keys=True))
    return {"pass": 0, "unsupported": 3, "incorrect": 2}[report["summary"]["verdict"]]


def dump_ptx(directory: str) -> int:
    """Write every case as <name>.ptx (no driver needed), e.g. to assemble them with NVIDIA's ptxas."""
    import os

    os.makedirs(directory, exist_ok=True)
    for case in CASES:
        with open(os.path.join(directory, case.name + ".ptx"), "w", encoding="ascii", newline="\n") as stream:
            stream.write(case.ptx)
    print(f"wrote {len(CASES)} PTX modules to {directory}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dump-ptx":
        sys.exit(dump_ptx(sys.argv[2]))
    sys.exit(_standalone())

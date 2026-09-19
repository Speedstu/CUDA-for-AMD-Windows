#!/usr/bin/env python3
"""Broad CUDA-facing capability probes for AMD/ZLUDA on Windows.

Each probe is designed to run in a separate process.  The PowerShell wrapper
therefore can terminate a single hanging CUDA backend without losing the rest
of the compatibility report.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import struct
import sys
import time
import traceback
from typing import Any, Callable

TESTS = (
    "device_info",
    "driver_pci_bus_id",
    "driver_api",
    "driver_launch_ex",
    "driver_pdl_semantics",
    "driver_func_attributes",
    "driver_function_metadata",
    "driver_ptx_selection",
    "driver_buffer_clear",
    "nvml",
    "memory_copy",
    "streams_events",
    "elementwise",
    "reductions",
    "matmul",
    "bf16_matmul",
    "fft",
    "sparse_mm",
    "linalg_solve",
    "linalg_cholesky",
    "linalg_qr",
    "linalg_inv",
    "linalg_lstsq",
    "linalg_svd",
    "linalg_pinv",
    "linalg_eigh",
    "linalg_eigvalsh",
    "softmax",
    "layernorm",
    "gather_scatter",
    "random",
    "autograd_backward",
    "optimizer_sgd",
    "optimizer_adam",
    "amp",
    "conv2d",
    "batch_norm",
    "sdpa_math",
    "sdpa_flash",
    "sdpa_mem_efficient",
    "cuda_graph",
)


def emit(test: str, status: str, **extra: Any) -> None:
    payload = json.dumps({"schema": 1, "test": test, "status": status, **extra}, sort_keys=True)
    # Keep stdout for direct use, plus a marked copy on stderr so device-side
    # printf noise cannot corrupt the machine-readable result.
    print(payload, flush=True)
    print(f"CUDAAMD_RESULT:{payload}", file=sys.stderr, flush=True)


def sync(torch) -> None:
    torch.cuda.synchronize()


def tensor_metrics(torch, got, ref, atol=1e-4, rtol=1e-4) -> dict[str, Any]:
    got = got.detach().float().cpu()
    ref = ref.detach().float().cpu()
    diff = (got - ref).abs()
    finite = bool(torch.isfinite(got).all().item())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    scale = max(float(ref.abs().max().item()) if ref.numel() else 0.0, 1e-7)
    return {
        "ok": finite and bool(torch.allclose(got, ref, atol=atol, rtol=rtol)),
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_abs_over_ref_scale": max_abs / scale,
        "atol": atol,
        "rtol": rtol,
    }


def run_device_info(torch):
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    return {
        "ok": True,
        "name": torch.cuda.get_device_name(0),
        "capability": list(cap),
        "total_memory": int(props.total_memory),
        "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
    }


def _load_win_dll(name: str):
    if not hasattr(ctypes, "WinDLL"):
        raise RuntimeError("WinDLL unavailable on this platform")
    return ctypes.WinDLL(name)


def run_driver_pci_bus_id(torch):
    import ctypes
    import re

    nvcuda = ctypes.WinDLL("nvcuda.dll")
    nvcuda.cuInit.argtypes = [ctypes.c_uint]
    nvcuda.cuInit.restype = ctypes.c_int
    nvcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    nvcuda.cuDeviceGet.restype = ctypes.c_int
    nvcuda.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    nvcuda.cuDeviceGetPCIBusId.restype = ctypes.c_int
    init_rc = nvcuda.cuInit(0)
    dev = ctypes.c_int()
    get_rc = nvcuda.cuDeviceGet(ctypes.byref(dev), 0) if init_rc == 0 else -1
    buf = ctypes.create_string_buffer(64)
    pci_rc = nvcuda.cuDeviceGetPCIBusId(buf, len(buf), dev.value) if get_rc == 0 else -1
    value = buf.value.decode("ascii", errors="replace")
    format_ok = bool(re.fullmatch(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]", value))
    ok = init_rc == 0 and get_rc == 0 and pci_rc == 0 and format_ok
    return {
        "ok": ok,
        "cuInit": init_rc,
        "cuDeviceGet": get_rc,
        "cuDeviceGetPCIBusId": pci_rc,
        "driver_device": dev.value,
        "pci_bus_id": value,
        "format_ok": format_ok,
    }


def run_driver_api(torch):
    cuda = _load_win_dll("nvcuda.dll")
    c_int_p = ctypes.POINTER(ctypes.c_int)
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDriverGetVersion.argtypes = [c_int_p]
    cuda.cuDriverGetVersion.restype = ctypes.c_int
    cuda.cuDeviceGetCount.argtypes = [c_int_p]
    cuda.cuDeviceGetCount.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [c_int_p, ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    cuda.cuDeviceGetName.restype = ctypes.c_int

    def ck(code: int, call: str):
        if code != 0:
            raise RuntimeError(f"{call} returned CUDA error {code}")

    ck(cuda.cuInit(0), "cuInit")
    version = ctypes.c_int()
    count = ctypes.c_int()
    dev = ctypes.c_int()
    ck(cuda.cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion")
    ck(cuda.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    if count.value < 1:
        raise RuntimeError("cuDeviceGetCount returned zero devices")
    ck(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    buf = ctypes.create_string_buffer(256)
    ck(cuda.cuDeviceGetName(buf, len(buf), dev.value), "cuDeviceGetName")
    return {"ok": True, "driver_version": version.value, "device_count": count.value, "device_name": buf.value.decode(errors="replace")}


def run_driver_launch_ex(torch):
    """Exercise cuLaunchKernelEx launch-attribute handling through the driver API.

    CUDA/ZLUDA regressions here matter to newer llama.cpp builds.  The probe
    deliberately uses a no-op PTX kernel so it tests launch semantics rather
    than application math.  PDL=1 is allowed to fail safely with 801 until an
    AMD backend exposes equivalent programmatic-stream-serialization semantics.
    """
    cuda = _load_win_dll("nvcuda.dll")

    class CUlaunchAttributeValue(ctypes.Union):
        _fields_ = [
            ("pad", ctypes.c_byte * 64),
            ("cooperative", ctypes.c_int),
            ("programmaticStreamSerializationAllowed", ctypes.c_int),
        ]

    class CUlaunchAttribute(ctypes.Structure):
        _fields_ = [
            ("id", ctypes.c_uint),
            ("_pad", ctypes.c_byte * 4),
            ("value", CUlaunchAttributeValue),
        ]

    class CUlaunchConfig(ctypes.Structure):
        _fields_ = [
            ("gridDimX", ctypes.c_uint),
            ("gridDimY", ctypes.c_uint),
            ("gridDimZ", ctypes.c_uint),
            ("blockDimX", ctypes.c_uint),
            ("blockDimY", ctypes.c_uint),
            ("blockDimZ", ctypes.c_uint),
            ("sharedMemBytes", ctypes.c_uint),
            ("hStream", ctypes.c_void_p),
            ("attrs", ctypes.POINTER(CUlaunchAttribute)),
            ("numAttrs", ctypes.c_uint),
        ]

    expected_attr_size = 72
    expected_config_size = 56
    actual_attr_size = ctypes.sizeof(CUlaunchAttribute)
    actual_config_size = ctypes.sizeof(CUlaunchConfig)
    if actual_attr_size != expected_attr_size or actual_config_size != expected_config_size:
        raise RuntimeError(
            "cuLaunchKernelEx ABI mismatch: "
            f"CUlaunchAttribute={actual_attr_size} (expected {expected_attr_size}), "
            f"CUlaunchConfig={actual_config_size} (expected {expected_config_size})"
        )

    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuCtxGetCurrent.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda.cuModuleLoadData.restype = ctypes.c_int
    cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    cuda.cuModuleGetFunction.restype = ctypes.c_int
    cuda.cuLaunchKernelEx.argtypes = [ctypes.POINTER(CUlaunchConfig), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    cuda.cuLaunchKernelEx.restype = ctypes.c_int
    cuda.cuCtxSynchronize.argtypes = []
    cuda.cuCtxSynchronize.restype = ctypes.c_int
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuModuleUnload.restype = ctypes.c_int

    def ck(code: int, call: str):
        if code != 0:
            raise RuntimeError(f"{call} returned CUDA error {code}")

    ck(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")

    ctx = ctypes.c_void_p()
    ck(cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    created_ctx = False
    if not ctx.value:
        ck(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value), "cuCtxCreate_v2")
        created_ctx = True

    ptx = b""".version 7.0
.target sm_80
.address_size 64
.visible .entry noop() {
    ret;
}
\0"""
    module = ctypes.c_void_p()
    try:
        ck(
            cuda.cuModuleLoadData(
                ctypes.byref(module),
                ctypes.cast(ctypes.c_char_p(ptx), ctypes.c_void_p),
            ),
            "cuModuleLoadData",
        )
        func = ctypes.c_void_p()
        ck(cuda.cuModuleGetFunction(ctypes.byref(func), module, b"noop"), "cuModuleGetFunction")

        def launch(attr_id=None, value=0):
            attr = CUlaunchAttribute()
            attr_ptr = None
            count = 0
            if attr_id is not None:
                attr.id = attr_id
                if attr_id == 2:
                    attr.value.cooperative = value
                elif attr_id == 6:
                    attr.value.programmaticStreamSerializationAllowed = value
                attr_ptr = ctypes.pointer(attr)
                count = 1
            cfg = CUlaunchConfig(1, 1, 1, 1, 1, 1, 0, None, attr_ptr, count)
            rc = int(cuda.cuLaunchKernelEx(ctypes.byref(cfg), func, None, None))
            sync_rc = int(cuda.cuCtxSynchronize()) if rc == 0 else None
            return {"launch": rc, "sync": sync_rc}

        results = {
            "no_attrs": launch(),
            "cooperative_0": launch(2, 0),
            "cooperative_1": launch(2, 1),
            "pdl_0": launch(6, 0),
            "pdl_1": launch(6, 1),
        }

        normal_ok = results["no_attrs"] == {"launch": 0, "sync": 0}
        cooperative_zero_ok = results["cooperative_0"] == {"launch": 0, "sync": 0}
        pdl_zero_ok = results["pdl_0"] == {"launch": 0, "sync": 0}
        # Until we have a semantic PDL ordering/serialization probe, a clean
        # NOT_SUPPORTED result is the only result we can call safe here.
        # Counting SUCCESS would allow a broken "ignore PDL=1" implementation
        # to pass this regression without proving CUDA-equivalent semantics.
        pdl_one_safe = results["pdl_1"]["launch"] == 801
        return {
            "ok": normal_ok and cooperative_zero_ok and pdl_zero_ok and pdl_one_safe,
            "results": results,
            "abi": {
                "launch_attribute_size": ctypes.sizeof(CUlaunchAttribute),
                "launch_config_size": ctypes.sizeof(CUlaunchConfig),
            },
            "notes": {
                "cooperative_1": "reported separately because device support can vary",
                "pdl_1": "801 is required until a semantic PDL implementation has its own correctness probe",
            },
        }
    finally:
        if module.value:
            try:
                cuda.cuModuleUnload(module)
            except Exception:
                pass
        if created_ctx and ctx.value:
            try:
                cuda.cuCtxDestroy_v2(ctx)
            except Exception:
                pass


def run_driver_pdl_semantics(torch):
    """Validate PDL conservatively instead of accepting a success code alone.

    A backend may safely implement PROGRAMMATIC_STREAM_SERIALIZATION=1 by
    preserving ordinary same-stream serialization (no overlap).  This probe
    uses a producer/consumer pair containing PTX griddepcontrol instructions.
    A clean 801 is reported as unsupported.  If PDL launch is accepted, the
    dependent consumer must repeatedly observe the producer's completed write.
    """
    cuda = _load_win_dll("nvcuda.dll")

    class CUlaunchAttributeValue(ctypes.Union):
        _fields_ = [
            ("pad", ctypes.c_byte * 64),
            ("programmaticStreamSerializationAllowed", ctypes.c_int),
        ]

    class CUlaunchAttribute(ctypes.Structure):
        _fields_ = [
            ("id", ctypes.c_uint),
            ("_pad", ctypes.c_byte * 4),
            ("value", CUlaunchAttributeValue),
        ]

    class CUlaunchConfig(ctypes.Structure):
        _fields_ = [
            ("gridDimX", ctypes.c_uint),
            ("gridDimY", ctypes.c_uint),
            ("gridDimZ", ctypes.c_uint),
            ("blockDimX", ctypes.c_uint),
            ("blockDimY", ctypes.c_uint),
            ("blockDimZ", ctypes.c_uint),
            ("sharedMemBytes", ctypes.c_uint),
            ("hStream", ctypes.c_void_p),
            ("attrs", ctypes.POINTER(CUlaunchAttribute)),
            ("numAttrs", ctypes.c_uint),
        ]

    if ctypes.sizeof(CUlaunchAttribute) != 72 or ctypes.sizeof(CUlaunchConfig) != 56:
        raise RuntimeError(
            "cuLaunchKernelEx ABI mismatch for PDL probe: "
            f"attribute={ctypes.sizeof(CUlaunchAttribute)}, config={ctypes.sizeof(CUlaunchConfig)}"
        )

    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuCtxGetCurrent.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda.cuModuleLoadData.restype = ctypes.c_int
    cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    cuda.cuModuleGetFunction.restype = ctypes.c_int
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuModuleUnload.restype = ctypes.c_int
    cuda.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    cuda.cuStreamCreate.restype = ctypes.c_int
    cuda.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuStreamDestroy_v2.restype = ctypes.c_int
    cuda.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
    cuda.cuStreamSynchronize.restype = ctypes.c_int
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuda.cuMemAlloc_v2.restype = ctypes.c_int
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda.cuMemFree_v2.restype = ctypes.c_int
    cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    cuda.cuMemcpyHtoD_v2.restype = ctypes.c_int
    cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    cuda.cuMemcpyDtoH_v2.restype = ctypes.c_int
    cuda.cuLaunchKernelEx.argtypes = [
        ctypes.POINTER(CUlaunchConfig),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    cuda.cuLaunchKernelEx.restype = ctypes.c_int

    def ck(code: int, call: str):
        if code != 0:
            raise RuntimeError(f"{call} returned CUDA error {code}")

    ck(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    ck(cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    created_ctx = False
    if not ctx.value:
        ck(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value), "cuCtxCreate_v2")
        created_ctx = True

    # The volatile scratch store keeps the producer busy long enough to expose
    # an implementation that incorrectly overlaps launches while treating
    # griddepcontrol as a no-op.  A conservative serialized implementation is
    # valid and should still produce 42 every time.
    ptx = b""".version 8.0
.target sm_90
.address_size 64

.visible .entry pdl_producer(
    .param .u64 data_ptr,
    .param .u64 scratch_ptr
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [data_ptr];
    ld.param.u64 %rd2, [scratch_ptr];
    mov.u32 %r1, 0;
L_pdl_spin:
    add.u32 %r1, %r1, 1;
    st.volatile.global.u32 [%rd2], %r1;
    setp.lt.u32 %p1, %r1, 200000;
    @%p1 bra L_pdl_spin;

    mov.u32 %r2, 41;
    st.global.u32 [%rd1], %r2;
    griddepcontrol.launch_dependents;
    ret;
}

.visible .entry pdl_consumer(
    .param .u64 data_ptr
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [data_ptr];
    griddepcontrol.wait;
    ld.global.u32 %r1, [%rd1];
    add.u32 %r1, %r1, 1;
    st.global.u32 [%rd1], %r1;
    ret;
}
\0"""

    module = ctypes.c_void_p()
    stream = ctypes.c_void_p()
    data_dev = ctypes.c_uint64()
    scratch_dev = ctypes.c_uint64()
    try:
        ck(
            cuda.cuModuleLoadData(
                ctypes.byref(module),
                ctypes.cast(ctypes.c_char_p(ptx), ctypes.c_void_p),
            ),
            "cuModuleLoadData(PDL semantic PTX)",
        )
        producer = ctypes.c_void_p()
        consumer = ctypes.c_void_p()
        ck(cuda.cuModuleGetFunction(ctypes.byref(producer), module, b"pdl_producer"), "cuModuleGetFunction(pdl_producer)")
        ck(cuda.cuModuleGetFunction(ctypes.byref(consumer), module, b"pdl_consumer"), "cuModuleGetFunction(pdl_consumer)")
        ck(cuda.cuStreamCreate(ctypes.byref(stream), 1), "cuStreamCreate")
        ck(cuda.cuMemAlloc_v2(ctypes.byref(data_dev), ctypes.sizeof(ctypes.c_uint32)), "cuMemAlloc(data)")
        ck(cuda.cuMemAlloc_v2(ctypes.byref(scratch_dev), ctypes.sizeof(ctypes.c_uint32)), "cuMemAlloc(scratch)")

        attr = CUlaunchAttribute()
        attr.id = 6  # CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        attr.value.programmaticStreamSerializationAllowed = 1
        attr_ptr = ctypes.pointer(attr)

        def cfg():
            return CUlaunchConfig(1, 1, 1, 1, 1, 1, 0, stream, attr_ptr, 1)

        def params_for(*device_ptrs):
            values = [ctypes.c_uint64(int(ptr.value)) for ptr in device_ptrs]
            array = (ctypes.c_void_p * len(values))(
                *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values]
            )
            return values, array

        attempts = []
        for iteration in range(5):
            zero = ctypes.c_uint32(0)
            ck(
                cuda.cuMemcpyHtoD_v2(
                    data_dev.value,
                    ctypes.cast(ctypes.byref(zero), ctypes.c_void_p),
                    ctypes.sizeof(zero),
                ),
                "cuMemcpyHtoD(data=0)",
            )
            ck(
                cuda.cuMemcpyHtoD_v2(
                    scratch_dev.value,
                    ctypes.cast(ctypes.byref(zero), ctypes.c_void_p),
                    ctypes.sizeof(zero),
                ),
                "cuMemcpyHtoD(scratch=0)",
            )

            producer_values, producer_params = params_for(data_dev, scratch_dev)
            producer_cfg = cfg()
            producer_rc = int(
                cuda.cuLaunchKernelEx(
                    ctypes.byref(producer_cfg),
                    producer,
                    ctypes.cast(producer_params, ctypes.c_void_p),
                    None,
                )
            )
            if producer_rc == 801:
                raise RuntimeError("programmatic dependent launch not supported (CUDA error 801)")
            ck(producer_rc, "cuLaunchKernelEx(pdl_producer)")

            consumer_values, consumer_params = params_for(data_dev)
            consumer_cfg = cfg()
            consumer_rc = int(
                cuda.cuLaunchKernelEx(
                    ctypes.byref(consumer_cfg),
                    consumer,
                    ctypes.cast(consumer_params, ctypes.c_void_p),
                    None,
                )
            )
            if consumer_rc == 801:
                raise RuntimeError("programmatic dependent launch not supported for dependent kernel (CUDA error 801)")
            ck(consumer_rc, "cuLaunchKernelEx(pdl_consumer)")
            ck(cuda.cuStreamSynchronize(stream), "cuStreamSynchronize(PDL pair)")

            data_host = ctypes.c_uint32()
            scratch_host = ctypes.c_uint32()
            ck(
                cuda.cuMemcpyDtoH_v2(
                    ctypes.cast(ctypes.byref(data_host), ctypes.c_void_p),
                    data_dev.value,
                    ctypes.sizeof(data_host),
                ),
                "cuMemcpyDtoH(data)",
            )
            ck(
                cuda.cuMemcpyDtoH_v2(
                    ctypes.cast(ctypes.byref(scratch_host), ctypes.c_void_p),
                    scratch_dev.value,
                    ctypes.sizeof(scratch_host),
                ),
                "cuMemcpyDtoH(scratch)",
            )
            attempts.append(
                {
                    "iteration": iteration,
                    "producer_launch": producer_rc,
                    "consumer_launch": consumer_rc,
                    "data": int(data_host.value),
                    "scratch": int(scratch_host.value),
                    "correct": int(data_host.value) == 42 and int(scratch_host.value) == 200000,
                }
            )

        ok = all(x["correct"] for x in attempts)
        return {
            "ok": ok,
            "mode": "semantic_ordering",
            "expected_data": 42,
            "attempts": attempts,
            "notes": {
                "safe_conservative_backend": "ordinary same-stream serialization is acceptable even without overlap",
                "success_requirement": "PDL launch success counts only when producer/consumer ordering remains correct",
            },
        }
    finally:
        if stream.value:
            try:
                cuda.cuStreamSynchronize(stream)
            except Exception:
                pass
        if data_dev.value:
            try:
                cuda.cuMemFree_v2(data_dev.value)
            except Exception:
                pass
        if scratch_dev.value:
            try:
                cuda.cuMemFree_v2(scratch_dev.value)
            except Exception:
                pass
        if stream.value:
            try:
                cuda.cuStreamDestroy_v2(stream)
            except Exception:
                pass
        if module.value:
            try:
                cuda.cuModuleUnload(module)
            except Exception:
                pass
        if created_ctx and ctx.value:
            try:
                cuda.cuCtxDestroy_v2(ctx)
            except Exception:
                pass




def run_driver_ptx_selection(torch):
    """Check that multi-PTX fatbins honor the advertised CUDA capability."""
    cuda = _load_win_dll("nvcuda.dll")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuCtxGetCurrent.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda.cuModuleLoadData.restype = ctypes.c_int
    cuda.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p
    ]
    cuda.cuModuleGetFunction.restype = ctypes.c_int
    cuda.cuFuncGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p
    ]
    cuda.cuFuncGetAttribute.restype = ctypes.c_int
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuModuleUnload.restype = ctypes.c_int
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuda.cuMemAlloc_v2.restype = ctypes.c_int
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda.cuMemFree_v2.restype = ctypes.c_int
    cuda.cuMemcpyDtoH_v2.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t
    ]
    cuda.cuMemcpyDtoH_v2.restype = ctypes.c_int
    cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    cuda.cuLaunchKernel.restype = ctypes.c_int
    cuda.cuCtxSynchronize.argtypes = []
    cuda.cuCtxSynchronize.restype = ctypes.c_int

    def ck(code: int, call: str):
        if code != 0:
            raise RuntimeError(f"{call} returned CUDA error {code}")

    def ptx(target_sm: int) -> bytes:
        return f""".version 8.4
.target sm_{target_sm}
.address_size 64
.visible .entry ptx_select_probe(.param .u64 output_ptr)
{{
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output_ptr];
    mov.u32 %r1, {target_sm};
    st.global.u32 [%rd1], %r1;
    ret;
}}
\0""".encode("ascii")

    def fatbin_file(target_sm: int) -> bytes:
        payload = ptx(target_sm)
        header = struct.pack(
            "<HHIIIIIIIIIQQQ",
            1, 0x101, 64, len(payload),
            0, 0, 0, 0,
            target_sm, 64, 0,
            0, 0, len(payload),
        )
        if len(header) != 64:
            raise RuntimeError(f"unexpected synthetic fatbin header size {len(header)}")
        return header + payload

    files = fatbin_file(80) + fatbin_file(90)
    fatbin = struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(files)) + files
    image = ctypes.create_string_buffer(fatbin, len(fatbin))

    ck(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    advertised = tuple(int(x) for x in torch.cuda.get_device_capability(0))
    advertised_sm = advertised[0] * 10 + advertised[1]
    available = (80, 90)
    compatible = [sm for sm in available if sm <= advertised_sm]
    if not compatible:
        raise RuntimeError(
            f"no synthetic PTX target <= advertised sm_{advertised_sm}"
        )
    expected = max(compatible)

    ctx = ctypes.c_void_p()
    ck(cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    created_ctx = False
    if not ctx.value:
        ck(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value), "cuCtxCreate_v2")
        created_ctx = True

    module = ctypes.c_void_p()
    device_out = ctypes.c_uint64()
    try:
        load_rc = int(
            cuda.cuModuleLoadData(
                ctypes.byref(module), ctypes.cast(image, ctypes.c_void_p)
            )
        )
        if load_rc != 0:
            return {
                "ok": False,
                "advertised_capability": list(advertised),
                "advertised_sm": advertised_sm,
                "available_targets": list(available),
                "expected_target": expected,
                "module_load": load_rc,
            }

        func = ctypes.c_void_p()
        ck(
            cuda.cuModuleGetFunction(
                ctypes.byref(func), module, b"ptx_select_probe"
            ),
            "cuModuleGetFunction",
        )

        ptx_version = ctypes.c_int()
        binary_version = ctypes.c_int()
        ptx_rc = int(cuda.cuFuncGetAttribute(ctypes.byref(ptx_version), 5, func))
        binary_rc = int(cuda.cuFuncGetAttribute(ctypes.byref(binary_version), 6, func))

        ck(
            cuda.cuMemAlloc_v2(
                ctypes.byref(device_out), ctypes.sizeof(ctypes.c_uint32)
            ),
            "cuMemAlloc",
        )
        ptr_arg = ctypes.c_uint64(device_out.value)
        params = (ctypes.c_void_p * 1)(
            ctypes.cast(ctypes.byref(ptr_arg), ctypes.c_void_p)
        )
        launch_rc = int(
            cuda.cuLaunchKernel(
                func, 1, 1, 1, 1, 1, 1, 0, None,
                ctypes.cast(params, ctypes.c_void_p), None
            )
        )
        sync_rc = int(cuda.cuCtxSynchronize()) if launch_rc == 0 else None
        host = ctypes.c_uint32(0)
        copy_rc = None
        if sync_rc == 0:
            copy_rc = int(
                cuda.cuMemcpyDtoH_v2(
                    ctypes.byref(host),
                    device_out.value,
                    ctypes.sizeof(host),
                )
            )

        selected = int(host.value) if copy_rc == 0 else None
        ok = (
            ptx_rc == 0
            and int(ptx_version.value) == 84
            and binary_rc == 0
            and launch_rc == 0
            and sync_rc == 0
            and copy_rc == 0
            and selected == expected
        )
        return {
            "ok": ok,
            "advertised_capability": list(advertised),
            "advertised_sm": advertised_sm,
            "available_targets": list(available),
            "expected_target": expected,
            "module_load": load_rc,
            "ptx_version": {
                "rc": ptx_rc,
                "value": int(ptx_version.value),
                "expected": 84,
            },
            "binary_version": {
                "rc": binary_rc,
                "value": int(binary_version.value),
                "note": "diagnostic only; execution is the selection oracle",
            },
            "execution": {
                "launch": launch_rc,
                "sync": sync_rc,
                "copy_back": copy_rc,
                "selected_value": selected,
                "expected": expected,
            },
        }
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
        if created_ctx and ctx.value:
            try:
                cuda.cuCtxDestroy_v2(ctx)
            except Exception:
                pass
def run_driver_buffer_clear(torch):
    """Isolate the llama.cpp-style clear/synchronize path at the Driver API level.

    This deliberately avoids llama.cpp and vendor fatbins. It tests whether a
    tiny PTX clear kernel can be loaded, resolved, launched on a non-default
    stream, synchronized, and observed correctly. Running both PTX 7.0/sm_80
    and PTX 8.4/sm_90 helps separate generic ZLUDA JIT/stream execution from
    application-specific embedded device-code selection.
    """
    cuda = _load_win_dll("nvcuda.dll")

    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuCtxGetCurrent.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda.cuModuleLoadData.restype = ctypes.c_int
    cuda.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    cuda.cuModuleGetFunction.restype = ctypes.c_int
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuModuleUnload.restype = ctypes.c_int
    cuda.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    cuda.cuStreamCreate.restype = ctypes.c_int
    cuda.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuStreamDestroy_v2.restype = ctypes.c_int
    cuda.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
    cuda.cuStreamSynchronize.restype = ctypes.c_int
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuda.cuMemAlloc_v2.restype = ctypes.c_int
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda.cuMemFree_v2.restype = ctypes.c_int
    cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    cuda.cuMemcpyHtoD_v2.restype = ctypes.c_int
    cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    cuda.cuMemcpyDtoH_v2.restype = ctypes.c_int
    cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    cuda.cuLaunchKernel.restype = ctypes.c_int

    def ck(code: int, call: str):
        if code != 0:
            raise RuntimeError(f"{call} returned CUDA error {code}")

    ck(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")

    ctx = ctypes.c_void_p()
    ck(cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    created_ctx = False
    if not ctx.value:
        ck(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value), "cuCtxCreate_v2")
        created_ctx = True

    stream = ctypes.c_void_p()
    device = ctypes.c_uint64()
    count = 256
    host_type = ctypes.c_uint32 * count
    initial = host_type(*([0xA5A5A5A5] * count))
    output = host_type()
    try:
        # CU_STREAM_NON_BLOCKING = 1. The gfx1150 llama failure was reported at
        # cudaStreamSynchronize on a non-default stream, so keep this explicit.
        ck(cuda.cuStreamCreate(ctypes.byref(stream), 1), "cuStreamCreate(non-default)")
        ck(cuda.cuMemAlloc_v2(ctypes.byref(device), ctypes.sizeof(initial)), "cuMemAlloc")
        ck(
            cuda.cuMemcpyHtoD_v2(
                device.value,
                ctypes.cast(initial, ctypes.c_void_p),
                ctypes.sizeof(initial),
            ),
            "cuMemcpyHtoD(initial pattern)",
        )

        advertised_capability = tuple(int(x) for x in torch.cuda.get_device_capability(0))
        advertised_sm = advertised_capability[0] * 10 + advertised_capability[1]
        variants = (
            ("ptx70_sm80", "7.0", "sm_80", 80),
            ("ptx84_sm90", "8.4", "sm_90", 90),
        )
        results = {}
        for name, ptx_version, target, target_sm in variants:
            module = ctypes.c_void_p()
            try:
                ptx = f""".version {ptx_version}
.target {target}
.address_size 64

.visible .entry buffer_clear(
    .param .u64 buffer_ptr,
    .param .u32 element_count
)
{{
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [buffer_ptr];
    ld.param.u32 %r1, [element_count];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r2, %r3, %r4, %r2;
    setp.ge.u32 %p, %r2, %r1;
    @%p bra CLEAR_DONE;
    mul.wide.u32 %rd2, %r2, 4;
    add.s64 %rd2, %rd1, %rd2;
    st.global.u32 [%rd2], 0;
CLEAR_DONE:
    ret;
}}
\0""".encode("ascii")

                load_rc = int(
                    cuda.cuModuleLoadData(
                        ctypes.byref(module),
                        ctypes.cast(ctypes.c_char_p(ptx), ctypes.c_void_p),
                    )
                )
                row = {
                    "ptx_version": ptx_version,
                    "target": target,
                    "module_load": load_rc,
                    "function_get": None,
                    "launch": None,
                    "stream_sync": None,
                    "copy_back": None,
                    "all_zero": False,
                    "first_nonzero": None,
                    "expected_compatible": target_sm <= advertised_sm,
                }
                if load_rc != 0:
                    results[name] = row
                    continue

                func = ctypes.c_void_p()
                row["function_get"] = int(
                    cuda.cuModuleGetFunction(
                        ctypes.byref(func), module, b"buffer_clear"
                    )
                )
                if row["function_get"] != 0:
                    results[name] = row
                    continue

                # Refill before each PTX variant so one successful run cannot
                # make a later failed launch look numerically correct.
                ck(
                    cuda.cuMemcpyHtoD_v2(
                        device.value,
                        ctypes.cast(initial, ctypes.c_void_p),
                        ctypes.sizeof(initial),
                    ),
                    f"cuMemcpyHtoD(refill {name})",
                )
                ptr_arg = ctypes.c_uint64(device.value)
                count_arg = ctypes.c_uint32(count)
                params = (ctypes.c_void_p * 2)(
                    ctypes.cast(ctypes.byref(ptr_arg), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(count_arg), ctypes.c_void_p),
                )
                block = 64
                grid = (count + block - 1) // block
                row["launch"] = int(
                    cuda.cuLaunchKernel(
                        func,
                        grid,
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        stream,
                        ctypes.cast(params, ctypes.c_void_p),
                        None,
                    )
                )
                if row["launch"] == 0:
                    row["stream_sync"] = int(cuda.cuStreamSynchronize(stream))
                if row["stream_sync"] == 0:
                    row["copy_back"] = int(
                        cuda.cuMemcpyDtoH_v2(
                            ctypes.cast(output, ctypes.c_void_p),
                            device.value,
                            ctypes.sizeof(output),
                        )
                    )
                if row["copy_back"] == 0:
                    first_nonzero = next(
                        (i for i, value in enumerate(output) if int(value) != 0),
                        None,
                    )
                    row["first_nonzero"] = first_nonzero
                    row["all_zero"] = first_nonzero is None
                results[name] = row
            finally:
                if module.value:
                    try:
                        cuda.cuModuleUnload(module)
                    except Exception:
                        pass

        def row_ok(row):
            if not row["expected_compatible"]:
                return row["module_load"] == 209
            return (
                row["module_load"] == 0
                and row["function_get"] == 0
                and row["launch"] == 0
                and row["stream_sync"] == 0
                and row["copy_back"] == 0
                and row["all_zero"]
            )

        ok = all(row_ok(row) for row in results.values())
        return {
            "ok": ok,
            "buffer_elements": count,
            "advertised_capability": list(advertised_capability),
            "advertised_sm": advertised_sm,
            "stream": "non-default_non-blocking",
            "variants": results,
            "interpretation": (
                "If this passes while llama.cpp still fails in ggml_backend_cuda_buffer_clear, "
                "the remaining boundary is application device-code/fatbin selection rather "
                "than generic PTX clear-kernel launch or stream synchronization."
            ),
        }
    finally:
        if stream.value:
            try:
                cuda.cuStreamSynchronize(stream)
            except Exception:
                pass
        if device.value:
            try:
                cuda.cuMemFree_v2(device.value)
            except Exception:
                pass
        if stream.value:
            try:
                cuda.cuStreamDestroy_v2(stream)
            except Exception:
                pass
        if created_ctx and ctx.value:
            try:
                cuda.cuCtxDestroy_v2(ctx)
            except Exception:
                pass


def run_driver_func_attributes(torch):
    """Validate the dynamic shared-memory function-attribute boundary.

    Recent llama.cpp Flash Attention paths opt in to larger dynamic shared
    memory with cudaFuncSetAttribute.  This probe records the device-advertised
    opt-in ceiling and verifies the driver accepts values at/below it while
    refusing a value beyond it.
    """
    cuda = _load_win_dll("nvcuda.dll")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    cuda.cuDeviceGetAttribute.restype = ctypes.c_int
    cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuCtxGetCurrent.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda.cuModuleLoadData.restype = ctypes.c_int
    cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    cuda.cuModuleGetFunction.restype = ctypes.c_int
    cuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    cuda.cuFuncSetAttribute.restype = ctypes.c_int
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuModuleUnload.restype = ctypes.c_int

    def ck(code: int, call: str):
        if code != 0:
            raise RuntimeError(f"{call} returned CUDA error {code}")

    ck(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")

    # CUDA Driver API enum values:
    #   8  = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    #   81 = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
    #   97 = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    device_attrs = {}
    for attr, name in (
        (8, "max_shared_per_block"),
        (81, "max_shared_per_sm"),
        (97, "max_shared_optin"),
    ):
        value = ctypes.c_int()
        rc = int(cuda.cuDeviceGetAttribute(ctypes.byref(value), attr, dev.value))
        device_attrs[name] = {"attribute": attr, "rc": rc, "value": int(value.value)}

    optin = device_attrs["max_shared_optin"]["value"]
    if device_attrs["max_shared_optin"]["rc"] != 0 or optin <= 0:
        raise RuntimeError("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN is unavailable")

    ctx = ctypes.c_void_p()
    ck(cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    created_ctx = False
    if not ctx.value:
        ck(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value), "cuCtxCreate_v2")
        created_ctx = True

    ptx = b""".version 7.0
.target sm_80
.address_size 64
.visible .entry noop() {
    ret;
}
\0"""
    module = ctypes.c_void_p()
    try:
        ck(
            cuda.cuModuleLoadData(
                ctypes.byref(module),
                ctypes.cast(ctypes.c_char_p(ptx), ctypes.c_void_p),
            ),
            "cuModuleLoadData",
        )
        func = ctypes.c_void_p()
        ck(cuda.cuModuleGetFunction(ctypes.byref(func), module, b"noop"), "cuModuleGetFunction")

        # CUDA function attribute enum:
        #   8 = CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        safe_value = min(optin, 32 * 1024)
        at_limit_rc = int(cuda.cuFuncSetAttribute(func, 8, optin))
        safe_rc = int(cuda.cuFuncSetAttribute(func, 8, safe_value))
        over_value = optin + 16 * 1024
        over_limit_rc = int(cuda.cuFuncSetAttribute(func, 8, over_value))

        ok = (
            device_attrs["max_shared_per_block"]["rc"] == 0
            and device_attrs["max_shared_per_sm"]["rc"] == 0
            and safe_rc == 0
            and at_limit_rc == 0
            and over_limit_rc != 0
        )
        return {
            "ok": ok,
            "device_attributes": device_attrs,
            "func_attribute": "MAX_DYNAMIC_SHARED_SIZE_BYTES",
            "safe_value": safe_value,
            "safe_rc": safe_rc,
            "at_limit_value": optin,
            "at_limit_rc": at_limit_rc,
            "over_limit_value": over_value,
            "over_limit_rc": over_limit_rc,
        }
    finally:
        if module.value:
            try:
                cuda.cuModuleUnload(module)
            except Exception:
                pass
        if created_ctx and ctx.value:
            try:
                cuda.cuCtxDestroy_v2(ctx)
            except Exception:
                pass


def run_driver_function_metadata(torch):
    """Validate CUDA PTX/BINARY function metadata with known source modules.

    CUDA defines CU_FUNC_ATTRIBUTE_PTX_VERSION as PTX ISA major/minor encoded
    as major*10+minor.  It is not the module target SM.  The sm_90 / PTX 8.4
    case mirrors the llama.cpp b10978 false-PDL gate: returning 90 instead of
    84 can incorrectly satisfy ptxVersion >= 90.
    """
    cuda = _load_win_dll("nvcuda.dll")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuCtxGetCurrent.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda.cuModuleLoadData.restype = ctypes.c_int
    cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    cuda.cuModuleGetFunction.restype = ctypes.c_int
    cuda.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
    cuda.cuFuncGetAttribute.restype = ctypes.c_int
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuModuleUnload.restype = ctypes.c_int

    def ck(code: int, call: str):
        if code != 0:
            raise RuntimeError(f"{call} returned CUDA error {code}")

    ck(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")

    ctx = ctypes.c_void_p()
    ck(cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    created_ctx = False
    if not ctx.value:
        ck(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value), "cuCtxCreate_v2")
        created_ctx = True

    device_capability = tuple(int(x) for x in torch.cuda.get_device_capability(0))
    expected_binary_version = device_capability[0] * 10 + device_capability[1]

    def probe_case(ptx_version_text: str, target_sm: int, expected_ptx_version: int, kernel_name: str):
        ptx = f""".version {ptx_version_text}
.target sm_{target_sm}
.address_size 64
.visible .entry {kernel_name}() {{
    ret;
}}
\0""".encode("ascii")
        module = ctypes.c_void_p()
        try:
            ck(
                cuda.cuModuleLoadData(
                    ctypes.byref(module),
                    ctypes.cast(ctypes.c_char_p(ptx), ctypes.c_void_p),
                ),
                f"cuModuleLoadData({ptx_version_text}/sm_{target_sm})",
            )
            func = ctypes.c_void_p()
            ck(
                cuda.cuModuleGetFunction(ctypes.byref(func), module, kernel_name.encode("ascii")),
                f"cuModuleGetFunction({kernel_name})",
            )

            # CUDA function attribute enum values:
            #   5 = CU_FUNC_ATTRIBUTE_PTX_VERSION
            #   6 = CU_FUNC_ATTRIBUTE_BINARY_VERSION
            ptx_version = ctypes.c_int()
            binary_version = ctypes.c_int()
            ptx_rc = int(cuda.cuFuncGetAttribute(ctypes.byref(ptx_version), 5, func))
            binary_rc = int(cuda.cuFuncGetAttribute(ctypes.byref(binary_version), 6, func))
            ptx_ok = ptx_rc == 0 and int(ptx_version.value) == expected_ptx_version
            return {
                "ok": bool(ptx_ok and binary_rc == 0 and binary_version.value >= 0),
                "ptx_source": {
                    "version": ptx_version_text,
                    "target": f"sm_{target_sm}",
                },
                "ptx_version": {
                    "rc": ptx_rc,
                    "value": int(ptx_version.value),
                    "expected": expected_ptx_version,
                    "matches_target_sm_by_accident": int(ptx_version.value) == target_sm,
                },
                "binary_version": {
                    "rc": binary_rc,
                    "value": int(binary_version.value),
                    "device_capability_value": expected_binary_version,
                    "matches_device_capability": int(binary_version.value) == expected_binary_version,
                },
            }
        finally:
            if module.value:
                try:
                    cuda.cuModuleUnload(module)
                except Exception:
                    pass

    try:
        baseline = probe_case("7.0", 80, 70, "metadata_probe_70")
        llama_gate = probe_case("8.4", 80, 84, "metadata_probe_84")
        false_pdl_gate = bool(
            llama_gate["ptx_version"]["value"] >= 90
            and llama_gate["ptx_version"]["expected"] < 90
        )
        return {
            "ok": bool(baseline["ok"] and llama_gate["ok"] and not false_pdl_gate),
            # Keep the original top-level fields for report consumers that
            # predate the multi-case regression.
            "ptx_source": baseline["ptx_source"],
            "device_capability": list(device_capability),
            "ptx_version": baseline["ptx_version"],
            "binary_version": baseline["binary_version"],
            "cases": {
                "baseline_ptx70_sm80": baseline,
                "llama_b10978_ptx84_sm80": llama_gate,
            },
            "llama_b10978_false_pdl_gate": false_pdl_gate,
            "notes": {
                "semantic_regression": "PTX_VERSION must describe PTX ISA version, not target SM",
                "llama_b10978": "Selected PTX 8.4 must report 84 independently of target SM; reporting target SM can falsely enable ptxVersion>=90 PDL dispatch",
                "binary_version": "recorded separately because applications may use it for architecture gating; mismatch is diagnostic until validated",
            },
        }
    finally:
        if created_ctx and ctx.value:
            try:
                cuda.cuCtxDestroy_v2(ctx)
            except Exception:
                pass


def run_nvml(torch):
    nvml = _load_win_dll("nvml.dll")
    init = getattr(nvml, "nvmlInit_v2", getattr(nvml, "nvmlInit", None))
    count_fn = getattr(nvml, "nvmlDeviceGetCount_v2", getattr(nvml, "nvmlDeviceGetCount", None))
    handle_fn = getattr(nvml, "nvmlDeviceGetHandleByIndex_v2", None)
    name_fn = getattr(nvml, "nvmlDeviceGetName", None)
    memory_fn = getattr(nvml, "nvmlDeviceGetMemoryInfo", None)
    shutdown = getattr(nvml, "nvmlShutdown", None)
    if init is None or count_fn is None or handle_fn is None or name_fn is None or memory_fn is None:
        raise RuntimeError("required NVML entry points are not exported")

    class NvmlMemory(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    init.restype = ctypes.c_int
    count_fn.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    count_fn.restype = ctypes.c_int
    handle_fn.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    handle_fn.restype = ctypes.c_int
    name_fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
    name_fn.restype = ctypes.c_int
    memory_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(NvmlMemory)]
    memory_fn.restype = ctypes.c_int

    rc = int(init())
    if rc != 0:
        raise RuntimeError(f"nvmlInit returned {rc}")
    try:
        count = ctypes.c_uint()
        rc = int(count_fn(ctypes.byref(count)))
        if rc != 0:
            raise RuntimeError(f"nvmlDeviceGetCount returned {rc}")
        if count.value < 1:
            raise RuntimeError("NVML reported zero devices")

        device = ctypes.c_void_p()
        rc = int(handle_fn(0, ctypes.byref(device)))
        if rc != 0 or not device.value:
            raise RuntimeError(f"nvmlDeviceGetHandleByIndex_v2 returned {rc}")

        name_buf = ctypes.create_string_buffer(256)
        rc = int(name_fn(device, name_buf, len(name_buf)))
        if rc != 0:
            raise RuntimeError(f"nvmlDeviceGetName returned {rc}")

        memory = NvmlMemory()
        rc = int(memory_fn(device, ctypes.byref(memory)))
        if rc != 0:
            raise RuntimeError(f"nvmlDeviceGetMemoryInfo returned {rc}")

        name = name_buf.value.decode(errors="replace")
        total = int(memory.total)
        free = int(memory.free)
        used = int(memory.used)
        sane_memory = total > 0 and 0 <= free <= total and 0 <= used <= total and abs((free + used) - total) <= max(4096, total // 1000)
        return {
            "ok": bool(name) and sane_memory,
            "device_count": int(count.value),
            "device_name": name,
            "memory": {"total": total, "free": free, "used": used},
        }
    finally:
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                pass
def run_memory_copy(torch):
    torch.manual_seed(101)
    x = torch.randn(1024, 256)
    g = x.cuda()
    y = torch.empty_like(g)
    y.copy_(g)
    out = y.cpu()
    sync(torch)
    check = tensor_metrics(torch, out, x, 0.0, 0.0)
    return {"ok": check["ok"], "numerics": check}


def run_streams_events(torch):
    x = torch.arange(4096, dtype=torch.float32, device="cuda")
    producer = torch.cuda.current_stream()
    s = torch.cuda.Stream()
    # Explicitly establish the dependency. Reading a tensor produced on the
    # default stream from another stream without this wait is a race even on
    # native CUDA and would make this a bad compatibility probe.
    s.wait_stream(producer)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(s):
        start.record(s)
        y = x * 2.0 + 1.0
        end.record(s)
    s.synchronize()
    elapsed = float(start.elapsed_time(end))
    ref = torch.arange(4096, dtype=torch.float32) * 2.0 + 1.0
    check = tensor_metrics(torch, y, ref, 0.0, 0.0)
    return {"ok": check["ok"] and elapsed >= 0.0, "event_elapsed_ms": elapsed, "numerics": check}


def run_elementwise(torch):
    x = torch.linspace(-2.0, 2.0, 4096, dtype=torch.float32)
    ref = torch.sin(x) + torch.cos(x * 0.7) + torch.tanh(x) + torch.exp(x * 0.1)
    g = x.cuda()
    got = torch.sin(g) + torch.cos(g * 0.7) + torch.tanh(g) + torch.exp(g * 0.1)
    sync(torch)
    check = tensor_metrics(torch, got, ref, 3e-5, 3e-5)
    return {"ok": check["ok"], "numerics": check}


def run_reductions(torch):
    torch.manual_seed(102)
    x = torch.randn(256, 257)
    ref = torch.stack((x.sum(), x.mean(), x.abs().max(), torch.linalg.vector_norm(x)))
    g = x.cuda()
    got = torch.stack((g.sum(), g.mean(), g.abs().max(), torch.linalg.vector_norm(g)))
    sync(torch)
    check = tensor_metrics(torch, got, ref, 2e-3, 2e-4)
    return {"ok": check["ok"], "numerics": check}


def run_matmul(torch):
    torch.manual_seed(103)
    results = {}
    checks = []
    specs = [
        ("fp32", torch.float32, 6e-3, 7e-4),
        ("fp64", torch.float64, 1e-10, 1e-10),
        ("complex64", torch.complex64, 8e-3, 1e-3),
        ("complex128", torch.complex128, 2e-10, 2e-10),
    ]
    for name, dtype, atol, rtol in specs:
        base_dtype = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
        if dtype.is_complex:
            ar = torch.randn(96, 128, dtype=base_dtype)
            ai = torch.randn(96, 128, dtype=base_dtype)
            br = torch.randn(128, 80, dtype=base_dtype)
            bi = torch.randn(128, 80, dtype=base_dtype)
            a = torch.complex(ar, ai).to(dtype)
            b = torch.complex(br, bi).to(dtype)
        else:
            a = torch.randn(96, 128, dtype=dtype)
            b = torch.randn(128, 80, dtype=dtype)
        ref = a @ b
        got = a.cuda() @ b.cuda()
        sync(torch)
        check = tensor_metrics(torch, got, ref, atol, rtol)
        results[name] = check
        checks.append(check)

    # Keep the historical FP16 check as a separate mixed-precision path.
    a16 = torch.randn(128, 128).half()
    b16 = torch.randn(128, 128).half()
    ref16 = a16.float() @ b16.float()
    got16 = a16.cuda() @ b16.cuda()
    sync(torch)
    c16 = tensor_metrics(torch, got16, ref16, 9e-2, 2e-2)
    results["fp16"] = c16
    checks.append(c16)
    return {"ok": all(c["ok"] for c in checks), "dtypes": results}


def run_bf16_matmul(torch):
    if not hasattr(torch, "bfloat16"):
        raise RuntimeError("bfloat16 is not available in this PyTorch build")
    torch.manual_seed(104)
    a = torch.randn(128, 128)
    b = torch.randn(128, 128)
    ref = a @ b
    got = a.to(torch.bfloat16).cuda() @ b.to(torch.bfloat16).cuda()
    sync(torch)
    check = tensor_metrics(torch, got, ref, 0.7, 5e-2)
    return {"ok": check["ok"], "numerics": check}


def run_fft(torch):
    torch.manual_seed(105)

    # FP32 real <-> complex, batched 1D.
    x = torch.randn(8, 1024, dtype=torch.float32)
    ref = torch.fft.rfft(x, dim=-1)
    got = torch.fft.rfft(x.cuda(), dim=-1)
    sync(torch)
    r32 = tensor_metrics(torch, got.real, ref.real, 3e-3, 5e-4)
    i32 = tensor_metrics(torch, got.imag, ref.imag, 3e-3, 5e-4)
    inv = torch.fft.irfft(got, n=x.shape[-1], dim=-1)
    sync(torch)
    inv32 = tensor_metrics(torch, inv, x, 3e-4, 3e-4)

    # Complex-to-complex exercises C2C and inverse direction.
    zr = torch.randn(4, 256, dtype=torch.float32)
    zi = torch.randn(4, 256, dtype=torch.float32)
    z = torch.complex(zr, zi)
    zref = torch.fft.fft(z, dim=-1)
    zgot = torch.fft.fft(z.cuda(), dim=-1)
    sync(torch)
    c2c_r = tensor_metrics(torch, zgot.real, zref.real, 4e-3, 7e-4)
    c2c_i = tensor_metrics(torch, zgot.imag, zref.imag, 4e-3, 7e-4)
    zinv = torch.fft.ifft(zgot, dim=-1)
    sync(torch)
    c2c_inv_r = tensor_metrics(torch, zinv.real, z.real, 5e-4, 5e-4)
    c2c_inv_i = tensor_metrics(torch, zinv.imag, z.imag, 5e-4, 5e-4)

    # FP64 real <-> complex exercises D2Z / Z2D.
    xd = torch.randn(2, 256, dtype=torch.float64)
    dref = torch.fft.rfft(xd, dim=-1)
    dgot = torch.fft.rfft(xd.cuda(), dim=-1)
    sync(torch)
    d_r = tensor_metrics(torch, dgot.real, dref.real, 1e-8, 1e-8)
    d_i = tensor_metrics(torch, dgot.imag, dref.imag, 1e-8, 1e-8)
    dinv = torch.fft.irfft(dgot, n=xd.shape[-1], dim=-1)
    sync(torch)
    d_inv = tensor_metrics(torch, dinv, xd, 1e-9, 1e-9)

    # Rank-2 plan exercises multi-dimensional Xt planning.
    x2 = torch.randn(2, 32, 64, dtype=torch.float32)
    ref2 = torch.fft.rfft2(x2, dim=(-2, -1))
    got2 = torch.fft.rfft2(x2.cuda(), dim=(-2, -1))
    sync(torch)
    r2 = tensor_metrics(torch, got2.real, ref2.real, 5e-3, 8e-4)
    i2 = tensor_metrics(torch, got2.imag, ref2.imag, 5e-3, 8e-4)
    inv2 = torch.fft.irfft2(got2, s=x2.shape[-2:], dim=(-2, -1))
    sync(torch)
    inv2c = tensor_metrics(torch, inv2, x2, 5e-4, 5e-4)

    checks = [r32, i32, inv32, c2c_r, c2c_i, c2c_inv_r, c2c_inv_i, d_r, d_i, d_inv, r2, i2, inv2c]
    return {
        "ok": all(c["ok"] for c in checks),
        "fp32_r2c": {"real": r32, "imag": i32, "inverse": inv32},
        "complex64_c2c": {"real": c2c_r, "imag": c2c_i, "inverse_real": c2c_inv_r, "inverse_imag": c2c_inv_i},
        "fp64_d2z": {"real": d_r, "imag": d_i, "inverse": d_inv},
        "rank2_r2c": {"real": r2, "imag": i2, "inverse": inv2c},
    }


def run_sparse_mm(torch):
    indices = torch.tensor([[0, 0, 1, 2, 3, 3], [0, 3, 1, 2, 0, 3]], dtype=torch.long)
    values = torch.tensor([1.0, -2.0, 3.0, 4.0, 0.5, 2.5])
    sp = torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()
    dense = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10.0
    ref = torch.sparse.mm(sp, dense)
    got = torch.sparse.mm(sp.cuda(), dense.cuda())
    sync(torch)
    check = tensor_metrics(torch, got, ref, 1e-5, 1e-5)
    return {"ok": check["ok"], "numerics": check}


def run_linalg_solve(torch):
    torch.manual_seed(106)
    cases = []
    specs = [
        ("fp32", torch.float32, 3e-3, 3e-3),
        ("fp64", torch.float64, 1e-8, 1e-8),
        ("complex64", torch.complex64, 5e-3, 5e-3),
        ("complex128", torch.complex128, 2e-8, 2e-8),
    ]
    results = {}
    for name, dtype, atol, rtol in specs:
        if dtype.is_complex:
            base = torch.randn(24, 24, dtype=torch.float64)
            imag = torch.randn(24, 24, dtype=torch.float64)
            m = (base + 1j * imag).to(dtype)
            br = torch.randn(24, 4, dtype=torch.float64)
            bi = torch.randn(24, 4, dtype=torch.float64)
            b = (br + 1j * bi).to(dtype)
            a = m.mH @ m + torch.eye(24, dtype=dtype) * 0.5
        else:
            m = torch.randn(24, 24, dtype=dtype)
            a = m.T @ m + torch.eye(24, dtype=dtype) * 0.5
            b = torch.randn(24, 4, dtype=dtype)
        ref = torch.linalg.solve(a, b)
        got = torch.linalg.solve(a.cuda(), b.cuda())
        sync(torch)
        check = tensor_metrics(torch, got, ref, atol, rtol)
        results[name] = check
        cases.append(check)
    return {"ok": all(c["ok"] for c in cases), "dtypes": results}


def run_linalg_cholesky(torch):
    torch.manual_seed(116)
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except Exception:
        pass
    specs = [
        ("fp32", torch.float32, 3e-4, 3e-4),
        ("fp64", torch.float64, 1e-10, 1e-10),
        ("complex64", torch.complex64, 5e-4, 5e-4),
        ("complex128", torch.complex128, 2e-10, 2e-10),
    ]
    results = {}
    checks = []
    for name, dtype, atol, rtol in specs:
        dtype_results = {}
        for batched in (False, True):
            n = 8
            shape = (3, n, n) if batched else (n, n)
            if dtype.is_complex:
                base_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
                xr = torch.randn(*shape, dtype=base_dtype)
                xi = torch.randn(*shape, dtype=base_dtype)
                x = torch.complex(xr, xi).to(dtype)
            else:
                x = torch.randn(*shape, dtype=dtype)
            a = x @ x.mH + torch.eye(n, dtype=dtype) * (n + 1)
            l_ref = torch.linalg.cholesky(a)
            l_gpu = torch.linalg.cholesky(a.cuda())
            sync(torch)
            l_check = tensor_metrics(torch, l_gpu, l_ref, atol, rtol)

            bshape = (3, n, 2) if batched else (n, 2)
            if dtype.is_complex:
                base_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
                br = torch.randn(*bshape, dtype=base_dtype)
                bi = torch.randn(*bshape, dtype=base_dtype)
                b = torch.complex(br, bi).to(dtype)
            else:
                b = torch.randn(*bshape, dtype=dtype)
            solve_ref = torch.cholesky_solve(b, l_ref)
            solve_gpu = torch.cholesky_solve(b.cuda(), l_gpu)
            sync(torch)
            solve_check = tensor_metrics(torch, solve_gpu, solve_ref, atol, rtol)

            inv_ref = torch.cholesky_inverse(l_ref)
            inv_gpu = torch.cholesky_inverse(l_gpu)
            sync(torch)
            inv_check = tensor_metrics(torch, inv_gpu, inv_ref, atol, rtol)

            case_ok = l_check["ok"] and solve_check["ok"] and inv_check["ok"]
            dtype_results["batch" if batched else "single"] = {
                "ok": case_ok,
                "cholesky": l_check,
                "solve": solve_check,
                "inverse": inv_check,
            }
            checks.extend((l_check, solve_check, inv_check))
        results[name] = dtype_results
    return {"ok": all(c["ok"] for c in checks), "dtypes": results, "backend": "cusolver"}



def run_linalg_qr(torch):
    torch.manual_seed(117)
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except Exception:
        pass

    specs = [
        ("fp32", torch.float32, 8e-4),
        ("fp64", torch.float64, 5e-9),
        ("complex64", torch.complex64, 1e-3),
        ("complex128", torch.complex128, 8e-9),
    ]
    results = {}
    all_ok = True
    for name, dtype, tol in specs:
        dtype_rows = []
        base_dtype = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
        for shape in ((12, 7), (3, 12, 7)):
            if dtype.is_complex:
                x = torch.complex(
                    torch.randn(*shape, dtype=base_dtype),
                    torch.randn(*shape, dtype=base_dtype),
                ).to(dtype)
            else:
                x = torch.randn(*shape, dtype=dtype)
            xg = x.cuda()
            q, r = torch.linalg.qr(xg, mode="reduced")
            sync(torch)
            recon = q @ r
            adj = q.mH if dtype.is_complex else q.transpose(-2, -1)
            k = q.shape[-1]
            eye = torch.eye(k, dtype=dtype, device="cuda").expand(*q.shape[:-2], k, k)
            orth = adj @ q
            sync(torch)
            rec_check = tensor_metrics(torch, recon, x, tol, tol)
            orth_check = tensor_metrics(torch, orth, eye.cpu(), tol, tol)
            row_ok = rec_check["ok"] and orth_check["ok"]
            dtype_rows.append({"shape": list(shape), "reconstruction": rec_check, "orthogonality": orth_check, "ok": row_ok})
            all_ok = all_ok and row_ok

        # Explicit Householder generation exercises legacy geqrf + orgqr/ungqr.
        shape = (12, 7)
        if dtype.is_complex:
            x = torch.complex(
                torch.randn(*shape, dtype=base_dtype),
                torch.randn(*shape, dtype=base_dtype),
            ).to(dtype)
        else:
            x = torch.randn(*shape, dtype=dtype)
        ag, tau = torch.geqrf(x.cuda())
        qg = torch.orgqr(ag, tau)
        sync(torch)
        r = torch.triu(ag[: shape[1], :])
        householder_check = tensor_metrics(torch, qg @ r, x, tol, tol)

        # Apply Q through ormqr/unmqr and compare with a CPU reference generated
        # from the exact GPU reflectors, avoiding QR sign-convention ambiguity.
        other = torch.randn(shape[0], 3, dtype=base_dtype)
        if dtype.is_complex:
            other = torch.complex(other, torch.randn_like(other)).to(dtype)
        else:
            other = other.to(dtype)
        ag_cpu, tau_cpu = ag.cpu(), tau.cpu()
        orm_gpu = torch.ormqr(ag, tau, other.cuda(), left=True, transpose=True)
        sync(torch)
        orm_ref = torch.ormqr(ag_cpu, tau_cpu, other, left=True, transpose=True)
        orm_check = tensor_metrics(torch, orm_gpu, orm_ref, tol * 4, tol * 4)
        dtype_ok = all(r["ok"] for r in dtype_rows) and householder_check["ok"] and orm_check["ok"]
        all_ok = all_ok and dtype_ok
        results[name] = {
            "qr": dtype_rows,
            "geqrf_orgqr": householder_check,
            "ormqr": orm_check,
            "ok": dtype_ok,
        }
    return {"ok": all_ok, "dtypes": results}



def _linalg_rand(torch, shape, dtype):
    if dtype.is_complex:
        base = torch.float64 if dtype == torch.complex128 else torch.float32
        return torch.complex(
            torch.randn(*shape, dtype=base),
            torch.randn(*shape, dtype=base),
        ).to(dtype)
    return torch.randn(*shape, dtype=dtype)


def run_linalg_inv(torch):
    torch.manual_seed(118)
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except Exception:
        pass
    specs = [
        ("fp32", torch.float32, 3e-4),
        ("fp64", torch.float64, 3e-9),
        ("complex64", torch.complex64, 4e-4),
        ("complex128", torch.complex128, 4e-9),
    ]
    results = {}
    all_ok = True
    for name, dtype, tol in specs:
        rows = []
        for shape in ((10, 10), (3, 10, 10)):
            x = _linalg_rand(torch, shape, dtype)
            adj = x.mH if dtype.is_complex else x.transpose(-2, -1)
            eye = torch.eye(10, dtype=dtype).expand(*shape[:-2], 10, 10)
            a = x @ adj + eye * 2
            ref = torch.linalg.inv(a)
            got = torch.linalg.inv(a.cuda())
            sync(torch)
            max_abs = float((got.cpu() - ref).abs().max())
            ident = torch.eye(10, dtype=dtype, device="cuda").expand(*shape[:-2], 10, 10)
            residual = float((got @ a.cuda() - ident).abs().max().cpu())
            ok = max_abs <= tol and residual <= tol * 5
            rows.append({"shape": list(shape), "max_abs": max_abs, "residual": residual, "ok": ok})
            all_ok = all_ok and ok
        results[name] = {"cases": rows, "ok": all(r["ok"] for r in rows)}
    return {"ok": all_ok, "dtypes": results}


def run_linalg_lstsq(torch):
    torch.manual_seed(119)
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except Exception:
        pass
    specs = [
        ("fp32", torch.float32, 4e-4),
        ("fp64", torch.float64, 4e-9),
        ("complex64", torch.complex64, 5e-4),
        ("complex128", torch.complex128, 5e-9),
    ]
    results = {}
    all_ok = True
    for name, dtype, tol in specs:
        rows = []
        for a_shape, b_shape in (((14, 8), (14, 3)), ((3, 14, 8), (3, 14, 3))):
            a = _linalg_rand(torch, a_shape, dtype)
            b = _linalg_rand(torch, b_shape, dtype)
            eye = torch.eye(8, dtype=dtype).expand(*a_shape[:-2], 8, 8)
            a = a.clone()
            a[..., :8, :] += eye * 2
            ref = torch.linalg.lstsq(a, b).solution
            got = torch.linalg.lstsq(a.cuda(), b.cuda()).solution
            sync(torch)
            max_abs = float((got.cpu() - ref).abs().max())
            gpu_residual = float(torch.linalg.vector_norm(a.cuda() @ got - b.cuda()).cpu())
            cpu_residual = float(torch.linalg.vector_norm(a @ ref - b))
            residual_delta = abs(gpu_residual - cpu_residual)
            ok = max_abs <= tol * 15 and residual_delta <= max(tol * 100, abs(cpu_residual) * 5e-4)
            rows.append({
                "a_shape": list(a_shape),
                "b_shape": list(b_shape),
                "max_abs": max_abs,
                "gpu_residual": gpu_residual,
                "cpu_residual": cpu_residual,
                "residual_delta": residual_delta,
                "ok": ok,
            })
            all_ok = all_ok and ok
        results[name] = {"cases": rows, "ok": all(r["ok"] for r in rows)}
    return {"ok": all_ok, "dtypes": results}


def run_linalg_svd(torch):
    torch.manual_seed(120)
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except Exception:
        pass
    cases = []
    for label, shape in (("batched_small", (2, 7, 5)), ("single_large", (64, 40))):
        x = torch.randn(*shape, dtype=torch.float32)
        u, s, vh = torch.linalg.svd(x.cuda(), full_matrices=False)
        sync(torch)
        rec = (u @ torch.diag_embed(s) @ vh).cpu()
        max_abs = float((rec - x).abs().max())
        ok = max_abs <= 5e-4
        cases.append({"case": label, "shape": list(shape), "max_abs": max_abs, "ok": ok})
    return {"ok": all(c["ok"] for c in cases), "cases": cases}


def run_linalg_pinv(torch):
    torch.manual_seed(121)
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except Exception:
        pass
    a = torch.randn(2, 9, 5, dtype=torch.float32)
    ref = torch.linalg.pinv(a)
    got = torch.linalg.pinv(a.cuda())
    sync(torch)
    max_abs = float((got.cpu() - ref).abs().max())
    residual = float((a.cuda() @ got @ a.cuda() - a.cuda()).abs().max().cpu())
    ok = max_abs <= 8e-4 and residual <= 2e-3
    return {"ok": ok, "shape": list(a.shape), "max_abs": max_abs, "residual": residual}


def run_linalg_eigh(torch):
    torch.manual_seed(122)
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except Exception:
        pass
    cases = []
    for name, dtype, shape, tol in (
        ("fp32_single", torch.float32, (8, 8), 3e-3),
        ("complex128_batch", torch.complex128, (2, 8, 8), 2e-9),
    ):
        z = _linalg_rand(torch, shape, dtype)
        a = (z + z.mH) / 2
        w, v = torch.linalg.eigh(a.cuda())
        sync(torch)
        lam = w.to(dtype).unsqueeze(-2)
        residual = float((a.cuda() @ v - v * lam).abs().max().cpu())
        ok = residual <= tol
        cases.append({"case": name, "shape": list(shape), "residual": residual, "ok": ok})
    return {"ok": all(c["ok"] for c in cases), "cases": cases}


def run_linalg_eigvalsh(torch):
    torch.manual_seed(123)
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except Exception:
        pass
    cases = []
    for name, dtype, shape, tol in (
        ("fp32_batch", torch.float32, (2, 8, 8), 3e-3),
        ("complex128_single", torch.complex128, (8, 8), 2e-9),
    ):
        z = _linalg_rand(torch, shape, dtype)
        a = (z + z.mH) / 2
        ref = torch.linalg.eigvalsh(a)
        got = torch.linalg.eigvalsh(a.cuda())
        sync(torch)
        max_abs = float((got.cpu() - ref).abs().max())
        ok = max_abs <= tol
        cases.append({"case": name, "shape": list(shape), "max_abs": max_abs, "ok": ok})
    return {"ok": all(c["ok"] for c in cases), "cases": cases}


def run_softmax(torch):
    import torch.nn.functional as F
    torch.manual_seed(107)
    x = torch.randn(32, 257)
    ref = F.softmax(x, dim=-1)
    got = F.softmax(x.cuda(), dim=-1)
    sync(torch)
    check = tensor_metrics(torch, got, ref, 2e-5, 2e-5)
    return {"ok": check["ok"], "numerics": check}


def run_layernorm(torch):
    import torch.nn.functional as F
    torch.manual_seed(108)
    x = torch.randn(16, 64, 128)
    w = torch.randn(128)
    b = torch.randn(128)
    ref = F.layer_norm(x, (128,), w, b)
    got = F.layer_norm(x.cuda(), (128,), w.cuda(), b.cuda())
    sync(torch)
    check = tensor_metrics(torch, got, ref, 3e-4, 3e-4)
    return {"ok": check["ok"], "numerics": check}


def run_gather_scatter(torch):
    torch.manual_seed(109)
    x = torch.randn(32, 64)
    idx = torch.randint(0, 64, (32, 64), dtype=torch.long)
    ref_g = torch.gather(x, 1, idx)
    base = torch.zeros_like(x)
    ref_s = base.scatter_add(1, idx, x)
    gg = torch.gather(x.cuda(), 1, idx.cuda())
    gs = torch.zeros_like(x.cuda()).scatter_add(1, idx.cuda(), x.cuda())
    sync(torch)
    a = tensor_metrics(torch, gg, ref_g, 0.0, 0.0)
    b = tensor_metrics(torch, gs, ref_s, 1e-6, 1e-6)
    return {"ok": a["ok"] and b["ok"], "gather": a, "scatter_add": b}


def run_random(torch):
    torch.cuda.manual_seed_all(123456)
    a = torch.randn(8192, device="cuda")
    sync(torch)
    torch.cuda.manual_seed_all(123456)
    b = torch.randn(8192, device="cuda")
    sync(torch)
    same = bool(torch.equal(a, b))
    finite = bool(torch.isfinite(a).all().item())
    mean = float(a.float().mean().item())
    std = float(a.float().std().item())
    ok = same and finite and abs(mean) < 0.08 and 0.85 < std < 1.15
    return {"ok": ok, "deterministic_after_reseed": same, "finite": finite, "mean": mean, "std": std}


def run_autograd_backward(torch):
    torch.manual_seed(110)
    x = torch.randn(128, 32)
    w = torch.randn(32, 8, requires_grad=True)
    ref_loss = ((x @ w) ** 2).mean()
    ref_loss.backward()
    ref_grad = w.grad.detach().clone()

    wg = w.detach().cuda().requires_grad_(True)
    xg = x.cuda()
    loss = ((xg @ wg) ** 2).mean()
    loss.backward()
    sync(torch)
    check = tensor_metrics(torch, wg.grad, ref_grad, 3e-3, 3e-3)
    return {"ok": check["ok"], "numerics": check, "loss": float(loss.detach().cpu().item())}


def run_optimizer_sgd(torch):
    torch.manual_seed(1101)
    x = torch.randn(128, 32, device="cuda")
    target = torch.randn(128, 8, device="cuda")
    w = torch.randn(32, 8, device="cuda", requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.01)
    losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = ((x @ w - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    sync(torch)
    finite = all(math.isfinite(v) for v in losses)
    return {"ok": finite and losses[-1] < losses[0], "losses": losses}


def run_optimizer_adam(torch):
    torch.manual_seed(1102)
    x = torch.randn(128, 32, device="cuda")
    target = torch.randn(128, 8, device="cuda")
    w = torch.randn(32, 8, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.01)
    losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = ((x @ w - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    sync(torch)
    finite = all(math.isfinite(v) for v in losses)
    return {"ok": finite and losses[-1] < losses[0], "losses": losses}


def run_amp(torch):
    import torch.nn.functional as F
    torch.manual_seed(111)
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        got = F.relu(a @ b)
    sync(torch)
    finite = bool(torch.isfinite(got).all().item())
    return {"ok": finite, "dtype": str(got.dtype), "finite": finite}


def run_conv2d(torch):
    import torch.nn.functional as F
    torch.manual_seed(112)
    cudnn_enabled = bool(torch.backends.cudnn.enabled)
    x = torch.randn(2, 8, 32, 32)
    w = torch.randn(16, 8, 3, 3)
    b = torch.randn(16)
    ref = F.conv2d(x, w, b, padding=1)
    got = F.conv2d(x.cuda(), w.cuda(), b.cuda(), padding=1)
    sync(torch)
    check = tensor_metrics(torch, got, ref, 8e-3, 8e-4)
    return {
        "ok": check["ok"],
        "numerics": check,
        "cudnn_enabled": cudnn_enabled,
        "path": "cudnn_or_bridge" if cudnn_enabled else "pytorch_no_cudnn_fallback",
    }



def run_batch_norm(torch):
    import torch.nn.functional as F
    torch.manual_seed(118)
    if not torch.backends.cudnn.enabled:
        return {"ok": False, "reason": "cudnn_disabled"}

    x = torch.randn(2, 8, 8, 8, requires_grad=True)
    w = torch.randn(8, requires_grad=True)
    b = torch.randn(8, requires_grad=True)
    ref = F.batch_norm(x, None, None, w, b, True, 0.1, 1e-5)
    ref.square().mean().backward()
    refs = (ref.detach(), x.grad.clone(), w.grad.clone(), b.grad.clone())

    gx = x.detach().cuda().requires_grad_(True)
    gw = w.detach().cuda().requires_grad_(True)
    gb = b.detach().cuda().requires_grad_(True)
    got = F.batch_norm(gx, None, None, gw, gb, True, 0.1, 1e-5)
    got.square().mean().backward()
    sync(torch)
    train = tensor_metrics(torch, got, refs[0], 5e-4, 5e-4)
    grad_x = tensor_metrics(torch, gx.grad, refs[1], 5e-4, 5e-4)
    grad_w = tensor_metrics(torch, gw.grad, refs[2], 5e-4, 5e-4)
    grad_b = tensor_metrics(torch, gb.grad, refs[3], 5e-4, 5e-4)

    running_mean = torch.randn(8)
    running_var = torch.rand(8) + 0.5
    ref_inf = F.batch_norm(x.detach(), running_mean, running_var, w.detach(), b.detach(), False, 0.1, 1e-5)
    got_inf = F.batch_norm(gx.detach(), running_mean.cuda(), running_var.cuda(), gw.detach(), gb.detach(), False, 0.1, 1e-5)
    sync(torch)
    inference = tensor_metrics(torch, got_inf, ref_inf, 5e-4, 5e-4)
    checks = (train, grad_x, grad_w, grad_b, inference)
    return {
        "ok": all(c["ok"] for c in checks),
        "cudnn_version": torch.backends.cudnn.version(),
        "training": train,
        "grad_input": grad_x,
        "grad_scale": grad_w,
        "grad_bias": grad_b,
        "inference": inference,
    }


def _sdpa_ref(torch, q, k, v):
    return torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1]), dim=-1) @ v


def run_sdpa(torch, backend: str):
    import torch.nn.functional as F
    torch.manual_seed(113)
    # Keep the memory-efficient probe deliberately small.  Broken SM/fatbin
    # dispatch can emit a very large amount of device-side diagnostic output;
    # a compact shape still exercises the backend while making failures fast
    # and classifiable instead of turning them into misleading timeouts.
    shape = (1, 1, 8, 32) if backend in ("flash", "memory-efficient") else (2, 4, 64, 64)
    q = torch.randn(*shape)
    k = torch.randn(*shape)
    v = torch.randn(*shape)
    ref = _sdpa_ref(torch, q, k, v)
    qg, kg, vg = q.half().cuda(), k.half().cuda(), v.half().cuda()
    if backend == "math":
        ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    elif backend == "flash":
        ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    elif backend == "memory-efficient":
        ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
    else:
        raise ValueError(backend)
    with ctx:
        got = F.scaled_dot_product_attention(qg, kg, vg, dropout_p=0.0, is_causal=False)
    sync(torch)
    check = tensor_metrics(torch, got, ref, 8e-2, 8e-2)
    return {"ok": check["ok"], "backend": backend, "shape": list(shape), "numerics": check}


def run_cuda_graph(torch):
    if not hasattr(torch.cuda, "CUDAGraph"):
        raise RuntimeError("torch.cuda.CUDAGraph is unavailable")
    x = torch.ones(1024, device="cuda")
    static_out = torch.empty_like(x)
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        static_out.copy_(x * 3.0 + 2.0)
    g.replay()
    sync(torch)
    ref = torch.full((1024,), 5.0)
    check = tensor_metrics(torch, static_out, ref, 0.0, 0.0)
    return {"ok": check["ok"], "numerics": check}


RUNNERS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "device_info": run_device_info,
    "driver_pci_bus_id": run_driver_pci_bus_id,
    "driver_api": run_driver_api,
    "driver_launch_ex": run_driver_launch_ex,
    "driver_pdl_semantics": run_driver_pdl_semantics,
    "driver_func_attributes": run_driver_func_attributes,
    "driver_function_metadata": run_driver_function_metadata,
    "driver_ptx_selection": run_driver_ptx_selection,
    "driver_buffer_clear": run_driver_buffer_clear,
    "nvml": run_nvml,
    "memory_copy": run_memory_copy,
    "streams_events": run_streams_events,
    "elementwise": run_elementwise,
    "reductions": run_reductions,
    "matmul": run_matmul,
    "bf16_matmul": run_bf16_matmul,
    "fft": run_fft,
    "sparse_mm": run_sparse_mm,
    "linalg_solve": run_linalg_solve,
    "linalg_cholesky": run_linalg_cholesky,
    "linalg_qr": run_linalg_qr,
    "linalg_inv": run_linalg_inv,
    "linalg_lstsq": run_linalg_lstsq,
    "linalg_svd": run_linalg_svd,
    "linalg_pinv": run_linalg_pinv,
    "linalg_eigh": run_linalg_eigh,
    "linalg_eigvalsh": run_linalg_eigvalsh,
    "softmax": run_softmax,
    "layernorm": run_layernorm,
    "gather_scatter": run_gather_scatter,
    "random": run_random,
    "autograd_backward": run_autograd_backward,
    "optimizer_sgd": run_optimizer_sgd,
    "optimizer_adam": run_optimizer_adam,
    "amp": run_amp,
    "conv2d": run_conv2d,
    "batch_norm": run_batch_norm,
    "sdpa_math": lambda torch: run_sdpa(torch, "math"),
    "sdpa_flash": lambda torch: run_sdpa(torch, "flash"),
    "sdpa_mem_efficient": lambda torch: run_sdpa(torch, "memory-efficient"),
    "cuda_graph": run_cuda_graph,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=TESTS)
    parser.add_argument("--list-tests", action="store_true")
    args = parser.parse_args()
    if args.list_tests:
        print("\n".join(TESTS))
        return 0
    if not args.test:
        parser.error("--test is required unless --list-tests is used")
    test = args.test
    try:
        import torch
    except Exception as exc:
        emit(test, "unavailable", reason="pytorch_import_failed", error=str(exc))
        return 4
    try:
        if not torch.cuda.is_available():
            emit(test, "unavailable", reason="cuda_not_available", torch_version=getattr(torch, "__version__", None))
            return 4
        common = {
            "torch_version": getattr(torch, "__version__", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device": torch.cuda.get_device_name(0),
        }
        result = RUNNERS[test](torch)
        status = "pass" if result.get("ok") else "incorrect"
        emit(test, status, **common, **result)
        return 0 if status == "pass" else 2
    except RuntimeError as exc:
        text = str(exc)
        lower = text.lower()
        unsupported_markers = (
            "not implemented",
            "not supported",
            "unsupported",
            "could not be found",
            "no available kernel",
            "no kernel image is available",
            "no viable backend",
            "not compiled with",
            "not exported",
        )
        status = "unsupported" if any(m in lower for m in unsupported_markers) else "error"
        emit(test, status, error=text, traceback=traceback.format_exc(limit=8))
        return 3 if status == "unsupported" else 2
    except Exception as exc:
        emit(test, "error", error=str(exc), traceback=traceback.format_exc(limit=8))
        return 2


if __name__ == "__main__":
    sys.exit(main())

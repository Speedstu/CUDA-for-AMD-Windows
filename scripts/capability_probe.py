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
import sys
import time
import traceback
from typing import Any, Callable

TESTS = (
    "device_info",
    "driver_api",
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
    "softmax",
    "layernorm",
    "gather_scatter",
    "random",
    "autograd_backward",
    "optimizer_sgd",
    "optimizer_adam",
    "amp",
    "conv2d",
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
    a = torch.randn(384, 512)
    b = torch.randn(512, 320)
    ref32 = a @ b
    got32 = a.cuda() @ b.cuda()
    sync(torch)
    c32 = tensor_metrics(torch, got32, ref32, 6e-3, 7e-4)
    a16, b16 = a[:256, :256].half(), b[:256, :256].half()
    ref16 = a16.float() @ b16.float()
    got16 = a16.cuda() @ b16.cuda()
    sync(torch)
    c16 = tensor_metrics(torch, got16, ref16, 9e-2, 2e-2)
    return {"ok": c32["ok"] and c16["ok"], "fp32": c32, "fp16": c16}


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
    x = torch.randn(2, 8, 32, 32)
    w = torch.randn(16, 8, 3, 3)
    b = torch.randn(16)
    ref = F.conv2d(x, w, b, padding=1)
    got = F.conv2d(x.cuda(), w.cuda(), b.cuda(), padding=1)
    sync(torch)
    check = tensor_metrics(torch, got, ref, 8e-3, 8e-4)
    return {"ok": check["ok"], "numerics": check}


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
    "driver_api": run_driver_api,
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
    "softmax": run_softmax,
    "layernorm": run_layernorm,
    "gather_scatter": run_gather_scatter,
    "random": run_random,
    "autograd_backward": run_autograd_backward,
    "optimizer_sgd": run_optimizer_sgd,
    "optimizer_adam": run_optimizer_adam,
    "amp": run_amp,
    "conv2d": run_conv2d,
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

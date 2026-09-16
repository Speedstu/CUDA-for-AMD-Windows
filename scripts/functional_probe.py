#!/usr/bin/env python3
"""Numerical correctness probes for CUDA-facing PyTorch through ZLUDA.

Each test is intentionally a separate process when called by test-functional.ps1.
That lets the PowerShell wrapper kill a single operation if a backend hangs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from typing import Any

TESTS = ("matmul", "conv2d", "sdpa_math", "sdpa_flash", "sdpa_mem_efficient")


def emit(test: str, status: str, **extra: Any) -> None:
    payload = {"schema": 1, "test": test, "status": status, **extra}
    encoded = json.dumps(payload, sort_keys=True)
    print(encoded, flush=True)
    print(f"CUDAAMD_RESULT:{encoded}", file=sys.stderr, flush=True)


def metrics(torch, got, ref, *, atol: float, rtol: float) -> dict[str, Any]:
    got = got.detach().float().cpu()
    ref = ref.detach().float().cpu()
    diff = (got - ref).abs()
    finite = bool(torch.isfinite(got).all().item())
    max_abs = float(diff.max().item())
    ref_scale = max(float(ref.abs().max().item()), 1e-7)
    max_abs_over_ref_scale = max_abs / ref_scale
    mean_abs = float(diff.mean().item())
    ok = finite and bool(torch.allclose(got, ref, atol=atol, rtol=rtol))
    return {
        "ok": ok,
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_abs_over_ref_scale": max_abs_over_ref_scale,
        "atol": atol,
        "rtol": rtol,
    }


def synchronize(torch) -> None:
    torch.cuda.synchronize()


def run_matmul(torch) -> dict[str, Any]:
    torch.manual_seed(3407)
    a = torch.randn((512, 512), dtype=torch.float32)
    b = torch.randn((512, 512), dtype=torch.float32)
    ref32 = a @ b

    start = time.perf_counter()
    got32 = a.cuda() @ b.cuda()
    synchronize(torch)
    fp32_ms = (time.perf_counter() - start) * 1000.0
    fp32 = metrics(torch, got32, ref32, atol=5e-3, rtol=5e-4)

    a16 = a[:256, :256].half()
    b16 = b[:256, :256].half()
    ref16 = a16.float() @ b16.float()
    start = time.perf_counter()
    got16 = a16.cuda() @ b16.cuda()
    synchronize(torch)
    fp16_ms = (time.perf_counter() - start) * 1000.0
    fp16 = metrics(torch, got16, ref16, atol=8e-2, rtol=2e-2)

    return {
        "ok": fp32["ok"] and fp16["ok"],
        "fp32": fp32,
        "fp16": fp16,
        "timing_ms": {"fp32": fp32_ms, "fp16": fp16_ms},
    }


def run_conv2d(torch) -> dict[str, Any]:
    import torch.nn.functional as F

    torch.manual_seed(3407)
    x = torch.randn((2, 8, 32, 32), dtype=torch.float32)
    w = torch.randn((16, 8, 3, 3), dtype=torch.float32)
    bias = torch.randn((16,), dtype=torch.float32)
    ref = F.conv2d(x, w, bias=bias, stride=1, padding=1)

    start = time.perf_counter()
    got = F.conv2d(x.cuda(), w.cuda(), bias=bias.cuda(), stride=1, padding=1)
    synchronize(torch)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    check = metrics(torch, got, ref, atol=8e-3, rtol=8e-4)
    return {"ok": check["ok"], "numerics": check, "timing_ms": elapsed_ms}


def sdpa_reference(torch, q, k, v):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def run_sdpa(torch, backend: str) -> dict[str, Any]:
    import torch.nn.functional as F
    torch.manual_seed(3407)
    shape = (1, 1, 8, 32) if backend in ("flash", "memory-efficient") else (2, 4, 64, 64)
    q = torch.randn(shape, dtype=torch.float32)
    k = torch.randn(shape, dtype=torch.float32)
    v = torch.randn(shape, dtype=torch.float32)
    ref = sdpa_reference(torch, q, k, v)

    qg, kg, vg = q.half().cuda(), k.half().cuda(), v.half().cuda()
    kwargs = dict(dropout_p=0.0, is_causal=False)

    if not hasattr(torch.backends, "cuda") or not hasattr(torch.backends.cuda, "sdp_kernel"):
        raise RuntimeError("torch.backends.cuda.sdp_kernel is unavailable in this PyTorch build")

    if backend == "math":
        context = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        )
    elif backend == "flash":
        context = torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        )
    elif backend == "memory-efficient":
        context = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=False, enable_mem_efficient=True
        )
    else:
        raise ValueError(backend)

    start = time.perf_counter()
    with context:
        got = F.scaled_dot_product_attention(qg, kg, vg, **kwargs)
    synchronize(torch)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    check = metrics(torch, got, ref, atol=8e-2, rtol=8e-2)
    return {
        "ok": check["ok"],
        "backend": backend,
        "shape": list(shape),
        "numerics": check,
        "timing_ms": elapsed_ms,
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
            emit(
                test,
                "unavailable",
                reason="cuda_not_available",
                torch_version=getattr(torch, "__version__", None),
            )
            return 4

        device_name = torch.cuda.get_device_name(0)
        common = {
            "torch_version": getattr(torch, "__version__", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device": device_name,
        }

        if test == "matmul":
            result = run_matmul(torch)
        elif test == "conv2d":
            result = run_conv2d(torch)
        elif test == "sdpa_math":
            result = run_sdpa(torch, "math")
        elif test == "sdpa_flash":
            result = run_sdpa(torch, "flash")
        elif test == "sdpa_mem_efficient":
            result = run_sdpa(torch, "memory-efficient")
        else:
            raise AssertionError(test)

        if result["ok"]:
            emit(test, "pass", **common, **result)
            return 0

        emit(test, "incorrect", **common, **result)
        return 2
    except RuntimeError as exc:
        text = str(exc)
        lower = text.lower()
        if test == "conv2d" and (
            "miopen.dll could not be found" in lower
            or "miopen could not be found" in lower
        ):
            emit(test, "unsupported", reason="miopen_unavailable", error=text)
            return 3
        if test.startswith("sdpa_") and (
            "no available kernel" in lower
            or "no kernel image is available" in lower
            or "not supported" in lower
            or "no viable backend" in lower
        ):
            emit(test, "unsupported", error=text)
            return 3
        emit(test, "error", error=text, traceback=traceback.format_exc(limit=8))
        return 2
    except Exception as exc:
        emit(test, "error", error=str(exc), traceback=traceback.format_exc(limit=8))
        return 2


if __name__ == "__main__":
    sys.exit(main())

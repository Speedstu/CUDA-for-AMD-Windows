#!/usr/bin/env python3
"""Isolated, explicit cuDNN v8 -> MIOpen tensor-transform GPU numerical test.

Only run with an intentionally staged experimental runtime containing an AMD
backend that provides miopenTransformTensor. Does not install or replace DLLs.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-dll", required=True, type=Path)
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float16"), default=["float32", "float16"])
    args = parser.parse_args()

    dll = args.bridge_dll.resolve(strict=True)
    if dll.name.lower() != "cudnn64_8.dll":
        parser.error("--bridge-dll must be the specifically staged cudnn64_8.dll")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-facing AMD GPU unavailable. Do not substitute a CPU test.")
    if not hasattr(ctypes, "WinDLL"):
        raise RuntimeError("This bridge ABI probe requires Windows.")

    cudnn = ctypes.WinDLL(str(dll))
    ptr = ctypes.c_void_p
    status = ctypes.c_int

    cudnn.cudnnCreate.argtypes = [ctypes.POINTER(ptr)]
    cudnn.cudnnCreate.restype = status
    cudnn.cudnnDestroy.argtypes = [ptr]
    cudnn.cudnnDestroy.restype = status
    cudnn.cudnnCreateTensorDescriptor.argtypes = [ctypes.POINTER(ptr)]
    cudnn.cudnnCreateTensorDescriptor.restype = status
    cudnn.cudnnDestroyTensorDescriptor.argtypes = [ptr]
    cudnn.cudnnDestroyTensorDescriptor.restype = status
    cudnn.cudnnSetTensor4dDescriptorEx.argtypes = [ptr, status] + [status] * 8
    cudnn.cudnnSetTensor4dDescriptorEx.restype = status
    cudnn.cudnnTransformTensor.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ptr]
    cudnn.cudnnTransformTensor.restype = status
    cudnn.cudnnSetTensor.argtypes = [ptr, ptr, ptr, ptr]
    cudnn.cudnnSetTensor.restype = status
    cudnn.cudnnScaleTensor.argtypes = [ptr, ptr, ptr, ptr]
    cudnn.cudnnScaleTensor.restype = status

    def ck(code: int, name: str) -> None:
        if code:
            raise RuntimeError(f"{name} returned cuDNN status={code}")

    handle = ptr()
    ck(cudnn.cudnnCreate(ctypes.byref(handle)), "cudnnCreate")
    results = []
    try:
        for name in args.dtypes:
            dtype = torch.float32 if name == "float32" else torch.float16
            x = torch.arange(1, 13, dtype=torch.float32).reshape(1, 2, 2, 3).to(device="cuda", dtype=dtype)
            # Expose a physically NHWC-strided allocation as logical NCHW.
            y = torch.full((1, 2, 3, 2), 10, dtype=dtype, device="cuda").permute(0, 3, 1, 2)
            before = y.detach().float().cpu().clone()
            expected = 2 * x.detach().float().cpu() + 0.5 * before
            alpha = ctypes.c_float(2.0)
            beta = ctypes.c_float(0.5)
            xd, yd = ptr(), ptr()
            try:
                ck(cudnn.cudnnCreateTensorDescriptor(ctypes.byref(xd)), "create x descriptor")
                ck(cudnn.cudnnCreateTensorDescriptor(ctypes.byref(yd)), "create y descriptor")
                def set_desc(desc, tensor):
                    n, c, h, w = tensor.shape
                    sn, sc, sh, sw = tensor.stride()
                    return cudnn.cudnnSetTensor4dDescriptorEx(
                        desc, 0 if name == "float32" else 2,
                        n, c, h, w, sn, sc, sh, sw,
                    )
                ck(set_desc(xd, x), "set x descriptor")
                ck(set_desc(yd, y), "set y descriptor")
                rc = cudnn.cudnnTransformTensor(
                    handle, ctypes.byref(alpha), xd, ptr(x.data_ptr()),
                    ctypes.byref(beta), yd, ptr(y.data_ptr())
                )
                ck(rc, f"cudnnTransformTensor {name}")
                torch.cuda.synchronize()
                actual = y.detach().float().cpu()
                max_abs = (actual - expected).abs().max().item()
                tolerance = 1e-5 if name == "float32" else 1e-2
                if max_abs > tolerance:
                    raise AssertionError(f"{name}: bad transformed values {max_abs} > {tolerance}")
                # Fill and scale a physically non-contiguous destination.
                fill_value = ctypes.c_float(3.0)
                scale_value = ctypes.c_float(2.0)
                ck(cudnn.cudnnSetTensor(handle, yd, ptr(y.data_ptr()),
                                        ctypes.byref(fill_value)), "cudnnSetTensor")
                torch.cuda.synchronize()
                set_error = (y.detach().float().cpu() - 3.0).abs().max().item()
                if set_error > tolerance:
                    raise AssertionError(f"{name} set tensor mismatch: {set_error}")
                ck(cudnn.cudnnScaleTensor(handle, yd, ptr(y.data_ptr()),
                                          ctypes.byref(scale_value)), "cudnnScaleTensor")
                torch.cuda.synchronize()
                scale_error = (y.detach().float().cpu() - 6.0).abs().max().item()
                if scale_error > tolerance:
                    raise AssertionError(f"{name} scale tensor mismatch: {scale_error}")
                results.append({
                    "dtype": name, "max_abs": max_abs, "tolerance": tolerance,
                    "source_strides": list(x.stride()), "dest_strides": list(y.stride()),
                    "set_max_abs": set_error, "scale_max_abs": scale_error,
                    "status": "pass",
                })
            finally:
                for desc in (yd, xd):
                    if desc.value:
                        ck(cudnn.cudnnDestroyTensorDescriptor(desc), "destroy descriptor")
    finally:
        ck(cudnn.cudnnDestroy(handle), "destroy handle")

    print(json.dumps({"schema": 1, "bridge_dll": str(dll),
                      "device": torch.cuda.get_device_name(0), "tests": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""GPU qualification: cuDNN activation subset via the experimental MIOpen bridge.

Requires a separately staged, real MIOpen/ZLUDA runtime, and runs no GPU code
unless invoked explicitly. Report real GPU results, not mock-ABI status.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bridge-dll", required=True, type=Path)
    a = p.parse_args()
    path = a.bridge_dll.resolve(strict=True)
    if path.name.casefold() != "cudnn64_8.dll":
        p.error("--bridge-dll must refer to the intentionally staged cudnn64_8.dll")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA-facing GPU unavailable; CPU fallback is not acceptable")
    if not hasattr(ctypes, "WinDLL"):
        raise RuntimeError("Requires Windows for the cuDNN ABI")
    dll = ctypes.WinDLL(str(path))
    pointer = ctypes.c_void_p
    cint = ctypes.c_int
    def bind(name, types):
        fn = getattr(dll, name)
        fn.argtypes = types
        fn.restype = cint
        return fn

    create_handle = bind("cudnnCreate", [ctypes.POINTER(pointer)])
    destroy_handle = bind("cudnnDestroy", [pointer])
    create_tensor = bind("cudnnCreateTensorDescriptor", [ctypes.POINTER(pointer)])
    destroy_tensor = bind("cudnnDestroyTensorDescriptor", [pointer])
    tensor4d = bind("cudnnSetTensor4dDescriptor", [pointer, cint, cint, cint, cint, cint, cint])
    create_activation = bind("cudnnCreateActivationDescriptor", [ctypes.POINTER(pointer)])
    destroy_activation = bind("cudnnDestroyActivationDescriptor", [pointer])
    set_activation = bind("cudnnSetActivationDescriptor", [pointer,cint,cint,ctypes.c_double])
    forward = bind("cudnnActivationForward", [pointer,pointer,pointer,pointer,pointer,pointer,pointer,pointer])
    backward = bind("cudnnActivationBackward", [pointer,pointer,pointer,pointer,pointer,pointer,pointer,pointer,pointer,pointer,pointer,pointer])

    def ck(status, label):
        if status:
            raise RuntimeError(f"{label}: cuDNN status {status}")
    h, a_desc, x_desc, y_desc = pointer(), pointer(), pointer(), pointer()
    ck(create_handle(ctypes.byref(h)), "create handle")
    rows = []
    try:
        ck(create_activation(ctypes.byref(a_desc)), "create activation")
        ck(create_tensor(ctypes.byref(x_desc)), "create x descriptor")
        ck(create_tensor(ctypes.byref(y_desc)), "create y descriptor")
        for dtype_name, dtype, abi in (("float32", torch.float32,0), ("float16", torch.float16,2)):
            x_cpu = torch.tensor([-2.0,-1.0,0.25,2.0, -3.0,1.25,4.0,-0.5]).reshape(1,2,2,2)
            grad_cpu = torch.tensor([1.0,2.0,1.5,-3.0,-0.5,2.5,1.25,0.75]).reshape(1,2,2,2)
            x = x_cpu.to(device="cuda", dtype=dtype)
            dy = grad_cpu.to(device="cuda", dtype=dtype)
            y = torch.empty_like(x)
            dx = torch.empty_like(x)
            for desc in (x_desc,y_desc):
                ck(tensor4d(desc,0,abi,1,2,2,2), "configure tensor")
            alpha,beta=ctypes.c_float(1.),ctypes.c_float(0.)
            for mode, name, function, derivative in (
                (0, "sigmoid", torch.sigmoid, lambda z: z*(1-z)),
                (1, "relu", torch.relu, lambda z: (z>0).float()),
                (2, "tanh", torch.tanh, lambda z: 1-z*z),
            ):
                ck(set_activation(a_desc,mode,0,0.), "set activation")
                ck(forward(h,a_desc,ctypes.byref(alpha),x_desc,pointer(x.data_ptr()),
                           ctypes.byref(beta),y_desc,pointer(y.data_ptr())),f"forward {name}")
                ck(backward(h,a_desc,ctypes.byref(alpha),y_desc,pointer(y.data_ptr()),
                            y_desc,pointer(dy.data_ptr()),x_desc,pointer(x.data_ptr()),
                            ctypes.byref(beta),x_desc,pointer(dx.data_ptr())),f"backward {name}")
                torch.cuda.synchronize()
                ref_y = function(x_cpu)
                ref_dx = derivative(ref_y if mode != 1 else x_cpu) * grad_cpu
                eps = 1e-5 if dtype_name == "float32" else 2e-2
                forward_err = (y.float().cpu()-ref_y).abs().max().item()
                backward_err = (dx.float().cpu()-ref_dx).abs().max().item()
                if max(forward_err, backward_err)>eps:
                    raise AssertionError(f"{name} {dtype_name}: {forward_err=} {backward_err=} > {eps}")
                rows.append({"operation":name,"dtype":dtype_name,"fwd_max_abs":forward_err,
                             "bwd_max_abs":backward_err,"tolerance":eps,"status":"pass"})
    finally:
        for desc, release in ((y_desc,destroy_tensor),(x_desc,destroy_tensor),(a_desc,destroy_activation),(h,destroy_handle)):
            if desc.value:
                ck(release(desc), "release descriptor/handle")

    print(json.dumps({"schema":1,"device":torch.cuda.get_device_name(0),
        "bridge_dll":str(path),"bridge_sha256":hash_file(path),
        "results":rows},indent=2))


if __name__ == "__main__":
    main()

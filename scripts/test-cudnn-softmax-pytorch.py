#!/usr/bin/env python3
"""GPU test for experimental cuDNN accurate/channel softmax v8 bridge.

Explicitly uses a staged cudnn64_8.dll and compares with a CPU PyTorch
reference on real CUDA-facing AMD GPU; not run by CI's CPU-only job.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bridge-dll", type=Path, required=True)
    args=parser.parse_args()
    bridge=args.bridge_dll.resolve(strict=True)
    if bridge.name.lower()!="cudnn64_8.dll":
        parser.error("Require a specifically staged cudnn64_8.dll")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-facing GPU unavailable (CPU fallback is forbidden)")
    dll=ctypes.WinDLL(str(bridge))
    ptr=ctypes.c_void_p
    c_int=ctypes.c_int

    def bind(name, argtypes):
        fn=getattr(dll,name)
        fn.argtypes=argtypes
        fn.restype=c_int
        return fn

    create=bind("cudnnCreate",[ctypes.POINTER(ptr)])
    destroy=bind("cudnnDestroy",[ptr])
    create_desc=bind("cudnnCreateTensorDescriptor",[ctypes.POINTER(ptr)])
    destroy_desc=bind("cudnnDestroyTensorDescriptor",[ptr])
    set_desc=bind("cudnnSetTensor4dDescriptor",[ptr,c_int,c_int,c_int,c_int,c_int,c_int])
    forward=bind("cudnnSoftmaxForward",[ptr,c_int,c_int,ptr,ptr,ptr,ptr,ptr,ptr])
    backward=bind("cudnnSoftmaxBackward",[ptr,c_int,c_int,ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr])

    def ck(rc,op):
        if rc:
            raise RuntimeError(f"{op} status {rc}")
    h,xd,yd=ptr(),ptr(),ptr()
    ck(create(ctypes.byref(h)),"create handle")
    results=[]
    try:
        ck(create_desc(ctypes.byref(xd)),"create tensor")
        ck(create_desc(ctypes.byref(yd)),"create tensor")
        alpha,beta=ctypes.c_float(1.),ctypes.c_float(0.)
        for dtype_name,dtype,dtype_abi in (("float32",torch.float32,0),
                                            ("float16",torch.float16,2)):
            x_cpu=torch.tensor([1.,2.,-1.,3., -2.,1.,4.,0., 2.,-3.,-0.25,1.5]).reshape(1,3,2,2)
            dy_cpu=torch.tensor([.5,-1.,1.,.75, 1.25,-2.,2.,-0.5, -.7,.25,1.5,-1.]).reshape(1,3,2,2)
            x=x_cpu.to(device="cuda",dtype=dtype)
            dy=dy_cpu.to(device="cuda",dtype=dtype)
            y=torch.empty_like(x)
            dx=torch.empty_like(x)
            for d in (xd,yd):
                ck(set_desc(d,0,dtype_abi,1,3,2,2),"configure descriptor")
            ck(forward(h,1,1,ctypes.byref(alpha),xd,ptr(x.data_ptr()),
                       ctypes.byref(beta),yd,ptr(y.data_ptr())),"softmax forward")
            ck(backward(h,1,1,ctypes.byref(alpha),yd,ptr(y.data_ptr()),
                        yd,ptr(dy.data_ptr()),ctypes.byref(beta),xd,ptr(dx.data_ptr())),"softmax backward")
            torch.cuda.synchronize()
            expected_y=torch.softmax(x_cpu,dim=1)
            expected_dx=expected_y*(dy_cpu-(dy_cpu*expected_y).sum(dim=1,keepdim=True))
            e1=(y.float().cpu()-expected_y).abs().max().item()
            e2=(dx.float().cpu()-expected_dx).abs().max().item()
            tolerance=1e-5 if dtype_name=="float32" else 2e-2
            if max(e1,e2)>tolerance:
                raise AssertionError(f"softmax {dtype_name}: fwd={e1}, bwd={e2}, tolerance={tolerance}")
            results.append(dict(dtype=dtype_name,forward_max_abs=e1,
                                backward_max_abs=e2,tolerance=tolerance,status="pass"))
    finally:
        for desc,fun in ((yd,destroy_desc),(xd,destroy_desc),(h,destroy)):
            if desc.value:
                ck(fun(desc),"release")
    print(json.dumps(dict(schema=1,bridge_dll=str(bridge),
        bridge_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),
        device=torch.cuda.get_device_name(0),results=results),indent=2))

if __name__=="__main__":
    main()

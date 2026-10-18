#!/usr/bin/env python3
"""Hardware qualification of cuDNN->MIOpen using direct HIP device allocations.

This does NOT use the CPU MIOpen stub. It runs only on demand in a dedicated
process with an externally imposed timeout, using explicitly selected DLL
paths from an isolated AMD TheRock installation. GPU kernels are tiny and
each output is independently checked against a CPU reference.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bind(dll, name, args, output=ctypes.c_int):
    proc = getattr(dll, name)
    proc.argtypes = args
    proc.restype = output
    return proc


def loaded_identity(name):
    win = ctypes.WinDLL("kernel32", use_last_error=True)
    win.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    win.GetModuleHandleW.restype = ctypes.c_void_p
    win.GetModuleFileNameW.argtypes = [ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint]
    win.GetModuleFileNameW.restype = ctypes.c_uint
    handle = win.GetModuleHandleW(name)
    if not handle:
        return {"status": "not_loaded"}
    buf = ctypes.create_unicode_buffer(32768)
    n = win.GetModuleFileNameW(handle,buf,len(buf))
    if not n or n >= len(buf)-1:
        return {"status": "path_unavailable"}
    p = Path(buf.value).resolve(strict=True)
    return {"status":"loaded","path":str(p),"sha256":digest(p)}


def main():
    parser = argparse.ArgumentParser()
    for flag in ("bridge-dll","hip-dll","miopen-dll"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--only",choices=("all","activation","softmax","pooling"),default="all")
    args=parser.parse_args()
    bridge,hip_path,miopen_path=(getattr(args,n).resolve(strict=True) for n in
                                 ("bridge_dll","hip_dll","miopen_dll"))
    if bridge.name.lower()!="cudnn64_8.dll":
        parser.error("the bridge must be named cudnn64_8.dll")
    if hip_path.name.lower()!="amdhip64_7.dll" or miopen_path.name.lower()!="miopen.dll":
        parser.error("TheRock Windows HIP 7.x and MIOpen paths required")
    if hip_path.parent != miopen_path.parent:
        parser.error("HIP and MIOpen must share the same matching TheRock bin directory")
    dll_search=os.add_dll_directory(str(hip_path.parent))
    os.environ["PATH"]=str(hip_path.parent)+os.pathsep+os.environ.get("PATH","")
    os.environ["CUDA_AMD_MIOPEN_DLL"]=str(miopen_path)
    os.environ.setdefault("MIOPEN_USER_DB_PATH",str(args.report.parent / "miopen-user-db"))
    ptr=ctypes.c_void_p
    cint=ctypes.c_int
    size_t=ctypes.c_size_t
    hip=ctypes.WinDLL(str(hip_path))
    init=bind(hip,"hipInit",[ctypes.c_uint])
    devices=bind(hip,"hipGetDeviceCount",[ctypes.POINTER(cint)])
    set_device=bind(hip,"hipSetDevice",[cint])
    alloc=bind(hip,"hipMalloc",[ctypes.POINTER(ptr),size_t])
    free=bind(hip,"hipFree",[ptr])
    memcpy=bind(hip,"hipMemcpy",[ptr,ptr,size_t,cint])
    sync=bind(hip,"hipDeviceSynchronize",[])
    def ck(rc,desc):
        if rc: raise RuntimeError(f"{desc} returned status {rc}")
    ck(init(0),"hipInit")
    count=cint(0);ck(devices(ctypes.byref(count)),"hipGetDeviceCount")
    if count.value<1: raise RuntimeError("No physical HIP devices; GPU validation must not fall back")
    ck(set_device(0),"hipSetDevice")
    # The bridge's loader resolves MIOpen by this exact explicit full path.
    cudnn=ctypes.WinDLL(str(bridge))
    create=bind(cudnn,"cudnnCreate",[ctypes.POINTER(ptr)])
    destroy=bind(cudnn,"cudnnDestroy",[ptr])
    create_t=bind(cudnn,"cudnnCreateTensorDescriptor",[ctypes.POINTER(ptr)])
    destroy_t=bind(cudnn,"cudnnDestroyTensorDescriptor",[ptr])
    set_t=bind(cudnn,"cudnnSetTensor4dDescriptor",[ptr,cint,cint,cint,cint,cint,cint])
    create_a=bind(cudnn,"cudnnCreateActivationDescriptor",[ctypes.POINTER(ptr)])
    destroy_a=bind(cudnn,"cudnnDestroyActivationDescriptor",[ptr])
    set_a=bind(cudnn,"cudnnSetActivationDescriptor",[ptr,cint,cint,ctypes.c_double])
    act_fwd=bind(cudnn,"cudnnActivationForward",[ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr])
    act_bwd=bind(cudnn,"cudnnActivationBackward",[ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr])
    sm_fwd=bind(cudnn,"cudnnSoftmaxForward",[ptr,cint,cint,ptr,ptr,ptr,ptr,ptr,ptr])
    sm_bwd=bind(cudnn,"cudnnSoftmaxBackward",[ptr,cint,cint,ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr])
    create_pool=bind(cudnn,"cudnnCreatePoolingDescriptor",[ctypes.POINTER(ptr)])
    destroy_pool=bind(cudnn,"cudnnDestroyPoolingDescriptor",[ptr])
    set_pool=bind(cudnn,"cudnnSetPooling2dDescriptor",[ptr]+[cint]*8)
    pool_dims=bind(cudnn,"cudnnGetPooling2dForwardOutputDim",[ptr,ptr]+[ctypes.POINTER(cint)]*4)
    pool_fwd=bind(cudnn,"cudnnPoolingForward",[ptr,ptr,ptr,ptr,ptr,ptr,ptr,ptr])
    handle=ptr()
    ck(create(ctypes.byref(handle)),"cudnnCreate")
    results=[]
    buffers=[]
    try:
        def device_buffer(values):
            raw=(ctypes.c_float*len(values))(*values)
            dev=ptr()
            ck(alloc(ctypes.byref(dev),ctypes.sizeof(raw)),"hipMalloc")
            buffers.append(dev)
            ck(memcpy(dev,ctypes.cast(raw,ptr),ctypes.sizeof(raw),1),"hipMemcpy H2D")
            return dev
        def host_values(dev,n):
            out=(ctypes.c_float*n)()
            ck(sync(),"hipDeviceSynchronize")
            ck(memcpy(ctypes.cast(out,ptr),dev,ctypes.sizeof(out),2),"hipMemcpy D2H")
            return list(out)
        def tensor_desc(n,c,h,w):
            d=ptr()
            ck(create_t(ctypes.byref(d)),"cudnnCreateTensorDescriptor")
            ck(set_t(d,0,0,n,c,h,w),"cudnnSetTensor4dDescriptor")
            return d
        one,zero=ctypes.c_float(1.),ctypes.c_float(0.)
        if args.only in ("all","activation"):
            v=[-2.,-1.,0.25,2.]
            grad=[1.,2.,1.5,-3.]
            inp,dy,y,dx=(device_buffer(z) for z in
                (v,grad,[0.]*4,[0.]*4))
            desc=tensor_desc(1,1,1,4)
            act=ptr()
            try:
                ck(create_a(ctypes.byref(act)),"create activation")
                for mode,name,func,d_func in (
                    (0,"sigmoid",lambda x:1/(1+math.exp(-x)),lambda x,y:y*(1-y)),
                    (1,"relu",lambda x:max(0.,x),lambda x,y:float(x>0)),
                    (2,"tanh",math.tanh,lambda x,y:1-y*y)):
                    ck(set_a(act,mode,0,0.),"set activation")
                    ck(act_fwd(handle,act,ctypes.byref(one),desc,inp,
                               ctypes.byref(zero),desc,y),f"{name} forward")
                    ck(act_bwd(handle,act,ctypes.byref(one),desc,y,desc,dy,
                               desc,inp,ctypes.byref(zero),desc,dx),f"{name} backward")
                    output=host_values(y,4); d_output=host_values(dx,4)
                    exp=[func(x) for x in v]
                    expd=[d_func(x,z)*g for x,z,g in zip(v,exp,grad)]
                    err=max(abs(a-b) for a,b in zip(output,exp))
                    derr=max(abs(a-b) for a,b in zip(d_output,expd))
                    if max(err,derr)>1.e-5:raise AssertionError(f"{name} mismatch fwd={err} backward={derr}")
                    results.append({"operator":name,"dtype":"float32","forward_max_abs":err,
                                    "backward_max_abs":derr,"status":"pass"})
            finally:
                if act.value: ck(destroy_a(act),"destroy activation")
                ck(destroy_t(desc),"destroy tensor")
        if args.only in ("all","softmax"):
            v=[0.,1.,2.,0.,-1.,3.]
            grad=[1.,-.3,2.,.4,-.7,1.]
            inp,dy,y,dx=(device_buffer(z) for z in (v,grad,[0.]*6,[0.]*6))
            desc=tensor_desc(1,3,1,2)
            try:
                ck(sm_fwd(handle,1,1,ctypes.byref(one),desc,inp,
                          ctypes.byref(zero),desc,y),"softmax forward")
                ck(sm_bwd(handle,1,1,ctypes.byref(one),desc,y,desc,dy,
                          ctypes.byref(zero),desc,dx),"softmax backward")
                output=host_values(y,6);d_output=host_values(dx,6)
                reference=[0.]*6; refdx=[0.]*6
                for w in range(2):
                    maxi=max(v[c*2+w] for c in range(3))
                    d=sum(math.exp(v[c*2+w]-maxi) for c in range(3))
                    for c in range(3):reference[c*2+w]=math.exp(v[c*2+w]-maxi)/d
                    dot=sum(reference[c*2+w]*grad[c*2+w] for c in range(3))
                    for c in range(3):
                        refdx[c*2+w]=reference[c*2+w]*(grad[c*2+w]-dot)
                err=max(abs(a-b) for a,b in zip(output,reference))
                derr=max(abs(a-b) for a,b in zip(d_output,refdx))
                if max(err,derr)>1e-5:raise AssertionError(f"softmax mismatch fwd={err}, back={derr}")
                results.append({"operator":"softmax_channel_accurate","dtype":"float32",
                                "forward_max_abs":err,"backward_max_abs":derr,"status":"pass"})
            finally:
                ck(destroy_t(desc),"destroy tensor")
        if args.only in ("all","pooling"):
            v=[float(i) for i in range(1,17)]
            inp,y=(device_buffer(a) for a in (v,[0.]*4))
            xdesc=tensor_desc(1,1,4,4)
            ydesc=tensor_desc(1,1,2,2)
            pd=ptr()
            try:
                ck(create_pool(ctypes.byref(pd)),"create pooling descriptor")
                rc=set_pool(pd,1,0,2,2,0,0,2,2)
                if rc!=9:
                    raise AssertionError(f"unsupported average pooling must refuse with 9, got {rc}")
                ck(set_pool(pd,0,0,2,2,0,0,2,2),"configure max pooling")
                n,c,h,w=cint(),cint(),cint(),cint()
                ck(pool_dims(pd,xdesc,ctypes.byref(n),ctypes.byref(c),
                             ctypes.byref(h),ctypes.byref(w)),"get pooling dimensions")
                if (n.value,c.value,h.value,w.value)!=(1,1,2,2):
                    raise AssertionError(f"bad pooling dimensions: {(n.value,c.value,h.value,w.value)}")
                ck(pool_fwd(handle,pd,ctypes.byref(one),xdesc,inp,
                            ctypes.byref(zero),ydesc,y),"pooling forward")
                result=host_values(y,4)
                expected=[6.,8.,14.,16.]
                err=max(abs(a-b) for a,b in zip(result,expected))
                if err>1e-5:raise AssertionError(f"pooling numerical mismatch {result}")
                results.append({"operator":"max_pool2d_nonoverlapping","dtype":"float32",
                                "forward_max_abs":err,"status":"pass"})
            finally:
                if pd.value:ck(destroy_pool(pd),"destroy pool")
                ck(destroy_t(ydesc),"destroy y tensor")
                ck(destroy_t(xdesc),"destroy x tensor")
    finally:
        for dev in reversed(buffers):
            if dev.value: ck(free(dev),"hipFree")
        ck(destroy(handle),"destroy handle")
        dll_search.close()
    report={"schema":1,"scope":"direct_HIP_device_allocations_real_MIOpen_no_PyTorch",
       "hip_device_count":count.value,"results":results,
       "loaded_cudnn_bridge":loaded_identity("cudnn64_8.dll"),
       "loaded_hip":loaded_identity("amdhip64_7.dll"),
       "loaded_miopen":loaded_identity("MIOpen.dll"),
       "artifact_hip_sha256":digest(hip_path),
       "artifact_miopen_sha256":digest(miopen_path)}
    if any(v.get("status")!="loaded" for v in
           (report["loaded_cudnn_bridge"],report["loaded_hip"],report["loaded_miopen"])):
        raise RuntimeError("Loaded DLL hash proof missing, refusing real-GPU success report")
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GPU_TEST_ERROR: {exc!r}",file=sys.stderr)
        raise

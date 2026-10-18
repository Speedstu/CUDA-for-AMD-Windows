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
    parser.add_argument("--only",choices=("all","activation","softmax","pooling","tensor","half"),default="all")
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
    tensor4d_ex=bind(cudnn,"cudnnSetTensor4dDescriptorEx",[ptr]+[cint]*9)
    tensor_transform=bind(cudnn,"cudnnTransformTensor",[ptr,ptr,ptr,ptr,ptr,ptr,ptr])
    tensor_set=bind(cudnn,"cudnnSetTensor",[ptr,ptr,ptr,ptr])
    tensor_scale=bind(cudnn,"cudnnScaleTensor",[ptr,ptr,ptr,ptr])
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
        if args.only in ("all","tensor"):
            x,y=(device_buffer(v) for v in ([1.,2.,3.,4.],[10.]*8))
            xd,yd=ptr(),ptr()
            try:
                ck(create_t(ctypes.byref(xd)),"create transform source")
                ck(create_t(ctypes.byref(yd)),"create transform destination")
                ck(tensor4d_ex(xd,0,1,2,1,2,4,2,2,1),"configure source tensor")
                ck(tensor4d_ex(yd,0,1,2,1,2,8,1,4,2),"configure strided output tensor")
                alpha2,beta_half=ctypes.c_float(2.),ctypes.c_float(.5)
                ck(tensor_transform(handle,ctypes.byref(alpha2),xd,x,
                    ctypes.byref(beta_half),yd,y),"transform tensor")
                actual=host_values(y,8)
                expected=[7.,11.,9.,13.,10.,10.,10.,10.]
                err=max(abs(a-b) for a,b in zip(actual,expected))
                if err>1e-5:
                    raise AssertionError(f"real GPU strided transform mismatch: {actual!r}")
                results.append({"operator":"transform_tensor_strided","dtype":"float32",
                                "forward_max_abs":err,"status":"pass"})
                fill_value=ctypes.c_float(3.)
                ck(tensor_set(handle,yd,y,ctypes.byref(fill_value)),"set tensor")
                actual=host_values(y,8)
                expected=[3.,3.,3.,3.,10.,10.,10.,10.]
                err=max(abs(a-b) for a,b in zip(actual,expected))
                if err>1e-5:raise AssertionError(f"real GPU set tensor mismatch: {actual!r}")
                results.append({"operator":"set_tensor_strided","dtype":"float32",
                                "forward_max_abs":err,"status":"pass"})
                scale=ctypes.c_float(2.)
                ck(tensor_scale(handle,yd,y,ctypes.byref(scale)),"scale tensor")
                actual=host_values(y,8)
                expected=[6.,6.,6.,6.,10.,10.,10.,10.]
                err=max(abs(a-b) for a,b in zip(actual,expected))
                if err>1e-5:raise AssertionError(f"real GPU scale tensor mismatch: {actual!r}")
                results.append({"operator":"scale_tensor_strided","dtype":"float32",
                                "forward_max_abs":err,"status":"pass"})
            finally:
                for d in (yd,xd):
                    if d.value:ck(destroy_t(d),"destroy transform descriptor")
        # Additional FP16 hardware qualification. This is deliberately a
        # separate set of allocations and descriptors, not a float32 result
        # relabelled as half precision.
        if args.only in ("all","half"):
            def device_half(values):
                raw=struct.pack("<"+str(len(values))+"e",*(float(v) for v in values))
                host=ctypes.create_string_buffer(raw)
                dev=ptr()
                ck(alloc(ctypes.byref(dev),len(raw)),"hipMalloc fp16")
                buffers.append(dev)
                ck(memcpy(dev,ctypes.cast(host,ptr),len(raw),1),"hipMemcpy fp16 H2D")
                return dev
            def host_half(dev,count):
                ck(sync(),"hipDeviceSynchronize fp16")
                host=ctypes.create_string_buffer(count*2)
                ck(memcpy(ctypes.cast(host,ptr),dev,count*2,2),"hipMemcpy fp16 D2H")
                return list(struct.unpack("<"+str(count)+"e",host.raw))
            def tensor_half(n,c,h,w):
                d=ptr()
                ck(create_t(ctypes.byref(d)),"create fp16 tensor descriptor")
                ck(set_t(d,0,2,n,c,h,w),"set fp16 tensor descriptor")
                return d
            v=[-2.,-1.,.25,2.]
            grad=[1.,2.,1.5,-3.]
            inp,dy,y,dx=(device_half(z) for z in (v,grad,[0.]*4,[0.]*4))
            desc=tensor_half(1,1,1,4)
            act=ptr()
            try:
                ck(create_a(ctypes.byref(act)),"create fp16 activation")
                for mode,name,func,d_func in (
                    (0,"sigmoid",lambda x:1/(1+math.exp(-x)),lambda x,y:y*(1-y)),
                    (1,"relu",lambda x:max(0.,x),lambda x,y:float(x>0)),
                    (2,"tanh",math.tanh,lambda x,y:1-y*y)):
                    ck(set_a(act,mode,0,0.),"set fp16 activation")
                    ck(act_fwd(handle,act,ctypes.byref(one),desc,inp,
                               ctypes.byref(zero),desc,y),"fp16 activation forward")
                    ck(act_bwd(handle,act,ctypes.byref(one),desc,y,desc,dy,
                               desc,inp,ctypes.byref(zero),desc,dx),"fp16 activation backward")
                    actual=host_half(y,4);actual_dx=host_half(dx,4)
                    expected=[func(x) for x in v]
                    exp_dx=[d_func(x,z)*g for x,z,g in zip(v,expected,grad)]
                    err=max(abs(a-b) for a,b in zip(actual,expected))
                    derr=max(abs(a-b) for a,b in zip(actual_dx,exp_dx))
                    if max(err,derr)>2e-2:
                        raise AssertionError(f"half {name} mismatch {err=} {derr=}")
                    results.append({"operator":name,"dtype":"float16","forward_max_abs":err,
                                    "backward_max_abs":derr,"status":"pass"})
            finally:
                if act.value:ck(destroy_a(act),"destroy half activation")
                ck(destroy_t(desc),"destroy half descriptor")

            v=[0.,1.,2.,0.,-1.,3.]
            grad=[1.,-.3,2.,.4,-.7,1.]
            inp,dy,y,dx=(device_half(z) for z in (v,grad,[0.]*6,[0.]*6))
            desc=tensor_half(1,3,1,2)
            try:
                ck(sm_fwd(handle,1,1,ctypes.byref(one),desc,inp,
                          ctypes.byref(zero),desc,y),"fp16 softmax forward")
                ck(sm_bwd(handle,1,1,ctypes.byref(one),desc,y,desc,dy,
                          ctypes.byref(zero),desc,dx),"fp16 softmax backward")
                actual=host_half(y,6);actual_dx=host_half(dx,6)
                expected=[0.]*6;exp_dx=[0.]*6
                for w in range(2):
                    mx=max(v[c*2+w] for c in range(3))
                    total=sum(math.exp(v[c*2+w]-mx) for c in range(3))
                    for c in range(3):
                        expected[c*2+w]=math.exp(v[c*2+w]-mx)/total
                    dot=sum(expected[c*2+w]*grad[c*2+w] for c in range(3))
                    for c in range(3):
                        exp_dx[c*2+w]=expected[c*2+w]*(grad[c*2+w]-dot)
                err=max(abs(a-b) for a,b in zip(actual,expected))
                derr=max(abs(a-b) for a,b in zip(actual_dx,exp_dx))
                if max(err,derr)>2e-2:
                    raise AssertionError(f"half softmax mismatch {err=} {derr=}")
                results.append({"operator":"softmax_channel_accurate","dtype":"float16",
                                "forward_max_abs":err,"backward_max_abs":derr,"status":"pass"})
            finally:
                ck(destroy_t(desc),"destroy half softmax descriptor")

            inp,y=(device_half(z) for z in ([float(i) for i in range(1,17)],[0.]*4))
            xd,yd=tensor_half(1,1,4,4),tensor_half(1,1,2,2)
            pd=ptr()
            try:
                ck(create_pool(ctypes.byref(pd)),"create half pool")
                ck(set_pool(pd,0,0,2,2,0,0,2,2),"set half pool")
                ck(pool_fwd(handle,pd,ctypes.byref(one),xd,inp,
                            ctypes.byref(zero),yd,y),"fp16 pooling forward")
                actual=host_half(y,4)
                err=max(abs(a-b) for a,b in zip(actual,[6.,8.,14.,16.]))
                if err>2e-2:raise AssertionError(f"half max pool mismatch {actual}")
                results.append({"operator":"max_pool2d_nonoverlapping",
                                "dtype":"float16","forward_max_abs":err,"status":"pass"})
            finally:
                if pd.value:ck(destroy_pool(pd),"destroy half pool")
                ck(destroy_t(yd),"destroy half y tensor")
                ck(destroy_t(xd),"destroy half x tensor")

            inp,y=(device_half(z) for z in ([1.,2.,3.,4.],[10.]*8))
            xd,yd=ptr(),ptr()
            try:
                ck(create_t(ctypes.byref(xd)),"create half source descriptor")
                ck(create_t(ctypes.byref(yd)),"create half dest descriptor")
                ck(tensor4d_ex(xd,2,1,2,1,2,4,2,2,1),"configure half source")
                ck(tensor4d_ex(yd,2,1,2,1,2,8,1,4,2),"configure half dest")
                alpha2,beta_half=ctypes.c_float(2.),ctypes.c_float(.5)
                ck(tensor_transform(handle,ctypes.byref(alpha2),xd,inp,
                                    ctypes.byref(beta_half),yd,y),"half tensor transform")
                actual=host_half(y,8)
                err=max(abs(a-b) for a,b in zip(actual,[7.,11.,9.,13.,10.,10.,10.,10.]))
                if err>2e-2:raise AssertionError(f"half tensor transform mismatch {actual}")
                results.append({"operator":"transform_tensor_strided","dtype":"float16",
                                "forward_max_abs":err,"status":"pass"})
                val=ctypes.c_float(3.)
                ck(tensor_set(handle,yd,y,ctypes.byref(val)),"half set tensor")
                actual=host_half(y,8)
                err=max(abs(a-b) for a,b in zip(actual,[3.,3.,3.,3.,10.,10.,10.,10.]))
                if err>2e-2:raise AssertionError(f"half set tensor mismatch {actual}")
                results.append({"operator":"set_tensor_strided","dtype":"float16",
                                "forward_max_abs":err,"status":"pass"})
                val=ctypes.c_float(2.)
                ck(tensor_scale(handle,yd,y,ctypes.byref(val)),"half scale tensor")
                actual=host_half(y,8)
                err=max(abs(a-b) for a,b in zip(actual,[6.,6.,6.,6.,10.,10.,10.,10.]))
                if err>2e-2:raise AssertionError(f"half scale tensor mismatch {actual}")
                results.append({"operator":"scale_tensor_strided","dtype":"float16",
                                "forward_max_abs":err,"status":"pass"})
            finally:
                for d in (yd,xd):
                    if d.value:ck(destroy_t(d),"destroy half tensor descriptor")
    finally:
        for dev in reversed(buffers):
            if dev.value: ck(free(dev),"hipFree")
        ck(destroy(handle),"destroy handle")
        dll_search.close()
    device_name=ctypes.create_string_buffer(256)
    try:
        name_func=bind(hip,"hipDeviceGetName",[ptr,cint,cint])
        ck(name_func(ctypes.cast(device_name,ptr),len(device_name),0),"hipDeviceGetName")
        gpu_name=device_name.value.decode("utf-8",errors="replace")
    except AttributeError:
        gpu_name="not_available_from_HIP"
    report={"schema":1,"scope":"direct_HIP_device_allocations_real_MIOpen_no_PyTorch",
       "hip_device_name":gpu_name,
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

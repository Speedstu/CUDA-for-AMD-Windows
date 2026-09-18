#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
import torch

p=argparse.ArgumentParser(); p.add_argument('--n',type=int,required=True); p.add_argument('--warmup',type=int,default=10); p.add_argument('--iterations',type=int,default=30); a=p.parse_args()
n=a.n
x=torch.empty((n,n),device='cuda',dtype=torch.float32); y=torch.empty_like(x); out=torch.empty_like(x)
x.fill_(0.5); y.fill_(0.25)
for _ in range(a.warmup): torch.mm(x,y,out=out)
torch.cuda.synchronize()
start=torch.cuda.Event(enable_timing=True); stop=torch.cuda.Event(enable_timing=True)
start.record(); wall=time.perf_counter()
for _ in range(a.iterations): torch.mm(x,y,out=out)
stop.record(); torch.cuda.synchronize(); wall_ms=(time.perf_counter()-wall)*1000.0
event_total_ms=float(start.elapsed_time(stop)); wall_per_ms=wall_ms/a.iterations; event_per_ms=event_total_ms/a.iterations; wall_tflops=(2.0*n*n*n)/(wall_per_ms*1.0e9)
print(json.dumps({'backend':'pytorch-cuda-zluda','device':torch.cuda.get_device_name(0),'n':n,'iterations':a.iterations,'wall_ms_per_gemm':wall_per_ms,'event_ms_per_gemm':event_per_ms,'wall_tflops':wall_tflops,'event_total_ms':event_total_ms,'wall_total_ms':wall_ms},sort_keys=True))

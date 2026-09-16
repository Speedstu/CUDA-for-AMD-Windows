# Troubleshooting

## `nvcuda.dll` not found

Stage the runtime beside the application and put ZLUDA at the front of `PATH`, or use `scripts/run-zluda.ps1`.

## HIP runtime mismatch

Do not mix arbitrary ROCm/HIP versions. The historical profile used a specific HIP 7.13 overlay with a ROCm 6.4 installation. For new machines, prefer a coherent current Windows HIP SDK; use the historical mix only to reproduce it.

## rocBLAS / hipBLASLt cannot find kernels

Set:

```text
ROCBLAS_TENSILE_LIBPATH=<HIP_ROOT>\bin\rocblas\library
HIPBLASLT_TENSILE_LIBPATH=<HIP_ROOT>\bin\hipblaslt\library
```

## CMake insists on CUDA toolkit discovery

Use the manual LibTorch import-library approach in `examples/manual-libtorch-cuda.cmake` and supply CUDA 11.8 headers explicitly.

## Wrong CUDA compute capability

`ZLUDA_CC=8.6` was intentionally exposed to CUDA-facing software. Do not replace it with `gfx1200`; those are different architecture namespaces.

## cuFFT errors

Some Windows ZLUDA/PyTorch FFT paths have historically returned unsupported errors. Treat FFT-heavy applications separately from GEMM-heavy ML/RL workloads.

## First iteration is very slow

JIT/kernel compilation and caches can dominate the first iteration. Benchmark warmed iterations under the same runtime/cache configuration.

## PyTorch fused SDPA returns `NO_BINARY_FOR_GPU`

On the experimental ZLUDA v7/TheRock path with PyTorch 2.0.1+cu118, Flash and memory-efficient SDPA depend on fused compute that is present only in NVIDIA cubins; their low-architecture PTX fallbacks are not complete implementations. The project patch fails closed rather than allowing a silent bad tensor.

For applications that can use PyTorch's math SDPA backend, launch Python with:

```powershell
.\scripts\run-zluda.ps1 -RuntimeRoot .\.runtime-v2 -Program C:\path\to\python.exe -ProgramArgs @('app.py') -PyTorchSafeSDPA
```

This is an opt-in PyTorch process setting. It does not change `ZLUDA_CC` or claim that the fused backends are implemented.
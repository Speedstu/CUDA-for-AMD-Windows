# Troubleshooting

## `nvcuda.dll` not found

Stage the runtime beside the application and put ZLUDA at the front of `PATH`, or use `scripts/run-zluda.ps1`.

## HIP runtime mismatch

Do not mix arbitrary ROCm/HIP versions. The current target is a coherent TheRock HIP SDK nightly 7.14.0a20260612 installation; the historical HIP 7.13 overlay with a ROCm 6.4 installation is only for reproducing old experiments.

## rocBLAS / hipBLASLt cannot find kernels

Set:

```text
ROCBLAS_TENSILE_LIBPATH=<HIP_ROOT>\bin\rocblas\library
HIPBLASLT_TENSILE_LIBPATH=<HIP_ROOT>\bin\hipblaslt\library
```

## CMake insists on CUDA toolkit discovery

Use the manual LibTorch import-library approach in `examples/manual-libtorch-cuda.cmake` and supply CUDA 11.8 headers explicitly.

## Wrong CUDA compute capability

`ZLUDA_CC=8.6` was intentionally exposed to CUDA-facing software. Do not replace it with `gfx1201`; those are different architecture namespaces.

## cuFFT errors

Some Windows ZLUDA/PyTorch FFT paths have historically returned unsupported errors. Treat FFT-heavy applications separately from GEMM-heavy ML/RL workloads.

## First iteration is very slow

JIT/kernel compilation and caches can dominate the first iteration. Benchmark warmed iterations under the same runtime/cache configuration.

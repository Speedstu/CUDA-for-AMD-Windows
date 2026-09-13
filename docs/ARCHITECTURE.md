# Architecture

## Exact recovered runtime composition

The retained staging script used two sources.

### ZLUDA core — v6-preview.69

Copied from the core tree: `nvcuda.dll`, `nvml.dll`, `zluda_redirect.dll`, `zluda_precompile.exe`, `nvcudart_hybrid64.dll`, cuFFT and cuSPARSE compatibility DLLs.

### Custom overlay

Copied from the recovered `amd-zluda-custom` snapshot: `amdhip64.dll`, `amdhip64_7.dll`, `rocm_kpack.dll`, `cublas64_11/12/13.dll`, and `cublasLt64_11/12/13.dll`.

For CUDA 11 the directory also retained the original ZLUDA BLAS DLLs under fallback names including `cublas64_11_zluda.dll` and `cublasLt64_11_zluda.dll`. The custom cuBLASLt shim forwards to the real ZLUDA implementation.

## Build-time stack

The target C++ application was compiled with AMD clang from ROCm 6.4 while manually linking CUDA-enabled LibTorch 2.3.0+cu118 import libraries. CUDA 11.8.89 headers were supplied locally.

- build-time API surface: CUDA / LibTorch CUDA
- runtime translation: ZLUDA
- runtime compute backend: HIP / rocBLAS / hipBLASLt

## Recovered runtime environment

```text
ZLUDA_CC=8.6
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
ROCBLAS_TENSILE_LIBPATH=<HIP_ROOT>\bin\rocblas\library
HIPBLASLT_TENSILE_LIBPATH=<HIP_ROOT>\bin\hipblaslt\library
```

PATH priority placed ZLUDA and HIP before unrelated CUDA installations.

## Why the setup is unusual

The recovered stack combines a pinned ZLUDA core, newer HIP runtime overlay, custom cuBLAS routing, cuBLASLt forwarding/heuristics, manual LibTorch CUDA linkage, local CUDA headers and workload-specific tuning. It is best treated as a compatibility stack rather than a single DLL swap.

# Architecture

The public runtime is intentionally built from upstream components.

```text
CUDA-targeted application
        |
      ZLUDA
        |
+-------+--------------------+
|       |        |           |
cuBLAS  cuBLASLt cuSPARSE    cuFFT
|       |        |           |
rocBLAS hipBLASLt rocSPARSE  HIP/ROCm backend
        |
      AMD GPU
```

## Public runtime

`scripts/install.ps1` creates the runtime from:

1. the installed AMD GPU driver and Windows HIP SDK;
2. the pinned official ZLUDA `v6-preview.69` Windows release;
3. optional pinned LibTorch `2.3.0+cu118` for CUDA-facing LibTorch applications.

The normal install path does **not** copy files from `local-artifacts/` and does not require any recovered custom DLL.

`scripts/stage-runtime.ps1` places the ZLUDA compatibility libraries beside the target executable. `scripts/run-zluda.ps1` then launches that executable through `zluda.exe` while putting the selected HIP SDK libraries on `PATH`.

## Runtime discovery

`gpu-scan.ps1` prefers AMD's `hipInfo.exe` because it exposes the native architecture (`gcnArchName`, for example `gfx1201`) directly. Windows GPU information is used as a fallback when possible.

`doctor.ps1` checks for the driver HIP runtime plus the HIP SDK libraries needed by the validated path:

- rocBLAS
- hipBLASLt
- rocSPARSE
- hipInfo

`test-runtime.ps1` executes ZLUDA's own `cuda_check.exe` and records a machine-readable report.

## LibTorch

LibTorch remains CUDA-facing. Applications link the normal CUDA-enabled LibTorch build; ZLUDA translates the CUDA-facing runtime calls to the AMD HIP/ROCm stack at execution time.

For projects where LibTorch's CMake package insists on discovering a full NVIDIA CUDA Toolkit, `examples/manual-libtorch-cuda.cmake` shows the manual import pattern used by the validated training project.

## cuDNN caveat

The validated stable Windows HIP SDK path provides the math libraries required by the tested dense/PPO workload, but it does not provide the complete Linux ROCm AI-library stack. In particular, ZLUDA's cuDNN checks fail without a MIOpen-compatible backend.

This is why compatibility is reported per workload rather than as a blanket CUDA-support claim.

## Historical custom overlay

The original development tree also contained an experimental/custom cuBLAS/cuBLASLt and HIP runtime overlay. Those files helped during earlier compatibility/performance work, but they are **not required** for the public HIP SDK 7.2 target path.

The recovered binaries remain locally fingerprinted for research and provenance work. They are not part of the public installation dependency chain.

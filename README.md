# CUDA for AMD on Windows

Run CUDA-targeted Windows applications on AMD GPUs using ZLUDA and ROCm/HIP.

[![Windows](https://img.shields.io/badge/platform-Windows%20x64-555555)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![AMD](https://img.shields.io/badge/GPU-AMD%20Radeon-555555)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![verify](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml/badge.svg)](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml)

This project packages a working Windows compatibility stack for software built against CUDA but running on an AMD GPU.

It combines a pinned ZLUDA runtime with HIP/ROCm and a custom BLAS overlay. The setup was tested on an RX 9060 XT (`gfx1200`) with real CUDA-enabled LibTorch workloads, including long-running training runs.

> This is a compatibility layer, not native CUDA. Applications still need to stay within the CUDA surface implemented by ZLUDA and the available HIP/ROCm backend libraries.

## Stack

```text
CUDA application
      |
    ZLUDA
      |
cuBLAS compatibility layer
      |
HIP / rocBLAS / hipBLASLt
      |
  AMD Radeon
```

Reference configuration:

| Component | Version |
| --- | --- |
| ZLUDA | `v6-preview.69` |
| ROCm toolchain | `6.4` |
| HIP runtime overlay | `7.13` |
| LibTorch | `2.3.0 + cu118` |
| CUDA headers/runtime metadata | `11.8.89` |
| AMD target | `gfx1200` |
| CUDA capability exposed through ZLUDA | `8.6` |

The exact upstream archives are pinned by filename and SHA-256 in [`manifests/upstream-assets.sha256`](manifests/upstream-assets.sha256).

## Quick start

Requirements:

- Windows x64
- AMD Radeon GPU with a working Windows HIP/ROCm installation
- PowerShell

Prepare ZLUDA and LibTorch:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 `
  -DownloadZluda `
  -DownloadLibTorch
```

Stage the compatibility DLLs beside an application:

```powershell
.\scripts\stage-runtime.ps1 -TargetDir C:\path\to\your-app
```

Launch it through ZLUDA:

```powershell
.\scripts\run-zluda.ps1 -Program C:\path\to\your-app\app.exe
```

Check the local setup and pinned hashes:

```powershell
.\scripts\verify.ps1
```

## What works

The recovered setup has successfully run CUDA-enabled LibTorch code on AMD under Windows, including:

- CUDA device-backed tensor workloads
- neural-network inference
- PPO / reinforcement-learning training
- GEMM-heavy workloads
- cuBLAS-compatible calls routed to AMD libraries
- long-running training with checkpoint save/resume

For the original workload, the ZLUDA + LibTorch path sustained roughly **70k-109k overall steps/s**. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the retained measurements and test context.

## Compatibility

This is not a universal replacement for an NVIDIA CUDA installation.

Likely candidates are applications dominated by common CUDA driver/runtime calls and BLAS operations. Applications may need extra work when they depend heavily on unsupported CUDA features, unusual PTX behavior, TensorRT, NCCL, custom CUDA extensions, or incomplete library paths such as some FFT/cuDNN workloads.

The current reconstructed `cuda_check.exe` probe reaches AMD hipBLASLt successfully, then stalls later in the probe. That result is tracked honestly as a partial smoke in [`docs/SMOKE_TEST.md`](docs/SMOKE_TEST.md).

## Custom BLAS layer

The original setup used a custom cuBLAS proxy and cuBLASLt forwarding shim on top of ZLUDA.

Recovered runtime switches include:

```text
ZLUDA_CUBLAS_USE_HIPBLASLT
ZLUDA_CUBLAS_AUTOTUNE
ZLUDA_CUBLAS_WORKSPACE_MB
ZLUDA_CUBLAS_SOLUTION_MAP
HUMAN_CUBLASLT_SHIM_STATS
HUMAN_CUBLASLT_FORCE_WORKSPACE_MB
```

The exact recovered wrapper binaries are fingerprinted in [`manifests/recovered-artifacts.sha256`](manifests/recovered-artifacts.sha256), but are not committed to this repository because their original source/provenance has not yet been recovered.

## Project layout

```text
scripts/      setup, staging, launch and verification
docs/         architecture, benchmarks and troubleshooting
examples/     CMake and launcher examples
manifests/    pinned versions and SHA-256 fingerprints
```

The generated runtime and recovered local binaries are ignored by Git.

## Notes for C++ / LibTorch

The working C++ setup manually linked CUDA-enabled LibTorch import libraries instead of depending on a normal NVIDIA runtime installation at execution time.

A minimal pattern is included in [`examples/manual-libtorch-cuda.cmake`](examples/manual-libtorch-cuda.cmake).

## Status

The repository currently preserves the reproducible runtime layout, dependency versions, setup scripts, hashes and benchmark evidence from the working Windows AMD setup.

The original source for the custom cuBLAS proxy and `cublasLtShim.c` has not been recovered yet. See [`docs/RECOVERY_NOTES.md`](docs/RECOVERY_NOTES.md) for the full recovery history.

## Credits

Built around [ZLUDA](https://github.com/vosen/ZLUDA), AMD ROCm/HIP and PyTorch/LibTorch.

Third-party components remain under their upstream licenses. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

# CUDA for AMD on Windows

**WORKING REPRODUCIBLE STACK IS NOW UPLOADED.**

Run CUDA-targeted Windows applications on AMD GPUs through ZLUDA + ROCm/HIP.

[![Windows](https://img.shields.io/badge/platform-Windows%20x64-555555)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![AMD](https://img.shields.io/badge/GPU-AMD%20Radeon-555555)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![verify](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml/badge.svg)](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml)

A reproducible Windows CUDA compatibility setup built around **ZLUDA + AMD HIP/ROCm**. It is intended for CUDA-facing compute applications, including workloads that use CUDA-enabled LibTorch.

> [!IMPORTANT]
> **Validated hardware is currently AMD Radeon RX 9060 XT (`gfx1200`) only.** Other AMD GPUs are candidates, not guaranteed working devices. If you test another card, please open a [GPU compatibility report](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/new?template=gpu-compatibility.yml), whether it works or fails.

## Verified today

The public, upstream-only path has been tested without any private/recovered DLLs:

- ZLUDA `v6-preview.69` from the official ZLUDA release
- AMD HIP SDK `6.4`
- LibTorch `2.3.0 + cu118`
- RX 9060 XT / `gfx1200`
- `nvcuda`, cuBLAS, cuBLASLt, cuSPARSE and cuFFT all pass `cuda_check`
- a real **2,216,347-parameter PPO network completed forward/inference, PPO learning and optimizer work on the CUDA-facing device**
- one clean validation iteration completed **65,536 timesteps** using the runtime produced by this repository

That integration test used the same CUDA-facing LibTorch training workload that originally motivated this project. See [`docs/VALIDATION.md`](docs/VALIDATION.md).

This does **not** mean every CUDA program or AI model works. CUDA API/library coverage is workload-dependent.

## How it works

```text
CUDA-targeted Windows application
              |
            ZLUDA
              |
 cuBLAS / cuSPARSE / cuFFT compatibility
              |
 rocBLAS / hipBLASLt / rocSPARSE / HIP
              |
           AMD GPU
```

## Install

### 1. Install the AMD prerequisites

Install a current AMD GPU driver and the **AMD HIP SDK for Windows including HIP Libraries**.

The validated reference uses HIP SDK 6.4. Newer versions may work but should be treated as unverified until reported.

AMD Windows HIP SDK guide:
https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.4.2/index.html

### 2. Clone and run the installer

```powershell
git clone https://github.com/Speedstu/CUDA-for-AMD-Windows.git
cd CUDA-for-AMD-Windows
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

`install.ps1` will:

1. detect the AMD GPU and native `gfxXXXX` target;
2. verify the AMD driver/HIP SDK and required math libraries;
3. download the pinned official ZLUDA Windows build;
4. download LibTorch `2.3.0+cu118` (about 2.66 GB);
5. verify the downloaded SHA-256 hashes;
6. generate `.runtime\runtime-config.json` and `.runtime\gpu-report.json`;
7. run ZLUDA's `cuda_check.exe` against the installed AMD stack.

If you do not need LibTorch:

```powershell
.\scripts\install.ps1 -SkipLibTorch
```

## Run a CUDA-targeted application

```powershell
.\scripts\run-zluda.ps1 -Program C:\path\to\app.exe
```

The launcher stages the required ZLUDA compatibility DLLs beside the target application and sets the HIP/ROCm runtime paths for that run.

You can also stage without launching:

```powershell
.\scripts\stage-runtime.ps1 -TargetDir C:\path\to\your-app
```

## Diagnose a machine

```powershell
.\scripts\doctor.ps1
.\scripts\gpu-scan.ps1
.\scripts\test-runtime.ps1
```

The GPU scanner records the model, `gfx` architecture, driver and HIP information. It does not intentionally collect usernames, tokens or user files.

Example on the validated machine:

```text
AMD Radeon RX 9060 XT -> gfx1200 -> RDNA4 -> validated-reference
```

## Current GPU status

| GPU | Target | Project status |
| --- | --- | --- |
| Radeon RX 9060 XT | `gfx1200` | ✅ validated reference |

The scanner recognizes other Windows HIP architecture families and marks them as **unverified candidates** rather than claiming support. Detection is not proof that a workload runs.

AMD's current Windows hardware table:
https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html

## Runtime coverage on the validated setup

Current upstream runtime check:

| CUDA-facing component | Result |
| --- | --- |
| CUDA driver / `nvcuda` | ✅ |
| cuBLAS | ✅ via rocBLAS |
| cuBLASLt | ✅ via hipBLASLt |
| cuSPARSE | ✅ via rocSPARSE |
| cuFFT | ✅ |
| cuDNN | ⚠️ unavailable with the validated stable Windows HIP SDK |

The stable Windows HIP SDK does not ship the full ROCm AI-library stack such as MIOpen, so convolution-heavy software that requires cuDNN can need a newer/nightly HIP stack or additional work. Dense/GEMM-heavy LibTorch training does not necessarily require cuDNN; the validated PPO workload completed without it.

## Performance

A controlled 2026-09-13 A/B ran **10 iterations per runtime** on the same RX 9060 XT PPO workload. After discarding the first iteration of each trial as warmup, the public upstream path reached **13,278 median overall SPS** versus **12,876** for the recovered custom overlay. In this workload the custom overlay was about **3.03% slower**, so upstream remains the default.

Historical tuned runs used a different training configuration and reached roughly **70k–109k overall steps/s**. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for methodology and raw data.

## Optional historical custom overlay

The original development environment also experimented with a custom cuBLAS/cuBLASLt/HIP overlay. It is **not required** for the validated public path and, based on the controlled A/B above, is not currently a performance win for the reference PPO workload.

The recovered DLLs remain fingerprinted in `manifests/recovered-artifacts.sha256`. They are not published as binary blobs because the original custom wrapper source/provenance is incomplete and the recovered HIP runtime contains third-party AMD binaries. See [`docs/CUSTOM_OVERLAY.md`](docs/CUSTOM_OVERLAY.md).

## Found a bug or tested another GPU?

Please publish an issue. Failed tests are useful too.

```powershell
.\scripts\gpu-scan.ps1 -OutputPath .\gpu-report.json
.\scripts\test-runtime.ps1
```

Then open a [GPU compatibility report](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/new?template=gpu-compatibility.yml) and include the application, result and first useful error/output.

## Repository layout

```text
scripts/              install, diagnostics, scanner, staging and launcher
manifests/            pinned versions, hashes and GPU architecture metadata
docs/                 validation, architecture, benchmarks and troubleshooting
examples/             integration/reference snippets
.runtime/             generated dependencies and reports; ignored by Git
local-artifacts/      local archival files; ignored by Git
```

## Limitations

- Only RX 9060 XT / `gfx1200` is currently validated by this project.
- ZLUDA is not a complete CUDA implementation.
- Windows exposes only a subset of the full ROCm ecosystem.
- cuDNN/MIOpen is not available in the validated stable HIP SDK path.
- NCCL, TensorRT, unsupported PTX behavior and some custom CUDA extensions may fail.
- `ZLUDA_CC=8.6` is a CUDA-facing compatibility value, not the AMD GPU architecture.

## License and third-party software

Project-owned scripts and documentation are MIT licensed. ZLUDA, AMD ROCm/HIP, NVIDIA CUDA components and PyTorch/LibTorch retain their own upstream licenses. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

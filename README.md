# CUDA for AMD on Windows

**WORKING REPRODUCIBLE STACK IS NOW UPLOADED.**

Run CUDA-targeted Windows applications on AMD GPUs through ZLUDA + ROCm/HIP.

[![Windows](https://img.shields.io/badge/platform-Windows%20x64-555555)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![AMD](https://img.shields.io/badge/GPU-AMD%20Radeon-555555)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![verify](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml/badge.svg)](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml)

A reproducible Windows CUDA compatibility setup built around **ZLUDA + AMD HIP/ROCm**. It is intended for CUDA-facing compute applications, including workloads that use CUDA-enabled LibTorch.

> [!IMPORTANT]
> **The validated reference remains AMD Radeon RX 9060 XT (`gfx1200`).** Other AMD GPUs are compatibility candidates, not automatically validated devices.
>
> A successful `doctor.ps1` or `cuda_check.exe` run proves that runtime/library surfaces load. It does **not** prove numerical correctness. Non-reference GPUs should also pass `scripts/test-functional.ps1` before their workload output is trusted.

## Verified reference

The public, upstream-only reference path was tested without private/recovered DLLs:

- ZLUDA `v6-preview.69` from the official ZLUDA release
- AMD HIP SDK `6.4`
- LibTorch `2.3.0 + cu118`
- RX 9060 XT / `gfx1200`
- `nvcuda`, cuBLAS, cuBLASLt, cuSPARSE and cuFFT pass `cuda_check`
- a real **2,216,347-parameter PPO network completed forward/inference, PPO learning and optimizer work on the CUDA-facing device**
- one clean validation iteration completed **65,536 timesteps** using the runtime produced by this repository

That integration test used the same CUDA-facing LibTorch training workload that originally motivated this project. See [`docs/VALIDATION.md`](docs/VALIDATION.md).

This does **not** mean every CUDA program or AI model works. CUDA API/library coverage is workload-dependent.

## Compatibility status

| GPU | Target | Status | Notes |
| --- | --- | --- | --- |
| Radeon RX 9060 XT | `gfx1200` | ✅ validated reference | Project integration workload completed |
| Radeon RX 9070 XT | `gfx1201` | ✅ validated external | Archived Windows AMD/ZLUDA training setup from a separately tested RX 9070 XT machine; see [`docs/RX9070XT_VALIDATION.md`](docs/RX9070XT_VALIDATION.md) |
| Radeon 890M | `gfx1150` | 🟡 community partial | HIP 7.2 runtime/GEMM worked; reported `conv2d` hang and incorrect memory-efficient SDPA output in issue #3 |
| Other recognized AMD GPUs | architecture-dependent | ⚪ unverified candidate | Runtime detection is not functional validation |

For `gfx1150`/RDNA 3.5, the project records **HIP SDK 7.2 or newer** as the minimum compatible floor. Do not install HIP 6.4 merely to match the historical RX 9060 XT reference profile.

The memory-efficient SDPA corruption reported on `gfx1150` is **not treated as architecture-specific**: the same probe currently reproduces an incorrect result on the validated `gfx1200` reference path. The repository therefore reports compatibility per capability/workload instead of turning one failing optional backend into a blanket GPU verdict.

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

### 1. Install AMD prerequisites

Install a current AMD GPU driver and a coherent **AMD HIP SDK for Windows including HIP Libraries**.

The correct HIP version depends on the GPU architecture. The installer now checks the architecture-specific minimum recorded in `manifests/windows-gpu-profiles.json` instead of assuming the historical reference version is valid for every GPU.

AMD Windows HIP SDK guide:
https://rocm.docs.amd.com/projects/install-on-windows/en/latest/

### 2. Clone and run the installer

```powershell
git clone https://github.com/Speedstu/CUDA-for-AMD-Windows.git
cd CUDA-for-AMD-Windows
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

`install.ps1` will:

1. detect the AMD GPU and native `gfxXXXX` target;
2. verify the AMD driver/HIP SDK and required math libraries;
3. enforce an architecture-aware HIP version floor when one is recorded;
4. download the pinned official ZLUDA Windows build;
5. download LibTorch `2.3.0+cu118` unless skipped;
6. verify downloaded SHA-256 hashes;
7. generate `.runtime\runtime-config.json` and `.runtime\gpu-report.json`;
8. run ZLUDA's `cuda_check.exe` as a **runtime smoke test**;
9. run numerical functional checks automatically when a Python environment with PyTorch is available.

If you already have an isolated CUDA-facing PyTorch environment, pass it explicitly so the installer can run the numerical suite:

```powershell
.\scripts\install.ps1 -FunctionalPython C:\path\to\venv\Scripts\python.exe
```

If you do not need LibTorch:

```powershell
.\scripts\install.ps1 -SkipLibTorch
```

## Runtime smoke vs functional correctness

There are now two separate validation layers.

### Runtime smoke

```powershell
.\scripts\test-runtime.ps1
```

This checks whether ZLUDA can expose/load the CUDA-facing driver and major libraries. A timeout is now treated as a failure even if earlier probes printed `OK`.

### Numerical functional validation

```powershell
.\scripts\test-functional.ps1 -PythonExe C:\path\to\venv\Scripts\python.exe -Strict
```

Each operation runs in its **own process with a timeout**, so a hanging backend cannot block the entire validation suite. Current probes cover:

- FP32 and FP16 matrix multiplication with CPU reference comparison;
- `conv2d` with CPU reference comparison;
- SDPA math backend with CPU reference comparison;
- SDPA memory-efficient backend with CPU reference comparison.

A backend that cleanly refuses unsupported work is reported as `unsupported` rather than numerically wrong. A returned tensor that exceeds tolerance, contains non-finite data, errors unexpectedly, or hangs is a functional failure.

Reports are written to:

```text
.runtime\runtime-test.json
.runtime\functional-test.json
```

For broader bring-up work, `scripts/test-capabilities.ps1` adds isolated probes for CUDA-facing runtime/device behavior, memory copies, streams/events, GEMM variants, convolution, FFT, sparse operations, linear algebra, RNG, AMP, optimizers, SDPA and NVML. It records numerical failures, clean unsupported results and process hangs separately instead of reducing compatibility to a single load test.

`core_correctness_ok` represents the dense/GEMM path used by the validated PPO workload. `correctness_ok` remains stricter and covers every probed capability. `-Strict` fails if any tested capability is incorrect, errors, or hangs; this is useful when validating a broader CUDA application rather than the PPO reference profile.

## Experimental ZLUDA v7 patch set

An experimental source patch for ZLUDA `v7-preview.10` / commit `9c8b43f` is available under [`patches/zluda-v7-preview10`](patches/zluda-v7-preview10/README.md). It is separate from the stable installer and does not change the current default runtime.

On the RX 9060 XT / `gfx1200` development system, the current patch set has numerically validated additional CUDA-facing paths through AMD libraries:

- common cuFFT and cuFFT Xt paths through hipFFT, including real/complex, FP32/FP64 and 2D round-trips;
- cuSPARSE paths needed by PyTorch sparse matrix multiplication, including stream binding, COO→CSR conversion and CSR descriptor creation through rocSPARSE;
- basic Windows NVML initialization, device enumeration/name and memory reporting through the ZLUDA CUDA driver rather than a separate direct-HIP context.

The NVML backend intentionally queries `nvcuda.dll`/ZLUDA instead of initializing HIP independently; this avoids a Windows context interaction that previously caused `cusparseCreate`/rocSPARSE handle creation to fail. A combined strict regression now passes NVML + `torch.sparse.mm` + FFT together. These are experimental results for the tested stack, not a claim of complete CUDA coverage. CUDA Graphs are not changed by this patch set and are tracked separately.

## Optional real PPO integration smoke

Maintainers with a local VelocityRL checkout can validate the same runtime with a real PPO update instead of relying only on synthetic kernels:

```powershell
.\scripts\test-velocityrl.ps1 -VelocityRoot D:\VelocityRL -Agents 4096 -Rollout 16 -Minibatch 16384 -SmokeUpdates 1
```

This runs VelocityRL through the ZLUDA/HIP runtime produced by this repository, performs rollout + forward + PPO backward/optimizer work, writes its temporary run under `.runtime\velocityrl-smoke\`, and records `.runtime\velocityrl-smoke.json`. VelocityRL is an optional external integration workload and is not downloaded by the installer.

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
.\scripts\gpu-scan.ps1
.\scripts\doctor.ps1
.\scripts\check-hip-compat.ps1
.\scripts\test-runtime.ps1
.\scripts\test-functional.ps1 -PythonExe C:\path\to\venv\Scripts\python.exe
```

The GPU scanner records the model, `gfx` architecture, driver and HIP information. It does not intentionally collect usernames, tokens or user files.

Example statuses:

```text
AMD Radeon RX 9060 XT -> gfx1200 -> validated-reference
AMD Radeon RX 9070 XT -> gfx1201 -> validated-external
AMD Radeon 890M       -> gfx1150 -> community-partial
```

The scanner recognizes other Windows HIP architecture families and marks them as **unverified candidates** rather than claiming support. Detection is not proof that a workload runs.

AMD's current Windows hardware table:
https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html

## Runtime coverage on the validated reference

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

Successful and failed reports are both useful. Please include both runtime and functional reports when possible:

```powershell
.\scripts\gpu-scan.ps1 -OutputPath .\gpu-report.json
.\scripts\test-runtime.ps1
.\scripts\test-functional.ps1 -PythonExe C:\path\to\venv\Scripts\python.exe
```

Then open a [GPU compatibility report](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/new?template=gpu-compatibility.yml) and include the application, result and first useful error/output.

## Repository layout

```text
scripts/              install, diagnostics, scanner, staging, launcher and functional probes
manifests/            pinned versions, hashes and GPU architecture metadata
patches/              experimental source patches for upstream compatibility layers
docs/                 validation, architecture, benchmarks and troubleshooting
examples/             integration/reference snippets
.runtime/             generated dependencies and reports; ignored by Git
local-artifacts/      local archival files; ignored by Git
```

## Limitations

- Only RX 9060 XT / `gfx1200` is currently a fully validated project reference.
- ZLUDA is not a complete CUDA implementation.
- Passing `cuda_check` does not establish numerical correctness.
- Windows exposes only a subset of the full ROCm ecosystem.
- cuDNN/MIOpen is not available in the validated stable HIP SDK path.
- NCCL, TensorRT, unsupported PTX behavior and some custom CUDA extensions may fail.
- `ZLUDA_CC=8.6` is a CUDA-facing compatibility value, not the AMD GPU architecture.

## License and third-party software

Project-owned scripts and documentation are MIT licensed. ZLUDA, AMD ROCm/HIP, NVIDIA CUDA components and PyTorch/LibTorch retain their own upstream licenses. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

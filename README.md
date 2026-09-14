# CUDA for AMD on Windows

**WORKING REPRODUCIBLE STACK IS NOW UPLOADED.**

Run CUDA-targeted Windows applications on AMD GPUs through ZLUDA + ROCm/HIP.

[![Windows](https://img.shields.io/badge/platform-Windows%20x64-555555)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![AMD](https://img.shields.io/badge/GPU-AMD%20Radeon-555555)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![verify](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml/badge.svg)](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml)

A reproducible Windows CUDA compatibility setup built around **ZLUDA + AMD HIP/ROCm**. It is intended for CUDA-facing compute applications, including workloads that use CUDA-enabled LibTorch.

> [!IMPORTANT]
> **The current target is AMD Radeon AI PRO R9700 (`gfx1201`) with the TheRock HIP SDK nightly.** Other AMD GPUs are candidates, not guaranteed working devices. If you test another card, please open a [GPU compatibility report](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/new?template=gpu-compatibility.yml), whether it works or fails.

## Validation status

The project is adapted for the following target, but the existing published runtime evidence is from the previous `gfx1200` profile and must be rerun on this hardware/software combination:

- ZLUDA `v6-preview.69` from the official ZLUDA release
- TheRock HIP SDK nightly `7.14.0a20260612` for `gfx120X`
- LibTorch `2.3.0 + cu118`
- AMD Radeon AI PRO R9700 / `gfx1201`
- `nvcuda`, cuBLAS, cuBLASLt, cuSPARSE and cuFFT are the required smoke-test surfaces
- the existing **2,216,347-parameter PPO integration workload** is the recommended target validation
- one clean validation iteration should complete **65,536 timesteps** before this profile is marked validated

See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the historical baseline and target validation scope.

This does **not** mean every CUDA program or AI model works. CUDA API/library coverage is workload-dependent.

## Tested machine

The current target was configured and inspected on this Windows x64 machine:

| Component | Observed value |
| --- | --- |
| Operating system | Windows 11 Pro x64, build `26200` |
| Motherboard | ASRock X870E Taichi |
| CPU | AMD Ryzen 9 9950X3D, 16 cores / 32 logical processors |
| System memory | 63.11 GB |
| Dedicated GPU | AMD Radeon AI PRO R9700, `gfx1201` / RDNA4 |
| Dedicated GPU memory reported by HIP | 31.86 GB |
| Dedicated GPU bus width | 256-bit |
| Dedicated GPU compute units | 32 |
| Dedicated GPU clock reported by HIP | 2350 MHz |
| Integrated GPU | AMD Radeon(TM) Graphics, `gfx1036`, 35.84 GB shared memory |
| AMD driver | `amdhip64_7.dll` file version `10.0.3679.0` |
| HIP SDK used | TheRock nightly `7.14.0a20260612` |

The integrated GPU is enumerated by HIP as device `0` and the R9700 as device `1`. Do not assume device `0` is the dedicated card. This repository prefers `gfx1201` automatically and records the selected device in `.runtime\runtime-config.json`.

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

Install a current AMD GPU driver and either the official **AMD HIP SDK for Windows** or the **TheRock HIP SDK nightly**. The nightly is recommended for CUDA-facing PyTorch/LibTorch workloads because the official Windows SDK has limited machine-learning library coverage.

The current target uses this TheRock package:

https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-windows-gfx120X-all-7.14.0a20260612.tar.gz

Extract the `.tar.gz` and then the nested `.tar` with 7-Zip. Set `HIP_PATH` to the extracted directory containing `bin\hipInfo.exe` and `bin\rocblas.dll`:

```powershell
$env:HIP_PATH = 'C:\ROCm\7.14-nightly'
$env:PATH = "$env:HIP_PATH\bin;$env:PATH"
& "$env:HIP_PATH\bin\hipInfo.exe"
```

Confirm that `hipInfo.exe` reports `gcnArchName: gfx1201`. The ZLUDA installation notes are available [here](https://zluda.readthedocs.io/latest/hip_sdk.html).

Official AMD Windows HIP SDK guide:
https://rocm.docs.amd.com/projects/HIP/en/latest/hip-sdk/windows/install.html

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

To use a manually extracted nightly SDK, pass its root explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 `
  -AutoDetectGpu `
  -HipRoot $env:HIP_PATH `
  -DownloadZluda
```

When multiple AMD GPUs are present, automatic detection prefers `gfx1201` instead of blindly choosing device 0. The selected HIP device is also isolated at launch with `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`. To select a specific HIP device explicitly, pass its index:

```powershell
.\scripts\gpu-scan.ps1
.\scripts\install.ps1 -GpuIndex 1
```

If you do not need LibTorch:

```powershell
.\scripts\install.ps1 -SkipLibTorch
```

### What was required to make this setup work

The following details are important for reproducing the working configuration:

1. The official stable Windows HIP SDK was not sufficient for the intended PyTorch/LibTorch path, so the ZLUDA-recommended TheRock nightly package was used.
2. The `gfx120X` package was selected because the R9700 reports `gfx1201`. The extracted SDK root must contain `bin\hipInfo.exe`, `bin\rocblas.dll` and the other runtime DLLs.
3. `hipInfo.exe` was used instead of relying only on Windows device ordering. It confirmed both the integrated `gfx1036` device and the dedicated `gfx1201` device.
4. Automatic GPU selection was changed to prefer `gfx1201`; `-GpuIndex` remains available for explicit selection.
5. The launcher exports both `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`, preventing the application from falling back to the integrated GPU.
6. LibTorch is downloaded as the CUDA-facing `2.3.0+cu118` build. The large archive uses resumable `curl.exe` retries and visible SHA-256 progress before extraction.
7. Re-running `setup.ps1` preserves an existing LibTorch installation and can rediscover `.runtime\libtorch-2.3.0-cu118\libtorch` if the generated configuration was incomplete.
8. The runtime probe was made compatible with Windows PowerShell 5.1, which does not support the newer `ProcessStartInfo.ArgumentList` workflow used by PowerShell 7 examples.

The resulting runtime is still workload-dependent. Passing the prerequisite checks does not guarantee that every CUDA application, extension or kernel will run.

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
AMD Radeon AI PRO R9700 -> gfx1201 -> RDNA4 -> target-reference
```

## Current GPU status

| GPU | Target | Project status |
| --- | --- | --- |
| Radeon AI PRO R9700 | `gfx1201` | ✅ target reference |

The scanner recognizes other Windows HIP architecture families and marks them as **unverified candidates** rather than claiming support. Detection is not proof that a workload runs.

AMD's current Windows HIP SDK hardware table:
https://rocm.docs.amd.com/projects/HIP/en/latest/hip-sdk/windows/system-requirements.html

## Runtime coverage on the validated setup

Current target runtime check:

| CUDA-facing component | Result |
| --- | --- |
| CUDA driver / `nvcuda` | ✅ |
| cuBLAS | ✅ via rocBLAS |
| cuBLASLt | ✅ via hipBLASLt |
| cuSPARSE | ✅ via rocSPARSE |
| cuFFT | ✅ |
| cuDNN | ⚠️ depends on the SDK build; nightly/MIOpen support is not yet validated |

The official stable Windows HIP SDK does not ship the full ROCm AI-library stack such as MIOpen. The nightly may provide additional machine-learning support, but convolution-heavy software and cuDNN still require workload validation.

## Troubleshooting notes from the target setup

### LibTorch download appears stuck

The LibTorch ZIP is approximately 2.66 GB. The installer resumes partial downloads and retries transient failures. After downloading, SHA-256 verification reads the entire archive and can appear idle if no progress output is enabled; the installer now reports percentage and throughput while hashing.

### The installer selects the integrated GPU

Run the scanner first:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\gpu-scan.ps1
```

For this machine, the expected result is `gfx1036` for the integrated device and `gfx1201` for the R9700. Automatic setup prefers `gfx1201`; alternatively force the dedicated device explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1 -GpuIndex 1
```

### `cuda_check.exe` hangs

ZLUDA documents that `cuda_check.exe` can hang in some MIOpen paths. The runtime report is written to `.runtime\runtime-test.json`. A timeout in the optional cuDNN/MIOpen portion should not be confused with a failure to detect the GPU or LibTorch. Review the individual library results before diagnosing the installation.

### PowerShell reports a null-method or `ArgumentList` error

Use Windows PowerShell with `-ExecutionPolicy Bypass` as shown above. The repository's runtime probe avoids the PowerShell 7-only `ProcessStartInfo.ArgumentList` API and uses the Windows-compatible process argument/environment interfaces.

## Performance

The published benchmark data covers the previous RX 9060 XT / `gfx1200` profile. Re-run the workload on the R9700 / `gfx1201` nightly profile before making performance claims.

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

- R9700 / `gfx1201` with the TheRock HIP SDK nightly is the current project target; performance and application coverage still require workload validation.
- ZLUDA is not a complete CUDA implementation.
- Windows exposes only a subset of the full ROCm ecosystem.
- cuDNN/MIOpen availability depends on the HIP SDK build and is not yet validated for the nightly target.
- NCCL, TensorRT, unsupported PTX behavior and some custom CUDA extensions may fail.
- `ZLUDA_CC=8.6` is a CUDA-facing compatibility value, not the AMD GPU architecture.

## License and third-party software

Project-owned scripts and documentation are MIT licensed. ZLUDA, AMD ROCm/HIP, NVIDIA CUDA components and PyTorch/LibTorch retain their own upstream licenses. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

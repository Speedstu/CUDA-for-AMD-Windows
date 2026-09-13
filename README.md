# CUDA for AMD on Windows

IM UPLOADING MY WORKING STACK RN 

Run CUDA-targeted Windows applications on AMD GPUs through ZLUDA + ROCm/HIP.

[![Windows](https://img.shields.io/badge/platform-Windows%20x64-555555)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![AMD](https://img.shields.io/badge/GPU-AMD%20Radeon-555555)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![verify](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml/badge.svg)](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml)

This project packages a Windows CUDA compatibility stack around ZLUDA, HIP/ROCm and CUDA-facing libraries. The goal is simple: make software built for CUDA usable on AMD hardware without pretending AMD implements CUDA natively.

> [!IMPORTANT]
> **The project has only been validated on an AMD Radeon RX 9060 XT (`gfx1200`) so far.** Other GPUs are experimental until someone reports a successful run. If you try another AMD GPU, please open a [GPU compatibility report](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/new?template=gpu-compatibility.yml), even if it fails. Those reports are how the compatibility matrix will grow.

## How it works

```text
CUDA application
      |
    ZLUDA
      |
cuBLAS / CUDA compatibility layer
      |
HIP / rocBLAS / hipBLASLt
      |
   AMD GPU
```

Reference configuration:

| Component | Version |
| --- | --- |
| Tested GPU | RX 9060 XT / `gfx1200` |
| ZLUDA | `v6-preview.69` |
| ROCm toolchain used by reference setup | `6.4` |
| Recovered HIP runtime overlay | `7.13` |
| LibTorch | `2.3.0 + cu118` |
| CUDA headers/runtime metadata | `11.8.89` |
| CUDA capability exposed through ZLUDA | `8.6` |

The exact upstream archives used by the reference setup are pinned in [`manifests/upstream-assets.sha256`](manifests/upstream-assets.sha256).

## GPU scanner

The scanner detects the installed AMD GPU, HIP installation and native LLVM target (`gfxXXXX`). When HIP is installed it reads the target directly from AMD's `hipInfo.exe`; if HIP is not installed yet it can fall back to Windows GPU information for known cards.

```powershell
.\scripts\gpu-scan.ps1
```

Save a report for an issue:

```powershell
.\scripts\gpu-scan.ps1 -OutputPath .\gpu-report.json
```

Example on the reference machine:

```text
index  name                    gfx      generation  detection  project_status
0      AMD Radeon RX 9060 XT   gfx1200  RDNA4       hipInfo    validated-reference
```

The report records GPU model, `gfx` target, driver, HIP version and detection status. It does not intentionally collect usernames, tokens or user files.

## Automatic setup

For a new machine, let the setup script scan the GPU and build a runtime configuration automatically:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 `
  -AutoDetectGpu `
  -DownloadZluda `
  -DownloadLibTorch
```

This creates `.runtime\gpu-report.json` and `.runtime\runtime-config.json`.

Then stage the compatibility DLLs beside an application:

```powershell
.\scripts\stage-runtime.ps1 -TargetDir C:\path\to\your-app
```

Launch through ZLUDA:

```powershell
.\scripts\run-zluda.ps1 -Program C:\path\to\your-app\app.exe
```

Verify the local runtime:

```powershell
.\scripts\verify.ps1
```

Multiple AMD GPUs can be selected with `-GpuIndex`:

```powershell
.\scripts\setup.ps1 -AutoDetectGpu -GpuIndex 1 -DownloadZluda
```

## GPU status

**Validated by this project** currently means exactly one configuration:

| GPU | Target | Status |
| --- | --- | --- |
| Radeon RX 9060 XT | `gfx1200` | ✅ validated reference |

The scanner also recognizes current Windows HIP SDK architecture families and marks them as **unverified candidates**, not as confirmed working GPUs. Current AMD documentation lists Windows HIP SDK support for RDNA4/RDNA3 targets including `gfx1200`, `gfx1201`, `gfx1100`, `gfx1101`, `gfx1102`, plus supported RDNA3.5 APU targets such as `gfx1150`/`gfx1151`.

RDNA2 `gfx103x` cards can still be detected, but current AMD Windows HIP SDK documentation does not list those Radeon cards as supported by the current SDK, so they are reported as experimental.

AMD's current Windows hardware table: https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html

## What has worked on the reference GPU

The RX 9060 XT setup has run real CUDA-enabled LibTorch workloads under Windows, including:

- CUDA-facing LibTorch code
- neural-network inference
- PPO / reinforcement-learning training
- GEMM-heavy training through cuBLAS-compatible calls
- long-running training with checkpoints

Retained PPO benchmark summaries were roughly **70k–109k overall steps/s**, with faster individual warmed iterations. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

This does **not** mean every CUDA program works. Compatibility depends on the CUDA APIs and libraries an application uses. NCCL, TensorRT, custom CUDA extensions, unsupported PTX behavior and incomplete CUDA-library paths can still fail.

## Custom reference overlay

The original RX 9060 XT environment also used a recovered custom cuBLAS/cuBLASLt + HIP runtime overlay. Its binaries are fingerprinted in [`manifests/recovered-artifacts.sha256`](manifests/recovered-artifacts.sha256) but are intentionally not committed to the repository.

`-AutoDetectGpu` does **not** apply this recovered overlay automatically to other GPUs. This is deliberate: a `gfx1200`-validated runtime should not silently be treated as universal.

If the local reference files are available, the exact setup can be reconstructed with:

```powershell
.\scripts\setup.ps1 `
  -AutoDetectGpu `
  -ZludaRoot C:\path\to\zluda `
  -LibTorchRoot C:\path\to\libtorch `
  -UseRecoveredCustomOverlay
```

Using that overlay on anything except the validated RX 9060 XT is explicitly experimental and produces a warning.

## Found a bug or tested another GPU?

Please publish an issue. Failed tests are useful too.

1. Run `scripts\gpu-scan.ps1 -OutputPath .\gpu-report.json`.
2. Open a [GPU compatibility report](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/new?template=gpu-compatibility.yml).
3. Include what you ran, whether it launched, and the first useful error/output.
4. Attach `gpu-report.json` if possible.

The goal is to turn community reports into an actual Windows AMD compatibility matrix instead of guessing which cards work.

## Repository layout

```text
scripts/              scanner, setup, staging, launch and verification
manifests/            pinned versions, hashes and GPU architecture metadata
docs/                 architecture, benchmarks and troubleshooting
examples/             manual LibTorch/CUDA integration examples
local-artifacts/      local recovered binaries; ignored by Git
.runtime/             generated runtime/configuration; ignored by Git
```

## Limitations

- Only RX 9060 XT / `gfx1200` is currently validated by this project.
- ZLUDA is not a complete CUDA implementation.
- Windows exposes only a subset of the full ROCm ecosystem.
- `ZLUDA_CC=8.6` is a CUDA-facing compatibility value, not the AMD architecture.
- Mixing HIP/ROCm generations can be fragile.
- The scanner can identify a GPU; detection alone is not proof that a CUDA workload will run.

## License and third-party software

Project-owned scripts and documentation are MIT licensed. ZLUDA, AMD ROCm/HIP, NVIDIA CUDA components and PyTorch/LibTorch retain their own upstream licenses. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

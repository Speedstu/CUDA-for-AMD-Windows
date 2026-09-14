# Validation

This document records what was actually tested, rather than what is assumed to work.

## Public reproducible path

The operational target is TheRock HIP SDK nightly **7.14.0a20260612** on `gfx1201`; the previous `gfx1200`/HIP 6.4 run below is retained as historical evidence and must not be read as validation of the new target.

Hardware/software:

- Windows x64
- AMD Radeon AI PRO R9700 (`gfx1201`) — target hardware
- ZLUDA `v6-preview.69`, official Windows release asset
- TheRock HIP SDK nightly `7.14.0a20260612` — target SDK
- LibTorch `2.3.0+cu118`
- `ZLUDA_CC=8.6`
- **no recovered/custom overlay DLLs**

The target runtime should be created from the same public path exposed by `scripts/install.ps1`: official ZLUDA plus the installed TheRock HIP SDK. The repository's ignored `local-artifacts/` directory is not required.

## ZLUDA runtime check

ZLUDA's `cuda_check.exe` completed and reported the following core surfaces.
The cuDNN line is retained as a **historical stable-HIP-SDK result**, not as
evidence for the current TheRock nightly target:

```text
nvcuda     OK
cuBLAS     OK -> rocBLAS
cuBLASLt   OK -> hipBLASLt
cuSPARSE   OK -> rocSPARSE
cuFFT      OK
cuDNN 8/9  historical: unavailable on the stable Windows HIP SDK configuration
```

`test-runtime.ps1` treats the first five groups as core runtime checks. cuDNN is reported separately because availability depends on the selected HIP SDK build.

## Real training integration test

The generated public runtime was then staged next to an existing CUDA-enabled
LibTorch PPO trainer in the historical baseline run below.

Observed during the validation run:

```text
Using CUDA GPU device...
Model parameters: 2,216,347
Fused CUDA LayerNorm+LeakyReLU kernel ready
Fused CUDA warp LayerNorm+LeakyReLU kernel ready
Fused CUDA masked categorical sampler ready

Collection Steps/Second: 32,420
Consumption Steps/Second: 11,389
Overall Steps/Second: 8,428
PPO Learn Time: 5.5658 s
Collected Timesteps: 65,536
Total Timesteps: 65,536
Total Iterations: 1
```

The process was stopped after the completed iteration because the purpose of this run was reproducibility validation, not a throughput benchmark.

The historical run verifies more than device enumeration: the workload performed CUDA-facing inference plus a real PPO learning/update phase using CUDA-enabled LibTorch on the AMD GPU stack. Repeat this probe on `gfx1201`/TheRock HIP SDK 7.14 before treating the target as validated.

## Historical performance

Older tuned runs of the same ZLUDA/LibTorch family retained approximately **70k-109k overall steps/s**. Those numbers are historical performance evidence and should not be confused with the short validation run above.

See `BENCHMARKS.md` for the retained performance notes.

## What this does not prove

This test does not claim universal CUDA compatibility. In particular:

- applications that require cuDNN can fail when the selected HIP SDK build lacks MIOpen;
- unsupported PTX/CUDA APIs may fail;
- custom CUDA extensions are workload-specific;
- other AMD GPU architectures remain unverified until tested.

The compatibility issue template exists specifically to grow evidence one GPU/application at a time.

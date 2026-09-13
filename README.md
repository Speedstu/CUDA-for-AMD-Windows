# ZLUDA AMD Windows Custom

Recovered Windows AMD CUDA-compatibility stack used for real LibTorch/CUDA workloads on an AMD Radeon RX 9060 XT (gfx1200).

This is a forensic reconstruction of a working setup, not a claim that AMD GPUs natively implement CUDA. CUDA-facing applications are loaded through ZLUDA; execution is backed by AMD HIP/ROCm.

## Exact recovered profile

- Windows x64
- AMD Radeon RX 9060 XT (`gfx1200`)
- ZLUDA **v6-preview.69** core
- installed ROCm/HIP **6.4** toolchain
- recovered HIP **7.13** runtime overlay (`amdhip64_7.dll`, build 3581) + `rocm_kpack.dll`
- LibTorch **2.3.0+cu118**
- CUDA 11.8 build headers/runtime metadata (`11.8.89`)
- `ZLUDA_CC=8.6`
- recovered custom cuBLAS proxy + cuBLASLt forwarding/heuristic shim

```text
CUDA-facing application / LibTorch cu118
                |
              ZLUDA
         v6-preview.69 core
                |
        custom cuBLAS layer
                |
      HIP / rocBLAS / hipBLASLt
                |
          AMD Radeon GPU
```

The original staging script copied the ZLUDA core first, then overlaid HIP 7.13 runtime pieces and custom BLAS DLLs before launching with `zluda.exe -- <program>`.

## Recovered custom behavior

`cublas64_11.dll` is a custom proxy capable of routing GEMM/GemmEx toward rocBLAS and optionally hipBLASLt. Recovered switches include `ZLUDA_CUBLAS_USE_HIPBLASLT`, `ZLUDA_CUBLAS_AUTOTUNE`, `ZLUDA_CUBLAS_WORKSPACE_MB` and `ZLUDA_CUBLAS_SOLUTION_MAP`.

`cublasLt64_11/12/13.dll` are forwarding/heuristic shims. The binary embeds the original source filename `cublasLtShim.c` and switches including `HUMAN_CUBLASLT_SHIM_STATS` and `HUMAN_CUBLASLT_FORCE_WORKSPACE_MB`.

The original source files for these wrappers have not been recovered. Their exact binaries are preserved locally and fingerprinted in `manifests/recovered-artifacts.sha256`; they are intentionally not Git-tracked by default until provenance/licensing is established.

## Performance evidence

On the recovered RX 9060 XT, the ZLUDA + LibTorch 2.3.0 cu118 path produced real PPO training at roughly **70k-109k overall steps/s** in the retained benchmark summary. Individual warmed iterations exceeded that range. See `docs/BENCHMARKS.md`.

A later native-HIP rewrite was faster, but it is a separate implementation.

## Quick start

Download LibTorch 2.3.0+cu118 and prepare a runtime tree:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -DownloadZluda -DownloadLibTorch
```

For an exact recovered reconstruction, supply the pinned ZLUDA and HIP roots and use the locally recovered overlay:

```powershell
.\scripts\setup.ps1 `
  -ZludaRoot C:\path\to\zluda-v6-preview69\zluda `
  -HipRoot 'C:\Program Files\AMD\ROCm\6.4' `
  -UseRecoveredCustomOverlay
```

Stage DLLs beside an application:

```powershell
.\scripts\stage-runtime.ps1 -TargetDir C:\path\to\your-app
```

Launch:

```powershell
.\scripts\run-zluda.ps1 -Program C:\path\to\your-app\app.exe
```

Verify recovered files:

```powershell
.\scripts\verify.ps1
```

## Layout

```text
docs/                 architecture, recovery notes, benchmarks, troubleshooting
examples/             generic manual LibTorch/CUDA integration
manifests/            exact recovered versions and SHA-256 records
scripts/              setup, staging, launch and verification
local-artifacts/      exact recovered files on this machine; ignored by Git
.runtime/             generated runtime/dependency tree; ignored by Git
```

## Known limitations

- ZLUDA is not a complete CUDA implementation; compatibility is workload-dependent.
- The recovered profile was tuned for gfx1200 and LibTorch 2.3.0+cu118.
- `ZLUDA_CC=8.6` is a CUDA-facing compatibility capability, not AMD's native architecture name.
- Windows ZLUDA has historically had incomplete support for some CUDA libraries such as FFT paths.
- Mixing HIP/ROCm generations is fragile. Reproduce the historical mix only when needed; new deployments should prefer a coherent supported HIP SDK first.

## Third-party software

ZLUDA, AMD ROCm/HIP, NVIDIA CUDA components and PyTorch/LibTorch keep their upstream licenses. This repository does not relicense them. See `THIRD_PARTY_NOTICES.md`.

## Recovery status

**Recovered:** runtime topology, exact versions, build strategy, environment, custom wrapper binaries, portable deployment design and benchmark evidence.

**Not recovered:** original source used to compile the custom cuBLAS proxy and `cublasLtShim.c`.

## Current reconstruction test

The reconstructed runtime currently reaches AMD hipBLASLt in the recovered cuda_check.exe probe, but that probe times out after printing a successful hipBLASLt load. See docs/SMOKE_TEST.md; this is intentionally reported as a partial smoke rather than a full pass.


## Pinned upstream assets

The exact official ZLUDA v6-preview.69 Windows archive and LibTorch 2.3.0+cu118 archive are pinned by filename, URL, size and SHA-256 in manifests/upstream-assets.sha256. CI redownloads and verifies the ZLUDA asset on every push/PR; LibTorch is pinned but not downloaded in CI because the archive is ~2.66 GB.

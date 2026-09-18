# CUDA for AMD on Windows

[![Windows](https://img.shields.io/badge/platform-Windows%20x64-555555)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![AMD Radeon](https://img.shields.io/badge/GPU-AMD%20Radeon-ED1C24)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![verify](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml/badge.svg)](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A reproducible Windows CUDA compatibility stack for AMD GPUs, built around ZLUDA + AMD ROCm/HIP.**

Run CUDA-targeted Windows applications on AMD hardware, verify that work really reaches the GPU, and distinguish a correct result from a crash, timeout, CPU fallback, or silently wrong tensor.

> [!IMPORTANT]
> This project is **not a complete CUDA implementation**. Support is capability- and workload-specific.
> The validated reference GPU is currently the **Radeon RX 9060 XT (`gfx1200`)**.

## Quick start

### 1. Install AMD prerequisites

Install a current AMD GPU driver and the **AMD HIP SDK for Windows with HIP Libraries**.

The required HIP version depends on the GPU architecture. The project checks the architecture-specific floor recorded in [`manifests/windows-gpu-profiles.json`](manifests/windows-gpu-profiles.json).

### 2. Clone and install

```powershell
git clone https://github.com/Speedstu/CUDA-for-AMD-Windows.git
cd CUDA-for-AMD-Windows
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

The installer detects the GPU, verifies the HIP environment, downloads pinned assets, stages the runtime, runs a runtime smoke test, and performs numerical checks when a compatible Python/PyTorch environment is available.

### 3. Run a CUDA-targeted application

```powershell
.\scripts\run-zluda.ps1 -Program C:\path\to\app.exe
```

### 4. Validate the runtime

```powershell
.\scripts\doctor.ps1
.\scripts\test-runtime.ps1
.\scripts\test-functional.ps1 -PythonExe C:\path\to\venv\Scripts\python.exe
```

The original low-level setup path is intentionally kept compatible for existing users and old posts:

```powershell
.\scripts\setup.ps1 -DownloadZluda -DownloadLibTorch
```

See [`scripts/README.md`](scripts/README.md) for the stable user-facing commands and maintainer tooling.

## Current compatibility

| GPU | Target | Project status | Notes |
| --- | --- | --- | --- |
| Radeon RX 9060 XT | `gfx1200` | ✅ validated reference | Main development and integration validation platform |
| Radeon RX 9070 XT | `gfx1201` | ✅ validated external | Separately tested Windows AMD/ZLUDA training setup |
| Radeon 890M | `gfx1150` | 🟡 community partial | HIP/GEMM reported working; broader paths still under validation |
| Other recognized AMD GPUs | architecture-dependent | ⚪ unverified candidate | Detection is not functional validation |

Full details, version floors, evidence levels, known limitations, and the **CUDA DLL → AMD backend map**: [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md#cuda-facing-dll-map).

## What has been validated?

### Stable public reference path

The public reference path uses pinned upstream components and has been tested with:

- ZLUDA `v6-preview.69`
- AMD HIP SDK `6.4`
- LibTorch `2.3.0 + cu118`
- Radeon RX 9060 XT / `gfx1200`
- CUDA-facing driver loading plus cuBLAS, cuBLASLt, cuSPARSE and cuFFT smoke coverage
- numerical CPU-vs-GPU correctness probes
- a real **2,216,347-parameter PPO** workload
- a clean validation iteration of **65,536 timesteps**

See [`docs/VALIDATION.md`](docs/VALIDATION.md).

### Experimental v7 path

The repository also maintains a pinned, source-based ZLUDA `v7-preview.10` patch series for newer CUDA-facing behavior.

Validated work on the reference machine includes areas such as:

- CUDA Graph compatibility
- driver metadata and launch probes
- cuFFT and cuSPARSE paths
- NVML compatibility
- experimental cuSOLVER → hipSOLVER bridging
- experimental cuDNN v8 forward/backward convolution → MIOpen bridging on the reference gfx1200 system
- fail-closed handling for unsafe fused SDPA fallbacks
- modern llama.cpp registration and GPU execution

The patch set remains separate from the stable installer so experimental work cannot silently redefine the stable path.

See [`patches/zluda-v7-preview10/README.md`](patches/zluda-v7-preview10/README.md).

## Real application checks

This project deliberately tests more than tiny API calls.

### PyTorch / LibTorch / PPO

The reference workload performs rollout, forward, backward, PPO learning, and optimizer work through the CUDA-facing device.

See [`docs/VALIDATION.md`](docs/VALIDATION.md) and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### llama.cpp

A recent llama.cpp `b10978` build has completed an end-to-end GPU smoke on the RX 9060 XT experimental v7 path:

- AMD GPU exposed as `CUDA0` through ZLUDA
- **6/6 model layers offloaded**
- KV cache on GPU
- compute buffer on GPU
- CUDA Graph warmup reached
- prompt processing and generation completed
- clean process exit, reproduced more than once

This is evidence for that tested path, **not a claim that every llama.cpp kernel or model is supported**.

See [`docs/LLAMA_CPP.md`](docs/LLAMA_CPP.md).

## Performance snapshot

Performance claims are kept same-GPU and reproducible where possible; this is **not** an AMD-vs-NVIDIA benchmark.

On the experimental v7/TheRock path, a clean four-pair FP32 SGEMM validation measured paired median overhead versus direct HIP/rocBLAS of **+1.79% at 1024²**, **+0.45% at 2048²**, and **+2.20% at 4096²**. The corresponding throughput ratios were **98.2%**, **99.6%**, and **97.8%**.

A real VelocityRL `512 agents × rollout 16` smoke separated first-use compilation from steady state: the warmed updates reached **72,306 SPS** and **70,333 SPS**, for **71,319.5 SPS median steady-state**.

Methodology and raw results: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and [`benchmarks/`](benchmarks/).

## Correctness first

A CUDA API returning success is not enough to call something supported.

The test suite distinguishes:

`PASS` · `UNSUPPORTED` · `INCORRECT` · `TIMEOUT` · `ERROR` · process failure

The project follows a fail-closed rule: an explicit unsupported result is better than a plausible-looking but incorrect tensor.

That policy already caught fused SDPA fallback paths that could return numerically wrong output instead of failing clearly.

For the evidence model and contribution rules, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Project map

| Area | Purpose |
| --- | --- |
| [`scripts/`](scripts/README.md) | Installation, launch, diagnostics, tests, staging and maintainer helpers |
| [`manifests/`](manifests/) | Pinned releases, hashes and GPU architecture metadata |
| [`patches/`](patches/) | Reproducible source patches against pinned upstream projects |
| [`native/`](native/) | Small source-built native compatibility components |
| [`docs/`](docs/README.md) | Architecture, validation, integrations, benchmarks and troubleshooting |
| [`examples/`](examples/) | Small integration/reference snippets |
| [`benchmarks/`](benchmarks/) | Raw benchmark data checked into the repository |
| [`.github/`](.github/) | CI and structured GPU compatibility reports |

Generated runtime files stay under `.runtime/` and are ignored by Git.

## Documentation

Start with [`docs/README.md`](docs/README.md) instead of searching through the repository.

- [Compatibility and support levels](docs/COMPATIBILITY.md)
- [Validation methodology and evidence](docs/VALIDATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [llama.cpp validation](docs/LLAMA_CPP.md)
- [RX 9070 XT external validation](docs/RX9070XT_VALIDATION.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Experimental v7 patch set](patches/zluda-v7-preview10/README.md)
- [Experimental cuSOLVER proxy](native/cusolver_proxy/README.md)
- [Experimental cuDNN v8 → MIOpen convolution bridge](docs/CUDNN_BRIDGE.md)

Historical reconstruction material is still preserved, but it is no longer part of the recommended path.

## Found a bug or tested another GPU?

Both successful and failed reports are useful.

```powershell
.\scripts\gpu-scan.ps1 -OutputPath .\gpu-report.json
.\scripts\test-runtime.ps1
.\scripts\test-functional.ps1 -PythonExe C:\path\to\venv\Scripts\python.exe
```

Then open a [GPU compatibility report](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/new?template=gpu-compatibility.yml).

Please include the GPU model, `gfxXXXX` target, driver, HIP version, ZLUDA channel/build, exact workload, and proof that the AMD/ZLUDA device actually executed the work rather than falling back to CPU.

## Known limitations

- ZLUDA does not implement the entire CUDA ecosystem.
- Passing `cuda_check` or device detection alone does not establish numerical correctness.
- Windows exposes only part of the full ROCm ecosystem.
- The validated stable Windows HIP path does not provide a complete cuDNN/MIOpen equivalent stack. A separate experimental v7 bridge validates a narrow cuDNN v8 2D forward + backward-data + backward-filter subset on gfx1200; see docs/CUDNN_BRIDGE.md.
- NCCL, TensorRT, unsupported PTX behavior, custom CUDA extensions and architecture-specific kernels may fail.
- Fused Flash/memory-efficient SDPA paths can depend on NVIDIA cubins; unsafe fallbacks are treated as unsupported rather than accepted as correct.
- `ZLUDA_CC=8.6` is a CUDA-facing compatibility value, not the native AMD GPU architecture.

More detail: [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

## Contributing

The project is evidence-driven. New compatibility claims should include a focused reproducer, numerical validation where applicable, exact versions/hashes, and a real workload when possible.

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing support claims or promoting experimental patches.

## License

Project-owned scripts and documentation are MIT licensed.

ZLUDA, AMD ROCm/HIP, NVIDIA CUDA components, PyTorch/LibTorch and other third-party projects retain their own licenses. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

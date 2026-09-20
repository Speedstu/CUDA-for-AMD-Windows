# Changelog

This file records user-visible project milestones. Detailed experimental evidence remains in the validation documents and Git history.

## Unreleased

### gfx1150 follow-up

- Added a ZLUDA `CUDA_LAUNCH_BLOCKING` / per-kernel trace candidate plus `driver_launch_blocking` to attribute asynchronous GPU faults to the launch that triggered them.
- Added a PTX target-selection candidate plus `driver_ptx_selection`: multi-PTX fatbins now have a testable path to select the highest target compatible with the advertised `ZLUDA_CC`, instead of silently choosing the highest PTX entry.
- Added a `gfx1150` PyTorch conv2d safety fallback: the launcher and capability probe disable the known hanging legacy cuDNN route by default, with an explicit `-AllowUnsafeCudnnConv` A/B override and bridge-aware detection.
- Fixed `project_revision` capture under PowerShell 7 in capability, functional, llama-registration, and trace reports.
- Fixed `-AllowUnsafeFusedSDPA` so inherited safe-mode environment state cannot silently keep fused SDPA disabled; unrelated `PYTHONPATH` entries are preserved.
- Added `driver_buffer_clear`, a focused PTX 7.0/8.4 non-default-stream clear/synchronize/readback probe for the remaining modern llama.cpp kernel-execution boundary.
- Recorded the Radeon 890M / gfx1150 re-test: launch/PTX metadata and SDPA fail-safe fixes reproduce, while conv2d is now confirmed correct through the guarded no-cuDNN fallback; modern llama.cpp kernel execution remains under investigation.

### Experimental cuDNN / MIOpen bridge

- Added a source-built cuDNN v8 compatibility proxy for a narrow 2D forward + backward-data + backward-filter subset on Windows.
- Preserves the original top-level cuDNN export surface while overriding only the validated bridge entry points.
- Added exact MIOpen solution-ID/workspace binding and fail-closed handling for unsupported streams, modes, algorithms and scaling semantics.
- Added header-free forward/backward correctness self-tests plus a PyTorch 2.3.0+cu118 training matrix validated on the RX 9060 XT / gfx1200 reference system, including FP32/FP16, padding, stride, dilation, groups and a bias-training case.
- No third-party cuDNN or MIOpen binaries are committed or redistributed.

### Repository organization

- Simplified the root README around the stable user path.
- Added a documentation index and a dedicated compatibility page.
- Documented the script interface so cleanup work does not break existing commands or old community posts.
- Kept the historical `setup.ps1` entry point alongside the recommended `install.ps1` flow.
- No CUDA runtime semantics are intentionally changed by this repository-organization pass.

## 2026-09-18

### Compatibility and validation

- Extended the pinned ZLUDA v7 experimental path with newer driver metadata/launch validation.
- Added modern llama.cpp registration and end-to-end RX 9060 XT GPU smoke evidence.
- Added fail-closed handling for additional unsafe fused-SDPA fallback behavior.
- Improved reproducibility and candidate-build CI around the pinned source patch series.

For exact hardware, versions and limitations, see [`docs/VALIDATION.md`](docs/VALIDATION.md), [`docs/LLAMA_CPP.md`](docs/LLAMA_CPP.md), and [`patches/zluda-v7-preview10/README.md`](patches/zluda-v7-preview10/README.md).

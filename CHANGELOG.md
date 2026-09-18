# Changelog

This file records user-visible project milestones. Detailed experimental evidence remains in the validation documents and Git history.

## Unreleased

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

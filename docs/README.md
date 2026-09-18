# Documentation

This directory is the documentation hub for **CUDA for AMD on Windows**.

The root [`README`](../README.md) is intentionally kept focused on installation, current status, and the shortest path to a working runtime. Detailed evidence and maintainer material live here.

## Start here

| Document | Use it for |
| --- | --- |
| [`COMPATIBILITY.md`](COMPATIBILITY.md) | GPU status, support levels, runtime coverage and limitations |
| [`VALIDATION.md`](VALIDATION.md) | What has actually been tested and how evidence is interpreted |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Installation/runtime failures and common diagnostics |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the CUDA → ZLUDA → HIP/ROCm stack is structured |

## Integrations and hardware validation

| Document | Scope |
| --- | --- |
| [`LLAMA_CPP.md`](LLAMA_CPP.md) | llama.cpp registration and end-to-end GPU validation |
| [`RX9070XT_VALIDATION.md`](RX9070XT_VALIDATION.md) | Separately tested RX 9070 XT / gfx1201 environment |
| [`SMOKE_TEST.md`](SMOKE_TEST.md) | Small integration smoke procedure |
| [`CUDNN_BRIDGE.md`](CUDNN_BRIDGE.md) | Experimental cuDNN v8 forward/backward convolution → MIOpen bridge and gfx1200 validation |

Real application checks are evidence for the exact versions and hardware recorded in each document. They are not blanket claims for every GPU or CUDA application.

## Performance

- [`BENCHMARKS.md`](BENCHMARKS.md) — benchmark methodology, same-GPU comparisons and workload notes.
- [`../benchmarks/`](../benchmarks/) — raw checked-in benchmark data.

## Experimental development

- [`../patches/zluda-v7-preview10/README.md`](../patches/zluda-v7-preview10/README.md) — pinned ZLUDA v7 source patch series.
- [`../native/cusolver_proxy/README.md`](../native/cusolver_proxy/README.md) — source-built experimental cuSOLVER → hipSOLVER bridge.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — evidence requirements and patch lifecycle.

Experimental components stay separate from the stable installer until their behavior has been validated.

## Historical / reconstruction material

These files are preserved because they document how the project was recovered and compared, but they are **not the recommended installation path**:

- [`CUSTOM_OVERLAY.md`](CUSTOM_OVERLAY.md)
- [`RECOVERY_NOTES.md`](RECOVERY_NOTES.md)

Keeping historical evidence does not mean users need those artifacts for the public reproducible stack.
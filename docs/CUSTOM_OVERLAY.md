# Recovered custom overlay

The historical development environment contained a custom Windows overlay in addition to upstream ZLUDA. It is kept locally for archaeology and A/B testing, but the public runtime does not require it.

## What was recovered

The recovered set contains a modified `cublas64_11.dll`, three small cuBLASLt forwarding/heuristic shims, an AMD HIP 7.13 runtime pair, and `rocm_kpack.dll`. Exact hashes are retained in [`../manifests/recovered-artifacts.sha256`](../manifests/recovered-artifacts.sha256).

Binary inspection found:

- `cublas64_12.dll` and `cublas64_13.dll` are byte-identical to the pinned upstream ZLUDA copies.
- `cublas64_11.dll` differs from upstream and contains ZLUDA Rust build/source strings plus custom rocBLAS/hipBLASLt behavior strings.
- `cublasLt64_11/12/13.dll` identify themselves as a `HumanLikeRL cublasLt shim` and contain the source filename `cublasLtShim.c` in debug information.
- the original `cublasLtShim.c` source was not found on the available disks.
- `amdhip64_7.dll` identifies itself as an AMD HIP 7.13 runtime (`Advanced Micro Devices Inc.`).

## Performance

A controlled 2026-09-13 A/B showed the recovered full overlay about **3.03% slower in median overall SPS** than the clean public upstream path on the reference PPO workload. See [`BENCHMARKS.md`](BENCHMARKS.md).

A separate smoke test staged only the four genuinely different BLAS wrappers (`cublas64_11.dll` plus the three cuBLASLt shims), with **no local `amdhip64_7.dll`, `amdhip64.dll`, or `rocm_kpack.dll`**. On the reference machine it successfully completed one PPO iteration while using the HIP 7 runtime already installed by the AMD driver. This proves the AMD runtime binaries do not need to be bundled just to experiment with the wrappers on that machine.

## Why the binary overlay is not committed

The project does not publish the recovered binary overlay yet. ZLUDA itself is dual MIT/Apache-2.0 licensed, but the exact source/provenance of the local custom modifications and the cuBLASLt shim has not been recovered completely. The AMD runtime files are third-party AMD binaries and are also intentionally not re-hosted here.

The safe path is therefore:

1. keep the hashes and behavior documented;
2. use the clean upstream runtime as the supported default;
3. reconstruct the custom changes from source before publishing a custom binary package.

This avoids presenting opaque recovered DLLs as project-owned redistributable artifacts.

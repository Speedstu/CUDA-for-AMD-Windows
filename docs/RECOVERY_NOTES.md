# Recovery notes

## Timeline

- Early experiments used a `zluda_tmp\zluda` tree and multiple LibTorch CUDA releases.
- A later project-local stack used `hip-sdk`, `zluda`, LibTorch cu118 and manual CMake linkage.
- The strongest surviving snapshot retained multiple ZLUDA variants and an explicit `amd-zluda-custom` overlay.

## Surviving experimental profiles

`zluda_blas_june`, `zluda_cu11_wrapped`, `zluda_cublas_hip7`, `zluda_hybrid`, `zluda_june_cublaswrap`, and `zluda_lt_stub` all survived on disk.

The exact staging script paired `zluda-v6-preview69` with `amd-zluda-custom`.

## Custom binary evidence

The recovered `cublas64_11.dll` contains messages for rocBLAS operation, optional hipBLASLt GemmEx routing and autotuning.

The recovered cuBLASLt shim contains the message `[HumanLikeRL cublasLt shim] failed to load real ZLUDA cublasLt DLL` and embeds the source filename `cublasLtShim.c`.

The original `.c` source and PDB were not found in the surviving project trees, recycle metadata or targeted source/build directories.

## Portable bundle evidence

A separate 6.68 GB deployment bundle from August 2026 preserved the deployment pattern: portable HIP runtime, ZLUDA, LibTorch cu118 rebuild dependencies, and a launcher that stages the DLLs before executing through ZLUDA.

Its portable HIP archive contains a HIP 7.13-era Windows runtime plus gfx1200/gfx1201 hipBLASLt kernel libraries.

## Publication rule

Do not silently invent source for the recovered DLLs. Keep hashes and provenance. If the original source is later recovered, add it separately with its real history/license.

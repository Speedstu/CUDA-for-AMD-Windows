# Validation

This document records what was actually tested, rather than what is assumed to work.

## Public reproducible path

The operational target is TheRock HIP SDK nightly **7.14.0a20260612** on
`gfx1201`. The original `gfx1200` profile remains recognized and preserved;
the `gfx1200`/HIP 6.4 run below is retained as its historical evidence and
must not be read as validation of the new `gfx1201` target.

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

The current TheRock nightly probe was also run against the generated runtime
configuration and reported:

```text
cudnn9: OK (C:\ROCm\7.14-nightly\bin\MIOpen.dll)
cudnn8: OK (C:\ROCm\7.14-nightly\bin\MIOpen.dll)
[PASS] cudnn
```

This supersedes the historical stable-SDK cuDNN result for the `gfx1201`
nightly profile. It does not imply that every stable HIP SDK build provides
the same MIOpen-backed compatibility.

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

The historical run verifies more than device enumeration: the workload performed CUDA-facing inference plus a real PPO learning/update phase using CUDA-enabled LibTorch on the AMD GPU stack. Repeat this probe on `gfx1201`/TheRock HIP SDK 7.14 before making equivalent PPO workload or performance claims. The runtime and native LibTorch build are already validated independently.

## 2026-09-15 VelocityRL regression smoke

The recovered issue-validation work was rerun locally with the repository runtime itself and a real VelocityRL PPO workload.

Runtime smoke:

```text
nvcuda     PASS
cuBLAS     PASS
cuBLASLt   PASS
cuSPARSE   PASS
cuFFT      PASS
cuDNN      unavailable (expected on this stable Windows HIP SDK)
```

Capability probes using VelocityRL's CUDA-facing PyTorch environment (`2.0.1+cu118`) produced:

```text
matmul                 PASS
conv2d                 UNSUPPORTED (MIOpen.dll unavailable)
sdpa_math               PASS
sdpa_mem_efficient      INCORRECT RESULT
```

At that 2026-09-15 stage, the memory-efficient SDPA failure was reproduced on the RX 9060 XT reference as well, proving it was not a `gfx1150`-only defect. The later experimental v7 patch below changes this specific cubin-only path from a silent numerical failure into an explicit safe refusal.

A real VelocityRL smoke then ran through this repository's ZLUDA launcher with **4,096 agents x 16 rollout steps = 65,536 decisions** and one PPO update:

```text
Checkpoint transfer max_abs: 0
Device: AMD Radeon RX 9060 XT [ZLUDA]
Trainable parameters: 986,239
PPO decisions added: 65,536
SPS: 6,933
KL: 0.00001
Entropy: 1.690
Checkpoint written: yes
```

This is the workload-level gate for the dense/GEMM PPO profile. Synthetic extended probes are still retained because a successful PPO update does not imply that convolution or every attention backend is safe.

## 2026-09-16 experimental v7/TheRock release-candidate regression

The source patch at ZLUDA commit `9c8b43f`, tested on RX 9060 XT / `gfx1200` with TheRock 7.14.1 and PyTorch `2.0.1+cu118`, produced the following isolated capability result:

```text
33/35 PASS
2/35 UNSUPPORTED (safe refusal)
0 incorrect
0 timeouts
0 process hangs
0 process crashes
0 errors
```

This is a **historical full-run snapshot**, not the score of the current larger probe list. On 2026-09-18, `driver_launch_ex` and `driver_func_attributes` were added specifically for the newer llama.cpp boundary; those diagnostics are not folded into the 33/35 percentage until the rebuilt patched runtime is re-run end-to-end.

On the RX 9060 XT with unpatched v7-preview.10, the direct launch probe records no-attribute and PDL=0 launches as successful while PDL=1 and cooperative launch attributes return 801. The project intentionally keeps PDL=1 as a safe refusal until the AMD backend has equivalent programmatic-stream-serialization semantics. A separate function-attribute probe records the dynamic shared-memory resource boundary; the manually reproduced gfx1200 value was 65,536 bytes opt-in, with requests through 64 KiB accepted and larger requests rejected.

The capability runner uses a two-stage timeout policy: probes get a short initial budget, and only true timeouts are retried with a longer budget (180 s by default). This prevents slow first-touch JIT compilation from being misreported as breakage and, more importantly, prevents slow numerical corruption from being hidden in the timeout bucket. On the RX 9060 XT with the unpatched upstream v7-preview.10 asset, a forced 1 s first attempt followed by the long retry correctly reclassified `linalg_svd` as numerically incorrect (large case reconstruction max_abs about 3.60) instead of leaving it as a timeout.

The two safe refusals are `sdpa_flash` and `sdpa_mem_efficient`. The capability matrix also includes a direct `cuDeviceGetPCIBusId` regression probe because current llama.cpp uses that API during CUDA backend registration; v6-preview.69 reproduces error 801 on the reference machine, while pinned upstream v7-preview.10 and the patched v7 runtime return a valid PCI BDF string. Diagnosis showed that their low-architecture PTX fallback modules do not contain the real fused compute path: memory-efficient attention is a diagnostic stub, while Flash FMHA retains surrounding control/softmax plumbing but its score accumulators are never populated because the fused GEMM lives in NVIDIA cubins. The patch now returns `NO_BINARY_FOR_GPU` for those specific fallback modules instead of allowing a silent incorrect tensor. `sdpa_math` remains numerically correct.

The patched cuDNN compatibility DLLs now route the standard PyTorch BatchNorm training, backward and inference path (`CUDNN_BATCHNORM_OPS_BN`) through MIOpen, including the cuDNN `ForwardTrainingEx`/`BackwardEx` workspace APIs used by PyTorch 2.0.1. The isolated `batch_norm` capability passes against CPU references with no hang/crash/error; fused BatchNorm+activation/add variants remain deliberately unsupported until separately validated.

With the optional reversible cuSOLVER → hipSOLVER proxy staged, both `linalg_cholesky` and `linalg_qr` pass. The tested proxy preserves **940/940 exports** and routes **131 entry points** to hipSOLVER, including LU, legacy/generic-X Cholesky, legacy/generic-X QR, Jacobi SVD, symmetric/Hermitian eigensolvers, and generic-X `Xsyevd` paths. QR validation covers FP32, FP64, complex64 and complex128, single and batched tall/wide/square matrices, `torch.linalg.qr`, `torch.geqrf` + `orgqr`/`ungqr`, and `ormqr`/`unmqr`; a separate multi-stream regression also passes on two reused non-default streams. The ZLUDA BLAS bridge adds FP64/complex GEMM and strided-batched GEMM plus batched TRSM/GELS/LU/GEQRF bridges to AMD backends. A dedicated isolated linalg regression currently passes `solve`, Cholesky, QR, `inv`, `lstsq`, SVD, `pinv`, `eigh`, and `eigvalsh` (9/9) without hangs, crashes, or numerical failures. After the matrix run, the user's original `cusolver64_11.dll` was restored to SHA-256 `ECCA66A9100A514F586F7710F5B761265DFFAC615786C90C868A02A114EDE533`.

The same clean-patch runtime completed a real VelocityRL `512 agents × rollout 16` three-update smoke. Warmed updates reached **72,306 SPS** and **70,333 SPS**, for a **71,319.5 SPS steady-state median**. A same-GPU direct-rocBLAS comparison kept paired median CUDA→ZLUDA overhead below the repository's 20% budget at 1024², 2048² and 4096²; the clean-run deltas were **+1.79%**, **+0.45%**, and **+2.20%** respectively.
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

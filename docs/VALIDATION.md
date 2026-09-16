# Validation

This document records what was actually tested, rather than what is assumed to work.

## Public reproducible path

Validation date: **2026-09-13**

Hardware/software:

- Windows x64
- AMD Radeon RX 9060 XT (`gfx1200`)
- ZLUDA `v6-preview.69`, official Windows release asset
- AMD HIP SDK `6.4`
- LibTorch `2.3.0+cu118`
- `ZLUDA_CC=8.6`
- **no recovered/custom overlay DLLs**

The runtime used for this validation was created from the same public path exposed by `scripts/install.ps1`: official ZLUDA plus the installed AMD HIP SDK. The repository's ignored `local-artifacts/` directory was not used.

## ZLUDA runtime check

ZLUDA's `cuda_check.exe` completed and reported:

```text
nvcuda     OK
cuBLAS     OK -> rocBLAS
cuBLASLt   OK -> hipBLASLt
cuSPARSE   OK -> rocSPARSE
cuFFT      OK
cuDNN 8/9  unavailable on this stable Windows HIP SDK configuration
```

`test-runtime.ps1` treats the first five groups as core runtime checks. cuDNN is reported separately because the Windows HIP SDK does not provide the complete Linux ROCm AI-library stack.

## Real training integration test

The generated public runtime was then staged next to an existing CUDA-enabled LibTorch PPO trainer.

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

This verifies more than device enumeration: the workload performed CUDA-facing inference plus a real PPO learning/update phase using CUDA-enabled LibTorch on the AMD GPU stack.

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

The memory-efficient SDPA failure was reproduced on the RX 9060 XT reference as well, so it must not be presented as a `gfx1150`-only defect. It remains an explicit capability failure for workloads that select that backend.

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
24/26 PASS
2/26 UNSUPPORTED (safe refusal)
0 incorrect
0 timeouts
0 process hangs
0 process crashes
0 errors
```

The two safe refusals are `sdpa_flash` and `sdpa_mem_efficient`. Diagnosis showed that their low-architecture PTX fallback modules do not contain the real fused compute path: memory-efficient attention is a diagnostic stub, while Flash FMHA retains surrounding control/softmax plumbing but its score accumulators are never populated because the fused GEMM lives in NVIDIA cubins. The patch now returns `NO_BINARY_FOR_GPU` for those specific fallback modules instead of allowing a silent incorrect tensor. `sdpa_math` remains numerically correct.

With the optional reversible cuSOLVER → hipSOLVER proxy staged, the added `linalg_cholesky` capability also passes. The tested proxy preserves **940/940 exports** and routes **49 entry points** to hipSOLVER, including LU plus legacy and generic-X Cholesky paths. `torch.linalg.cholesky`, `torch.cholesky_solve` and `torch.cholesky_inverse` were checked against CPU references for FP32, FP64, complex64 and complex128 in both single and batched tensor cases. After the matrix run, the user's original `cusolver64_11.dll` was restored to SHA-256 `ECCA66A9100A514F586F7710F5B761265DFFAC615786C90C868A02A114EDE533`.

The same clean-patch runtime completed a real VelocityRL `512 agents × rollout 16` three-update smoke. Warmed updates reached **70,124 SPS** and **70,113 SPS**, for a **70,118.5 SPS steady-state median**. A same-GPU direct-rocBLAS comparison kept paired median CUDA→ZLUDA overhead below the repository's 20% budget at 1024², 2048² and 4096²; the 4096² delta was **+9.86%**.
## Historical performance

Older tuned runs of the same ZLUDA/LibTorch family retained approximately **70k-109k overall steps/s**. Those numbers are historical performance evidence and should not be confused with the short validation run above.

See `BENCHMARKS.md` for the retained performance notes.

## What this does not prove

This test does not claim universal CUDA compatibility. In particular:

- applications that require cuDNN can fail on the stable Windows HIP SDK path;
- unsupported PTX/CUDA APIs may fail;
- custom CUDA extensions are workload-specific;
- other AMD GPU architectures remain unverified until tested.

The compatibility issue template exists specifically to grow evidence one GPU/application at a time.

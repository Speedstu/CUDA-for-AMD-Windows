# Current smoke-test status

The project retains two reference profiles: the original RX 9060 XT (`gfx1200`)
stable-SDK path and the Radeon AI PRO R9700 (`gfx1201`) TheRock HIP SDK nightly
7.14.0a20260612 path. Results for one profile do not replace or invalidate the
other.

## Runtime probe

`cuda_check.exe` completed successfully for the core CUDA-facing libraries:

- `nvcuda`: PASS
- cuBLAS: PASS through rocBLAS
- cuBLASLt: PASS through hipBLASLt
- cuSPARSE: PASS through rocSPARSE
- cuFFT: PASS
- cuDNN 8/9: PASS through TheRock `MIOpen.dll` on the tested `7.14-nightly`

The earlier partial/hanging probe was not representative of the final public
path. The clean TheRock nightly configuration now exits normally and reports
both cuDNN compatibility groups as passing.

## Training probe

A CUDA-enabled LibTorch PPO workload was previously run with the runtime
produced by the public installation path, without the recovered custom
overlay. The original `gfx1200` profile completed one full training iteration /
65,536 timesteps; repeat the same probe on the `gfx1201` nightly profile before
claiming an equivalent training result there.

See `VALIDATION.md` for the exact recorded output and scope of the claim.

# Current smoke-test status

The public upstream-only runtime was revalidated on 2026-09-13 on an RX 9060 XT (`gfx1200`).

## Runtime probe

`cuda_check.exe` completed successfully for the core CUDA-facing libraries:

- `nvcuda`: PASS
- cuBLAS: PASS through rocBLAS
- cuBLASLt: PASS through hipBLASLt
- cuSPARSE: PASS through rocSPARSE
- cuFFT: PASS
- cuDNN: unavailable on the validated stable Windows HIP SDK configuration

The earlier partial/hanging probe was not representative of the final public path. The clean upstream configuration now exits normally.

## Training probe

A CUDA-enabled LibTorch PPO workload was run with the runtime produced by the public installation path, without the recovered custom overlay. It completed one full training iteration / 65,536 timesteps on the CUDA-facing device.

See `VALIDATION.md` for the exact recorded output and scope of the claim.

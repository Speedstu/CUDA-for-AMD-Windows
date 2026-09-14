# Current smoke-test status

The current smoke-test target is an RX 9070 (`gfx1201`) with HIP SDK 7.2. The RX 9060 XT (`gfx1200`) result below is historical and does not validate the new target.

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

A CUDA-enabled LibTorch PPO workload was previously run with the runtime produced by the public installation path, without the recovered custom overlay. Repeat one full training iteration / 65,536 timesteps on the `gfx1201` target before marking it validated.

See `VALIDATION.md` for the exact recorded output and scope of the claim.

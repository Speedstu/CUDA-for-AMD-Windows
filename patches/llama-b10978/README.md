# Pinned llama.cpp b10978 compatibility candidates

The only supported source for this patch is llama.cpp at commit `1e7bcf3da4b2741868d152fa47976fb2501c85e3` (b10978). The `llama-zluda-safe-build` workflow verifies this revision, applies the patch, and compiles it using CUDA Toolkit 12.4.

## F32 MMF fallback for ZLUDA (issue #6)

`zluda-safe-f32-mmf-fallback.patch` makes `ggml_cuda_should_use_mmf(GGML_TYPE_F32, ...)` return false **only** when compiled with `GGML_CUDA_ZLUDA_DISABLE_F32_MMF`.

The existing dispatch then keeps F32 MMVF for eligible small column counts and falls back to cuBLAS for ordinary larger F32 matrix multiplication. This avoids attempting to launch `mul_mat_f<float, 32, 4, 8, false>` under `ZLUDA_CC=8.6`. F16/BF16 MMF dispatch, the underlying CUDA API and the advertised compute capability are unchanged. The fallback can be slower than native F32 MMA; it is not a fix for arbitrary missing PTX symbols.

Why: a Radeon 890M tester reproduced `hipModuleGetFunction` error 500 with that **exact** missing kernel on six GGUFs (Qwen3.5, Qwen3.6, Gemma and Ornith). Setting `ZLUDA_CC=7.5` avoided the launch in 3/3 per model, but globally lowering the advertised compute capability can change unrelated dispatch decisions. This source-only fallback isolates the affected path without changing `ZLUDA_CC`. The original 7.5 model outputs have **not** been compared against native reference, and the new build likewise requires independent GPU inference/correctness tests before claiming those six models fixed.

The workflow records `disable_f32_mmf=true` in `build-manifest.json` so consumers can distinguish builds. It runs a native CPU-only source-extracted dispatch regression twice, with/without the macro, and then compiles the CUDA artifact. No NVIDIA runtime redistributables are packaged.

**Not addressed:** `top_k_cub` in MTP/speculative decoding and Nemotron `ggml_cuda_cpy` `operation not supported`. Do not test these with the previously crashing generic build; use the pinned safe artifact and cautious targeted tests only.
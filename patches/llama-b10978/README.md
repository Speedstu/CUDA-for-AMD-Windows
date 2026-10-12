# Pinned llama.cpp b10978 compatibility candidates

The only supported source for this patch is llama.cpp at commit `1e7bcf3da4b2741868d152fa47976fb2501c85e3` (b10978). The `llama-zluda-safe-build` workflow verifies this revision, applies the patch, and compiles it using CUDA Toolkit 12.4.

## F32 MMF fallback for ZLUDA (issue #6)

`zluda-safe-f32-mmf-fallback.patch` makes `ggml_cuda_should_use_mmf(GGML_TYPE_F32, ...)` return false **only** when compiled with `GGML_CUDA_ZLUDA_DISABLE_F32_MMF`.

The existing dispatch then keeps F32 MMVF for eligible small column counts and falls back to cuBLAS for ordinary larger F32 matrix multiplication. This avoids attempting to launch `mul_mat_f<float, 32, 4, 8, false>` under `ZLUDA_CC=8.6`. F16/BF16 MMF dispatch, the underlying CUDA API and the advertised compute capability are unchanged. The fallback can be slower than native F32 MMA; it is not a fix for arbitrary missing PTX symbols.

Why: a Radeon 890M tester reproduced `hipModuleGetFunction` error 500 with that **exact** missing kernel on six GGUFs (Qwen3.5, Qwen3.6, Gemma and Ornith). Setting `ZLUDA_CC=7.5` avoided the launch in 3/3 per model, but globally lowering the advertised compute capability can change unrelated dispatch decisions. This source-only fallback isolates the affected path without changing `ZLUDA_CC`. The pinned safe artifact from GitHub Actions run `38056434647` was verified by the gfx1150 reporter: all six formerly failing GGUF files completed three repeat runs each at `ZLUDA_CC=8.6`, without the missing-symbol error. Qwen3.5-4B MTP also completed a 10-question comparison against native ROCm (9/10 final answers identical; all 30 ZLUDA answers passed the basic grader, versus 27/30 native). This is strong evidence the missing-symbol failure is avoided, but the other five models have **not** been graded against native and the test is not proof of general numerical equivalence.

The workflow records `disable_f32_mmf=true` in `build-manifest.json` so consumers can distinguish builds. It runs a native CPU-only source-extracted dispatch regression twice, with/without the macro, and then compiles the CUDA artifact. No NVIDIA runtime redistributables are packaged.

**Not addressed:** `top_k_cub` in MTP/speculative decoding and Nemotron `ggml_cuda_cpy` `operation not supported`. Do not test these with the previously crashing generic build; use the pinned safe artifact and cautious targeted tests only.
## Experimental CUB TOP_K argsort fallback (issue #6)

`zluda-safe-topk-argsort-candidate.patch` disables only the CCCL 3.2 `DeviceTopK::MaxPairs` branch when **`GGML_CUDA_ZLUDA_DISABLE_CUB_TOPK`** is defined. `top-k.cu` then uses its existing CUDA CUB argsort/bitonic sort followed by a device-to-device copy. This is a potential workaround for `invalid device function` during MTP speculative decoding. It is not validated on the Radeon 890M and may be slower and/or numerically different in tie cases.

The separate workflow `llama-zluda-topk-candidate-build.yml` preserves the known-good `llama-zluda-safe-build.yml` untouched. It applies both the already-validated F32 MMF patch and this experimental TOP_K patch, records `disable_cub_topk=true`, and packages a **candidate-only** artifact. Its native no-GPU gate test checks both branches compile correctly, but real MTP compatibility and token correctness need a tester with a safe, previously working runtime. `ZLUDA_CC` remains 8.6; graph and FA safety flags remain disabled. Do not retry with the generic build that hard-locked gfx1150.

Recommended validation: first ensure Qwen3.5-4B MTP *plain inference* remains healthy in a fresh staged runtime. Only then try the existing `Qwen3.5-35B-A3B + draft-mtp` reproducer, with the launch-blocking driver, short `-c 4096`, clear tracing and stop-on-error. Compare accepted tokens and output with native ROCm where possible. If there is a hard lock or error, stop and report the first useful diagnostic rather than retrying. Nemotron `ggml_cuda_cpy` is unrelated and stays open.
## Experimental plain-CUDA radix TOP_K candidate (issue #6, follow-up)

The gfx1150 tester validated that the previous TOP_K argsort candidate
(7e77f05) avoids DeviceTopK but fails at the next CUB operation,
DeviceRadixSort::SortPairsDescending. This happens before either sort is
launched, during its temporary-storage query; stock and patched drivers
both reject the corresponding CUB calls. A different CUB algorithm is
not an adequate fallback.

The new opt-in zluda-safe-topk-radix-candidate.patch exposes the existing
llama.cpp HIP radix-selection implementation under the CUDA build macro
GGML_CUDA_ZLUDA_TOPK_RADIX. It does not globally disable CUB or alter
the known-good ordinary inference build. The experimental workflow
llama-zluda-topk-radix-candidate-build.yml applies F32 MMF, prior
TOP_K argsort and this final radix patch in that order; it packages an
artifact with manifest topk_implementation=cuda_radix_select.

The source path contains ordinary CUDA-style kernels with histogram,
prefix selection and atomic gather, but has not yet been evaluated on
gfx1150. Build success or the CPU-only source guard does NOT validate
its numerical results, ranking ties, performance, or speculative decoding.
Use a freshly staged separate candidate runtime, known-good launch-blocking
driver and small bounded tests. First repeat plain inference, then only
the already reported MTP reproducer if no prior tests fail. If the GPU
locks or resets, stop rather than repeating the same test. The CUB
attribute metadata regression documented in issue #3 must also be
corrected before trusting any CUB-backed PyTorch numeric tests.

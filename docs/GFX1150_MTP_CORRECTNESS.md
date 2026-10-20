# gfx1150 draft-MTP correctness confirmation — candidate-only

This is a **reproducible acceptance protocol**, not a claim that MTP works
correctly on every AMD GPU. It follows the hardware A/B report in
[issue #6](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/6#issuecomment-6204628522)
and the corrected driver controls in
[issue #3](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/3#issuecomment-6204628801).

## Established results (Radeon 890M / gfx1150 only)

- The pinned llama.cpp b10978 build in
  [Actions 38634153056](https://github.com/Speedstu/CUDA-for-AMD-Windows/actions/runs/38634153056)
  passed the compiled PTX gate: both histogram variants use
  `atom.shared.add.u32` and neither uses `red.shared.add`.
- Gemma-4-12B draft-MTP completed 3/3 runs on **each** of the f60a9d4
  and async-copy drivers, about **15.9–16.3 generated tokens/s**;
  ROCm native recorded about **17.1–17.7 tokens/s**.
- Qwen3.5-35B-A3B draft-MTP completed 3/3 runs per driver,
  about **16.8–17.7 tokens/s**; ROCm native about
  **18.5–18.9 tokens/s**.
- The original missing histogram symbol no longer occurs; ordinary
  non-MTP llama regression results remained unchanged in the report.
- Draft-token acceptance on a bounded 200-token run (`n_max=3`) was
  **57.9% ZLUDA vs 60.7% native** for Gemma,
  and **74.9% ZLUDA vs 72.2% native** for Qwen.
- Outputs were repeatable within each backend, but **the MTP outputs are
  not byte-identical across ZLUDA and ROCm**. Qwen ZLUDA MTP matched
  ZLUDA *plain* generation, whereas native MTP diverged from native plain
  after 633 characters. Gemma diverged within its thinking block.
  **Do not substitute text equality for correctness grading.**

The first trace-enabled, launch-blocking 200-token test exceeded a 300s
test harness timeout; it was logging 77k+ individual kernel launches,
not exhibiting a GPU fault. The bounded 24-token trace run passed,
and ordinary-speed 200-token runs completed. **Avoid repeating long
per-kernel traced runs to estimate performance or stability.**

## October 17: ten-question correctness set completed (reported 2026-10-16)

Cary5031 ran the fixed independently graded ten-question set from the
protocol below, on Gemma-4-12B and Qwen3.5-35B-A3B. Each model used four
modes (native plain, native MTP, ZLUDA plain, ZLUDA MTP), with three
repeats per question. The full question text, answer-key regexes and
per-mode summary are in
[issue #6](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/6#issuecomment-6226906104).
The individual run output files remain with the reporter; the project
maintainer has not independently rerun those 240 inferences.

| Graded set | Native plain | Native MTP | ZLUDA plain | ZLUDA MTP |
| --- | --- | --- | --- | --- |
| Gemma-4-12B | 30/30 | 30/30 | 30/30 | **30/30** |
| Qwen3.5-35B-A3B | 30/30 | 30/30 | 30/30 | **30/30** |

Across the reported 240 inference runs, every process exited normally,
was graded correct against its independent final-answer rubric, none
stopped incomplete or crashed, and all three repeats in every
question/model/mode cell returned identical bytes. Some wording
differs between modes while final-answer grading remains the same.

| Model | Native plain | Native MTP | ZLUDA plain | ZLUDA MTP |
| --- | ---: | ---: | ---: | ---: |
| Gemma-4-12B | 8.5 | 21.9 | 8.5 | 18.6 |
| Qwen3.5-35B-A3B | 12.3 | 20.2 | 12.4 | 17.5 |

Speeds are mean generated tokens/second on the gfx1150 test system.
MTP speedup over that backend's own plain mode: ZLUDA Gemma 2.18x
(native 2.59x), ZLUDA Qwen 1.41x (native 1.64x). Aggregated accepted
draft fractions were 87.1% ZLUDA versus 87.4% native for Gemma,
and 83.1% ZLUDA versus 85.1% native for Qwen. The reporter used
greedy generation, the same quantized model/draft weights across
backends, an upper limit of 400 generated tokens and logged actual
loaded CUDA driver SHA-256 for each ZLUDA process.

**Scope:** Strong correctness evidence for the 10 graded questions,
these two GGUF models and the specific gfx1150 runtime, NOT a universal
GPU/kernel correctness guarantee, numerical equivalence proof or
upstream ZLUDA PTX reduction parser fix. Keep artifacts opt-in.
## Completed ten-question evaluation protocol

To make the next comparison reproducible:

1. Reuse the **same fixed ten previously graded correctness questions**
   if the contributor still has them. Publish each question's text,
   reference answer, evaluation rubric, and unmodified prompt template.
   Do not cherry-pick easier questions after seeing results.
2. On both Gemma-4-12B and Qwen3.5-35B-A3B, run the same questions in
   the same order on **native ROCm MTP, ZLUDA MTP, native ROCm plain,
   and ZLUDA plain**. Use the **same GGUF**, draft weights,
   quantization, context length and prompt format; document the
   unavoidable backend differences. Keep sampling deterministic
   (greedy, temperature 0; pin other generation settings).
3. Save actual output text separately for every run and grade each
   **final answer** against the reference, not just native output
   text. If any response hasn't left the thinking block at the token
   limit, grade it **incomplete**, not correct. State the exact limit.
   Run at least three repeats per configuration where practical.
4. Separately record for each run: process exit status, elapsed time,
   generated tokens/s, total tokens, total draft tokens and accepted
   draft tokens, draft calls, model identities, `llama-cli` artifact
   SHA-256, launcher settings, and SHA-256 of the **nvcuda.dll loaded
   into the llama-cli process**.
5. Publish a 10-row table **per model** with reference answer, graded
   correct/incorrect/incomplete for each backend and MTP/plain mode,
   plus the first differing response and whether the difference
   changes the final answer. Include any stderr error/symbol lines.
6. Calculate, do not assume: number correct out of ten for each mode,
   incomplete count, run-to-run reproducibility, throughput versus
   that backend's own plain generation, and accepted-draft fraction
   (accepted tokens / proposed tokens). Report the original raw counts.

**Decision criteria.** A candidate is eligible for *further
maintainer review* if all bounded runs complete without hangs, crashes
or missing CUDA symbols, final-answer grading is comparable to that
backend's native reference, and a negative numerical comparison is
reported openly. This 10-question set **cannot** establish mathematical
equivalence, correctness of all kernels, or support for other GPUs.
Keep the artifacts opt-in and do not promote to global stable defaults.

## Separate open limitations

- Upstream ZLUDA's general `red.global.add`/`red.shared.add` PTX
  grammar is still missing. The llama-specific histogram rewrite is not
  a parser fix.
- The original cuDNN-enabled gfx1150 conv2d timeouts in issue #3 have not been
  demonstrated fixed by these llama or async-copy changes. The no-cuDNN fallback did pass the CPU reference on September 20. Use a
  bounded, isolated CPU-reference conv2d repro only after the
  contributor confirms their original case is safe to repeat.
- On gfx1100 the safe math-only SDPA wrapper is tested, but the
  underlying raw memory-efficient SDPA numeric bug in issue #9
  remains outstanding. Do not enable the unsafe fused path by default.


## October 20: Nemotron graded plain inference

Cary5031 independently graded ten fixed questions, three repeats each,
on Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf (22,833,947,424 bytes,
SHA256 627F5B04AEDC97F967332F331BD75B7A4ED2F33CA83E6EE74B44235CC1887890).
ZLUDA used the async-copy candidate driver (in-process nvcuda.dll
SHA256 94D3B56D... in all 30 child runs) and the histogram llama
candidate from Actions run 38634153056. The same model, prompts,
independent rubric, deterministic decoding and 400-token cap
were applied to native and ZLUDA plain inference:

| Backend | Correct final answers | Exits | Mean generation t/s |
| --- | ---: | ---: | ---: |
| Native ROCm plain | 30/30 | 30/30 | 13.8 |
| ZLUDA plain | 30/30 | 30/30 | 13.6 |

All repeats were byte-identical within each question/backend cell,
zero incomplete/incorrect answers, and zero CUDA faults/missing symbols.
Some wording differs between native and ZLUDA without changing the
graded final answer. ZLUDA/native throughput ~0.99 on Radeon 890M.
See https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/6#issuecomment-6281010418 .
The 60 underlying summaries and text outputs remain with the reporter,
not with this repository. Nemotron DRAFT-MTP was NOT tested.

## Reproduce the opt-in llama draft-MTP graded run (not a safe smoke)

Use a known-good and separately staged runtime, the SHA-verified
llama-b10978-zluda-topk-atom-candidate from Actions run 38634153056,
and the async-copy driver from run 38450929271. Verify the manifest
and actual loaded driver DLL SHA256, not just the file on disk.
The conservative run-llama-zluda-safe.ps1 defaults to one GPU layer,
one token and launch blocking; it is NOT a full MTP launcher.

The reporter's exact native arguments (substitute valid absolute
paths from the same fixed ten-question set) were:

    zluda.exe -- llama-cli.exe -m MODEL.gguf -md DRAFT.gguf -ngl 99 -ngld 99 -dev CUDA0 -devd CUDA0 -sm none --spec-type draft-mtp -nkvo --no-op-offload -fa off -st --jinja --reasoning off -sysf system.txt -f q01.txt --temp 0 --top-k 1 --seed 1234 -n 400 -c 4096 --verbose

Set ZLUDA_CC=8.6, ZLUDA_TRACE_MISSING_SYMBOLS=1 and
GGML_CUDA_DISABLE_GRAPHS=1 for this gfx1150 candidate.
Do NOT enable CUDA_LAUNCH_BLOCKING for throughput comparisons;
earlier per-launch tracing was orders of magnitude slower.
The full 99-layer command needs substantial memory and is NOT
authorized for a device that cannot accommodate it.

For plain mode use identical flags but remove -md, -ngld, -devd and
--spec-type. Use --no-jinja for the original Qwen-specific repro
when appropriate, but preserve the deterministic ten-question
grading protocol for matched comparisons.

## Candidate artifact lifespan / release safety

Actions run 38634153056 (histogram llama) was reported to expire
2026-10-28; 37965774616 (argv launcher) on 2026-11-08;
38300337884 (f60a9d4) on 2026-11-11;
38450929271 (async-copy driver) on 2026-11-12.
The older F32-MMF safe build artifact in run 38056434647 has
already expired. Longer retention on FUTURE runs cannot restore
expired artifacts; a reproducible rebuild will have a new SHA.
No candidate is globally promoted or published as stable.
A candidate-only prerelease needs manual review of original
license/redistribution terms, hashes, test evidence and risks.

## Remaining PTX and driver convergence

The llama histogram rewrite replaces a few PTX red.shared.add
instructions with supported atom.shared.add; generic red.global,
red.shared and red.generic parsing is still missing in ZLUDA.
Use the contributor's proposed address-space, register/immediate
and shared-address conversion tests with a [64,64,64,64] CPU
reference before any general parser patch is considered safe.
Build future driver candidates on top of the async-copy series
to preserve both the CUB/launch metadata fix and Nemotron copy
support, and rerun the full gating suite before promotion.

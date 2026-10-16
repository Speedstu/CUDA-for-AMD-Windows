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

## Requested final 10-question review

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
- The separate gfx1150 conv2d hangs in issue #3 have not been
  demonstrated fixed by these llama or async-copy changes. Use a
  bounded, isolated CPU-reference conv2d repro only after the
  contributor confirms their original case is safe to repeat.
- On gfx1100 the safe math-only SDPA wrapper is tested, but the
  underlying raw memory-efficient SDPA numeric bug in issue #9
  remains outstanding. Do not enable the unsafe fused path by default.

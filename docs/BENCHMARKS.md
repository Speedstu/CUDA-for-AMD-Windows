# Benchmarks

Benchmarks on the reference AMD Radeon RX 9060 XT (`gfx1200`) are split into two categories: historical tuned runs and a controlled custom-vs-upstream A/B.

## Controlled A/B: recovered custom overlay vs public upstream path

Run date: **2026-09-13**.

The same CUDA-enabled LibTorch PPO executable and training configuration were used for both runtimes:

- GPU: Radeon RX 9060 XT (`gfx1200`)
- model: 2,216,347 parameters
- seed: `123`
- arenas: `2048`
- timesteps per iteration: `65,536`
- PPO epochs: `1`
- max episode: `1s`
- FP32 path
- async overlap disabled
- Torch CPU threads: `10`

Each runtime was run as **2 trials × 5 iterations**. Trial order was upstream → custom → upstream → custom. The first iteration of every trial was treated as warmup, leaving **8 measured iterations per runtime**.

| Metric | Public upstream | Recovered custom | Custom delta |
| --- | ---: | ---: | ---: |
| Overall SPS, median | **13,278.46** | 12,875.80 | **-3.03%** |
| Overall SPS, mean | **13,172.49** | 12,649.83 | -3.97% |
| Collection SPS, median | **63,306.00** | 59,360.67 | -6.23% |
| Consumption SPS, median | **16,806.36** | 16,445.66 | -2.15% |
| Inference time, median | **0.5863 s** | 0.6293 s | +7.33% |
| PPO learn time, median | **3.2076 s** | 3.2958 s | +2.75% |

For this workload, the recovered custom runtime was **not faster**. The public upstream path is therefore the default and recommended configuration.

Raw data: [`../benchmarks/2026-09-13-rx9060xt-custom-vs-upstream.csv`](../benchmarks/2026-09-13-rx9060xt-custom-vs-upstream.csv)
Summary: [`../benchmarks/2026-09-13-rx9060xt-custom-vs-upstream.json`](../benchmarks/2026-09-13-rx9060xt-custom-vs-upstream.json)

## Historical tuned ZLUDA + LibTorch path

Older retained benchmark summaries report approximately **70k–109k overall steps/s** for tuned ZLUDA + LibTorch training runs. One warmed historical iteration reached roughly 147k overall steps/s.

Those numbers used a different, heavily tuned training configuration and should **not** be compared directly with the controlled A/B above. They remain useful evidence that the stack was used for long-running real PPO training rather than only CUDA enumeration.

## Later native HIP path

A later rewrite removed LibTorch/ZLUDA from PPO and reached substantially higher throughput. Those measurements belong to a different implementation and must not be presented as ZLUDA performance.

## Experimental v7/TheRock same-GPU translation overhead

Run date: **2026-09-16** on the RX 9060 XT (`gfx1200`). The native side calls HIP + rocBLAS directly. The compatibility side performs the same FP32 square matrix multiply through PyTorch CUDA → ZLUDA → rocBLAS. Both run on the same physical GPU.

Latest post-guard release-candidate validation using **4 paired repetitions per size**. The authoritative metric is synchronized monotonic **wall-clock time**, not HIP event timing: direct HIP testing on this Windows stack reproduced invalid negative `hipEventElapsedTime` values. Event measurements remain in the JSON for diagnostics only. Execution order alternates `direct → ZLUDA` and `ZLUDA → direct`; the reported delta is the median of the six paired percentage differences.

| SGEMM | Direct median wall time | ZLUDA median wall time | Paired median overhead | Paired throughput ratio |
| --- | ---: | ---: | ---: | ---: |
| 1024 × 1024 | 0.4471 ms | 0.4549 ms | +1.77% | 98.3% |
| 2048 × 2048 | 3.6579 ms | 3.4593 ms | -5.37% | 105.7%* |
| 4096 × 4096 | 18.1603 ms | 20.0220 ms | +9.86% | 91.0% |

The final clean-patch run keeps the translation path within the repository's **20% paired-median overhead budget** at every tested size. The 4096² case retains about **91.0%** of direct rocBLAS throughput. The measured -5.37% delta at 2048² is marked with an asterisk because it is treated as clock/thermal/order variance, not as evidence that the translation layer is inherently faster than direct rocBLAS. Individual samples still move with clocks, thermals and kernel selection, which is why the runner alternates execution order, keeps every pair, and applies its regression threshold to the paired median rather than a single sample.

These results measure the compatibility path against an AMD-native backend on the same physical GPU; they are not an AMD-vs-NVIDIA comparison.

### Cold start vs steady-state training

ZLUDA's persistent compute cache matters strongly for PyTorch training. During diagnosis, the first uncached `clamp.backward()` took about **50 s** and an uncached `minimum.backward()` about **96 s**. Once compiled, the same kernel families dropped to roughly millisecond/sub-millisecond latency.

A real VelocityRL test at `512 agents × rollout 16` on the final clean-patch runtime produced **519 SPS** for a non-representative cold first update, followed by **70,124 SPS** and **70,113 SPS** on updates 2 and 3. The report records a **70,118.5 SPS median steady-state**. The cold update is intentionally separated because its value moves dramatically with first-use compilation/cache state.

The helper below builds the local cache without shipping machine-specific compiled artifacts:

```powershell
.\scripts\warmup-pytorch.ps1 -PythonExe C:\path\to\venv\Scripts\python.exe
```

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

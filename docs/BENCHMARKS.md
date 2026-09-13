# Benchmarks

Recovered measurements were produced on AMD Radeon RX 9060 XT (`gfx1200`), Ryzen 7 5700X, Windows, ROCm 6.4 and LibTorch 2.3.0+cu118.

## ZLUDA + LibTorch path

The retained benchmark summary reports approximately **70k-109k overall steps/s** for the ZLUDA + LibTorch 2.3.0 cu118 training path.

One warmed benchmark iteration recorded approximately:

- collection: 219k steps/s
- consumption: 448k steps/s
- overall: 147k steps/s

Longer runs settled lower; 70k-109k is the conservative retained summary. The path executed real PPO training rather than only CUDA enumeration.

## Later native HIP path

A later rewrite removed LibTorch/ZLUDA from PPO and reached much higher throughput. Those measurements belong to a different implementation and must not be presented as ZLUDA performance.

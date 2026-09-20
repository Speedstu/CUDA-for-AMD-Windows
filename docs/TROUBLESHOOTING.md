# Troubleshooting

## `nvcuda.dll` not found

Stage the runtime beside the application and put ZLUDA at the front of `PATH`, or use `scripts/run-zluda.ps1`.

## HIP runtime mismatch

Do not mix arbitrary ROCm/HIP versions. The historical profile used a specific HIP 7.13 overlay with a ROCm 6.4 installation. For new machines, prefer a coherent current Windows HIP SDK; use the historical mix only to reproduce it.

## rocBLAS / hipBLASLt cannot find kernels

Set:

```text
ROCBLAS_TENSILE_LIBPATH=<HIP_ROOT>\bin\rocblas\library
HIPBLASLT_TENSILE_LIBPATH=<HIP_ROOT>\bin\hipblaslt\library
```

## CMake insists on CUDA toolkit discovery

Use the manual LibTorch import-library approach in `examples/manual-libtorch-cuda.cmake` and supply CUDA 11.8 headers explicitly.

## Wrong CUDA compute capability

`ZLUDA_CC=8.6` is a CUDA-facing compatibility identity. Do not replace it with `gfx1200`; CUDA SM versions and AMD `gfxXXXX` targets are different architecture namespaces.

Also do not assume that advertising a higher CUDA compute capability is automatically safer or faster. CUDA applications use the reported SM generation to select kernels, instructions, launch attributes and resource profiles. A path tuned for an NVIDIA SM can therefore be a bad fit for the real AMD backend even when device enumeration succeeds.

If you experiment with a different `ZLUDA_CC`, treat it as a workload-specific compatibility test: record the value, compare numerical output against a known-good path, and do not promote the result from "launches" to "supported" without end-to-end validation.

For recent llama.cpp behavior, see [`LLAMA_CPP.md`](LLAMA_CPP.md).

## `cuLaunchKernelEx` returns 801

Run the focused driver probe:

```powershell
.\scripts\test-capabilities.ps1 `
  -PythonExe C:\path\to\venv\Scripts\python.exe `
  -Tests driver_launch_ex
```

The probe separates no-attribute launches, `COOPERATIVE`, and `PROGRAMMATIC_STREAM_SERIALIZATION` (PDL). A zero-valued attribute should not be treated the same as a semantic feature request.

The project deliberately keeps non-zero PDL as a safe refusal until the AMD backend has equivalent ordering/serialization semantics. Silently ignoring `PDL=1` can turn an obvious compatibility failure into wrong kernel execution.

## `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` returns invalid argument

Run:

```powershell
.\scripts\test-capabilities.ps1 `
  -PythonExe C:\path\to\venv\Scripts\python.exe `
  -Tests driver_func_attributes
```

This reports the device's normal, per-SM and opt-in shared-memory limits and checks the `cuFuncSetAttribute` boundary with a no-op PTX kernel.

If the application asks for more dynamic shared memory than the driver advertises, do **not** clamp the request silently. That normally means the application selected a kernel/resource profile that is incompatible with the real device. Prefer a supported kernel path, disable that optimization, or fix the application's dispatch logic.

## cuFFT errors

Some Windows ZLUDA/PyTorch FFT paths have historically returned unsupported errors. Treat FFT-heavy applications separately from GEMM-heavy ML/RL workloads.

## First iteration is very slow

JIT/kernel compilation and caches can dominate the first iteration. Benchmark warmed iterations under the same runtime/cache configuration.

## PyTorch fused SDPA returns `NO_BINARY_FOR_GPU`

On the experimental ZLUDA v7/TheRock path with PyTorch 2.0.1+cu118, Flash and memory-efficient SDPA depend on fused compute that is present only in NVIDIA cubins; their low-architecture PTX fallbacks are not complete implementations. The project patch fails closed rather than allowing a silent bad tensor.

For applications that can use PyTorch's math SDPA backend, the launcher now enables this policy automatically when the runtime config uses the unpatched upstream `latest` channel. You can also force it explicitly:

```powershell
.\scripts\run-zluda.ps1 -RuntimeRoot .\.runtime-v2 -Program C:\path\to\python.exe -ProgramArgs @('app.py') -PyTorchSafeSDPA
```

This does not change `ZLUDA_CC` or claim that the fused backends are implemented. Use `-AllowUnsafeFusedSDPA` only for deliberate compatibility experiments against the raw upstream fused paths.

`-AllowUnsafeFusedSDPA` also clears an inherited `CUDAAMD_PYTORCH_SAFE_SDPA` value and removes only this project's `pytorch-safe-site` entry from `PYTHONPATH`. Other `PYTHONPATH` entries are preserved. This matters when `run-zluda.ps1` is called repeatedly from the same PowerShell environment: an explicit unsafe run must not silently inherit a previous safe-mode launch.
## gfx1150 PyTorch `conv2d` hangs

Community testing on Radeon 890M / `gfx1150` showed that the legacy cuDNN/ZLUDA convolution path can hang indefinitely even though GEMM is healthy.

For Python launched through `run-zluda.ps1`, the project now disables cuDNN automatically on `gfx1150` unless a staged cuDNN→MIOpen bridge with a concrete MIOpen backend is detected. PyTorch then uses its non-cuDNN convolution fallback instead of entering the known hanging path.

The capability probe applies the same policy automatically for `-Tests conv2d` on `gfx1150` and records whether `torch.backends.cudnn.enabled` was true or false in the result JSON.

To deliberately reproduce the raw legacy cuDNN path for A/B testing only:

```powershell
.\scripts\run-zluda.ps1 `
  -Program C:\path\to\python.exe `
  -ProgramArgs @('app.py') `
  -AllowUnsafeCudnnConv
```

Do not use that override on a machine where a GPU/driver reset would be unacceptable. The default fallback is intentionally conservative.

## gfx1150 llama.cpp can GPU-page-fault / bugcheck Windows

A Radeon 890M / `gfx1150` reporter reproduced the remaining b10978 failure with a full Windows kernel dump. The dump identified `0x119 VIDEO_SCHEDULER_INTERNAL_ERROR` in `dxgmms2!VidSchiProcessIsrHwQueuePageFaulted`, consistent with a GPU hardware-queue page fault followed by a failed reset/recovery sequence.

This means later errors reported at `ggml_backend_cuda_buffer_clear`, `buffer_set_tensor`, or a generic synchronize call can be asynchronous messengers rather than the kernel that actually faulted.

Do **not** chase those synchronization line numbers as if they were the root kernel.

The launch-blocking candidate adds a safer first diagnostic:

```powershell
.\scripts\test-capabilities.ps1 `
  -RuntimeRoot C:\path\to\candidate-runtime `
  -PythonExe C:\path\to\venv\Scripts\python.exe `
  -Tests driver_launch_blocking
```

That probe uses a tiny known-safe PTX kernel and verifies that `CUDA_LAUNCH_BLOCKING=1` is actually honored by the ZLUDA driver path and that per-kernel trace markers are emitted.

For application tracing, `CUDA_LAUNCH_BLOCKING=1` synchronizes after each kernel launch and automatically enables `[zluda-launch]` tracing. `ZLUDA_LAUNCH_TRACE=1` can enable the names without blocking. Set `ZLUDA_LAUNCH_TRACE_FILE=C:\path\to\zluda-kernels.log` to append the same numbered records directly to disk; each record is flushed and synced so the last unmatched `#N begin` has the best chance of surviving a system crash.

Important: launch blocking improves attribution; it does **not** make a page-faulting kernel harmless. If the workload is already known to page-fault the GPU, avoid repeated or long inference runs on a machine where a TDR/bugcheck is unacceptable.

After a crash or controlled run, analyze the persistent launch trace with:

```powershell
python .\scripts\summarize-zluda-trace.py C:\path\to\zluda-kernels.log --json .\launch-summary.json
```

For numbered `[zluda-launch]` records, the report includes completed launches, explicit launch/sync errors, and the last `#N begin` that has no matching terminal record. That unmatched launch is the first kernel to investigate after a crash; it is evidence of the last in-flight launch, not by itself proof that the kernel source is defective.

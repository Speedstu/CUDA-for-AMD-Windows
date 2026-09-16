# ZLUDA v7-preview.10 Windows compatibility patch

This is an experimental source patch for upstream ZLUDA `v7-preview.10` / commit `9c8b43f`.

It is **not** the repository's default stable runtime yet. The stable installer remains unchanged while this patch set is validated on more workloads and GPUs.

## What this patch adds

On the RX 9060 XT / `gfx1200` development machine, using a recent TheRock Windows HIP stack and CUDA-facing PyTorch `2.0.1+cu118`, the patch enables additional CUDA-facing functionality that is missing or stubbed in upstream ZLUDA v7-preview.10:

- **cuFFT → hipFFT**: common plan, workspace, stream, execution and Xt paths. Validated paths include FP32 R2C/C2R, complex64 C2C, FP64 D2Z/Z2D and 2D FFT round-trips.
- **cuSPARSE → rocSPARSE**: adds the paths required by the tested PyTorch sparse matrix multiplication flow, including `cusparseSetStream`, `cusparseXcoo2csr` and `cusparseCreateCsr`.
- **Windows NVML through ZLUDA**: initialization, device count, handle-by-index, PCI-bus lookup, device name and basic memory information query the CUDA-facing `nvcuda.dll` driver instead of opening an independent HIP runtime context.
- **CUDA Graph legacy ABI**: implements the CUDA 10.x `cuStreamBeginCapture` and `cuStreamGetCaptureInfo` entry points that PyTorch requests through `cuGetProcAddress`, routing them to the existing HIP graph implementation.
- **Windows cuFFT loadability**: embeds the same Common Controls v6 manifest used by the other Windows compatibility DLLs so `cufft64_*.dll` loads cleanly instead of failing with Win32 error 127 on `TaskDialogIndirect`.
- **CUDA event elapsed-time invariant**: Windows HIP/TheRock can occasionally return a negative interval for completed in-order events. The CUDA-facing wrapper preserves positive measurements and clamps only impossible negative values to `0 ms`.
- **Fused-SDPA fail-closed guard**: PyTorch 2.0.1 Flash and memory-efficient attention ship low-architecture PTX fallbacks whose real fused compute path is absent and lives in NVIDIA cubins. The patch detects these specific fallbacks and returns NO_BINARY_FOR_GPU instead of executing code that can silently produce invalid tensors.
- **Extended cuBLAS linalg bridges**: FP64/complex GEMM + strided-batched GEMM, batched TRSM/GELS, and batched LU/QR helpers are routed through rocBLAS/rocSOLVER/hipSOLVER so PyTorch `inv`, `lstsq`, and QR batch paths do not fall into unimplemented CUDA entry points.
- **cuBLAS dtype and QR coverage**: FP64/complex GEMM plus FP64/complex strided-batched GEMM route to rocBLAS. `cublas[S/D/C/Z]geqrfBatched` is bridged to hipSOLVER generic-X using the active cuBLAS stream; the bridge explicitly synchronizes before consuming PyTorch's device-side pointer arrays and before workspace teardown so reused non-default streams remain correct.
- **ROCm DLL coherence on Windows**: the QR bridge dynamically anchors `amdhip64_7.dll` and `hipsolver.dll` to the selected HIP/TheRock tree rather than adding a static HIP import that could resolve to an incompatible System32 runtime.
- **PyTorch BatchNorm through MIOpen**: standard cuDNN BatchNorm inference, training and backward are bridged to MIOpen, including the `ForwardTrainingEx`/`BackwardEx` workspace/reserve APIs imported by PyTorch 2.0.1 for `CUDNN_BATCHNORM_OPS_BN`. Fused BatchNorm+activation/add modes remain fail-closed until separately validated.

The NVML design is deliberate. An earlier direct-HIP NVML prototype passed isolated NVML probes but caused `cusparseCreate`/`rocsparse_create_handle` to fail later in the same process. Routing NVML queries through ZLUDA keeps CUDA-facing libraries on one context model.

The patch also guards late `hipfftDestroy` / `rocsparse_destroy_handle` calls when Windows is already in DLL shutdown. Without this guard, FFT and sparse operations returned numerically correct results but the Python process later terminated with `0xC0000409` during cached-handle teardown. Normal runtime destruction still calls the AMD backend; the guard applies only once Windows reports DLL shutdown in progress.

A clean-checkout rebuild passes a combined strict **NVML + `torch.sparse.mm` + FFT + CUDA Graph + streams/events** regression with zero numerical errors, zero hangs and zero post-result process crashes on the tested `gfx1200` stack. A real PyTorch `torch.cuda.CUDAGraph` capture → replay → synchronization probe also passes with correct output. The event-timing workaround was separately stress-tested across isolated processes after reproducing the negative-timing behavior in direct HIP. With the optional cuSOLVER proxy staged for the linear-algebra probes, the broader isolated capability matrix currently records **32/34 clean passes (94.1%) plus 2 safe refusals** with zero incorrect results, timeouts, hangs, post-result crashes or errors. Flash SDPA and memory-efficient SDPA are the two deliberate refusals on PyTorch 2.0.1+cu118; the math SDPA backend remains numerically correct.

Advanced cuFFT/cuSPARSE/NVML/Graph entry points not covered by the patch still fall back to normal ZLUDA unsupported behavior.

## Apply

From a clean ZLUDA checkout at commit `9c8b43f`:

```powershell
git apply C:\path\to\CUDA-for-AMD-Windows\patches\zluda-v7-preview10\windows-amd-compat.patch
cargo +1.96.0 build -p zluda -p zluda_sparse -p zluda_fft -p zluda_ml --release
```

The patch is intentionally source-only. No third-party AMD or NVIDIA binaries are included.

## Validation

Use the repository capability runner against an isolated runtime:

```powershell
.\scripts\test-capabilities.ps1 `
  -RuntimeRoot C:\path\to\runtime `
  -PythonExe C:\path\to\cuda-pytorch-venv\Scripts\python.exe
```

For a focused regression of the newly patched surfaces:

```powershell
.\scripts\test-capabilities.ps1 `
  -RuntimeRoot C:\path\to\runtime `
  -PythonExe C:\path\to\cuda-pytorch-venv\Scripts\python.exe `
  -Tests nvml,sparse_mm,fft,cuda_graph,streams_events `
  -Strict
```

Results are workload- and stack-dependent. A pass on `gfx1200` does not imply that every AMD architecture or every CUDA API is supported.

## First-use compilation

PyTorch can trigger expensive one-time ZLUDA compilation for dynamically instantiated backward kernels. On the tested environment, PPO clipping primitives such as `clamp`/`minimum` took tens of seconds on their first ever execution but became millisecond/sub-millisecond operations once cached. This is compilation latency, not steady-state kernel latency.

Use the repository helper after staging the experimental runtime:

```powershell
.\scripts\warmup-pytorch.ps1 `
  -RuntimeRoot C:\path\to\runtime `
  -PythonExe C:\path\to\cuda-pytorch-venv\Scripts\python.exe
```

It invokes ZLUDA's source-build `zluda_precompile.exe` when available and then executes a small dynamic training warmup. Add `-Extended` to also prewarm especially cold sort/topk, determinant, BatchNorm/GroupNorm, dropout, grid-sample and transpose-convolution paths. The cache remains local to the user and is never committed.

# ZLUDA v7-preview.10 Windows compatibility patch

This is an experimental source patch for upstream ZLUDA `v7-preview.10` / commit `9c8b43f`.

It is **not** the repository's default stable runtime yet. The stable installer remains unchanged while this patch set is validated on more workloads and GPUs.

## What this patch adds

On the RX 9060 XT / `gfx1200` development machine, using a recent TheRock Windows HIP stack and CUDA-facing PyTorch `2.0.1+cu118`, the patch enables additional CUDA-facing functionality that is missing or stubbed in upstream ZLUDA v7-preview.10:

- **cuFFT → hipFFT**: common plan, workspace, stream, execution and Xt paths. Validated paths include FP32 R2C/C2R, complex64 C2C, FP64 D2Z/Z2D and 2D FFT round-trips.
- **cuSPARSE → rocSPARSE**: adds the paths required by the tested PyTorch sparse matrix multiplication flow, including `cusparseSetStream`, `cusparseXcoo2csr` and `cusparseCreateCsr`.
- **Windows NVML through ZLUDA**: initialization, device count, handle-by-index, PCI-bus lookup, device name and basic memory information query the CUDA-facing `nvcuda.dll` driver instead of opening an independent HIP runtime context.

The NVML design is deliberate. An earlier direct-HIP NVML prototype passed isolated NVML probes but caused `cusparseCreate`/`rocsparse_create_handle` to fail later in the same process. Routing NVML queries through ZLUDA keeps CUDA-facing libraries on one context model.

The patch also guards late `hipfftDestroy` / `rocsparse_destroy_handle` calls when Windows is already in DLL shutdown. Without this guard, FFT and sparse operations returned numerically correct results but the Python process later terminated with `0xC0000409` during cached-handle teardown. Normal runtime destruction still calls the AMD backend; the guard applies only once Windows reports DLL shutdown in progress.

A clean-checkout rebuild now passes a combined strict **NVML + `torch.sparse.mm` + FFT** regression with zero numerical errors, zero hangs and zero post-result process crashes on the tested `gfx1200` stack.

Advanced cuFFT/cuSPARSE/NVML entry points not covered by the patch still fall back to normal ZLUDA unsupported behavior. CUDA Graphs are not modified by this patch set.

## Apply

From a clean ZLUDA checkout at commit `9c8b43f`:

```powershell
git apply C:\path\to\CUDA-for-AMD-Windows\patches\zluda-v7-preview10\windows-amd-compat.patch
cargo build -p zluda_sparse -p zluda_fft -p zluda_ml --release
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
  -Tests nvml,sparse_mm,fft `
  -Strict
```

Results are workload- and stack-dependent. A pass on `gfx1200` does not imply that every AMD architecture or every CUDA API is supported.

# llama.cpp through ZLUDA on AMD/Windows

This page tracks the current llama.cpp compatibility boundary separately from the project's PyTorch/LibTorch validation.

> [!IMPORTANT]
> A CUDA device appearing in `--list-devices` does **not** prove that inference kernels are safe or correct. Recent llama.cpp builds exercise CUDA driver/runtime surfaces that older builds did not.

## Current status

The most detailed community report is [issue #3](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/3) on Radeon 890M / `gfx1150`.

| llama.cpp build | gfx1150 result reported in issue #3 |
| --- | --- |
| b4500 | Runs end-to-end through ZLUDA; decode was reported near the native ROCm result |
| b9009 | Device registration works, but compute reaches a kernel-level failure |
| b10978 | Registration works with the pinned ZLUDA v7 channel; newer launch/kernel paths still fail |

The project does **not** currently claim modern llama.cpp support on every AMD GPU.

## Registration: fixed by the v7 channel

Current llama.cpp builds call `cudaDeviceGetPCIBusId`. ZLUDA v6-preview.69 returns `CUDA_ERROR_NOT_SUPPORTED` for the corresponding driver path, while the pinned v7-preview.10 channel implements it.

Use:

```powershell
.\scripts\install.ps1 -SkipLibTorch -ZludaChannel latest
```

Then confirm the CUDA backend is actually visible. The repository includes a registration-only smoke test that does not load a model or launch inference kernels:

```powershell
.\scripts\test-llama-registration.ps1 `
  -LlamaRoot C:\path\to\llama-b10978-bin-win-cuda-12.4-x64
```

It runs `llama-cli.exe --list-devices` through the configured ZLUDA runtime, requires an AMD `[ZLUDA]` CUDA device, rejects `(none)` / CPU-only fallback, and writes `.runtime\llama-registration-test.json` with executable/runtime hashes.

You can still inspect the application directly with:

```powershell
llama-cli.exe --list-devices
```

Do not trust a throughput number until the output shows an AMD device through ZLUDA and the model layers are actually assigned to that device. A silent CPU fallback can otherwise look like a successful run.

## `cuLaunchKernelEx`: explicit regression coverage

Recent llama.cpp uses `cudaLaunchKernelEx` and launch attributes.

The repository capability matrix now includes `driver_launch_ex`, which uses a no-op PTX kernel to isolate driver launch semantics from model math.

The expected policy is:

- no attributes: must launch successfully;
- `PROGRAMMATIC_STREAM_SERIALIZATION=0`: must launch successfully;
- `COOPERATIVE=0`: must launch successfully;
- `PROGRAMMATIC_STREAM_SERIALIZATION=1`: may return 801 / NOT_SUPPORTED until the AMD backend has equivalent semantics;
- non-zero cooperative launch is reported separately because support is device/backend dependent.

The project deliberately does **not** turn `PDL=1` into a silent no-op just to make an application advance further. A clean refusal is safer than changing launch ordering semantics and risking incorrect computation.

Run the focused driver probes with:

```powershell
.\scripts\test-capabilities.ps1 `
  -PythonExe C:\path\to\cuda-pytorch-venv\Scripts\python.exe `
  -Tests driver_pci_bus_id,driver_launch_ex,driver_func_attributes
```

## Dynamic shared-memory boundary

Recent Flash Attention paths in llama.cpp opt in to larger per-kernel dynamic shared memory with `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`.

The capability matrix now includes `driver_func_attributes`. It records:

- `MAX_SHARED_MEMORY_PER_BLOCK`;
- `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`;
- `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`;
- whether `cuFuncSetAttribute` accepts a value within the advertised limit;
- whether it accepts the exact advertised limit;
- whether a request beyond that limit is correctly refused.

On the RX 9060 XT / `gfx1200` reference machine, the manually reproduced boundary was 65,536 bytes: requests through 64 KiB succeeded and larger requests were rejected.

That matters because a CUDA application selecting an NVIDIA-tuned kernel can ask for more shared memory than the AMD backend exposes even when device registration itself succeeded.

## `ZLUDA_CC` is a compatibility identity, not the AMD ISA

ZLUDA exposes a CUDA compute capability so CUDA applications can choose code paths. That value must not be interpreted as proof that the AMD GPU implements every NVIDIA instruction or resource profile associated with that SM generation.

For example, advertising an Ampere-like compute capability can make an application select Ampere-specific MMA / Flash Attention kernels. The actual AMD target remains something like `gfx1150`, `gfx1200`, or `gfx1201`.

Do not fix a kernel failure by blindly changing `ZLUDA_CC` and then treating a launch as validated. Any alternate value needs numerical validation and workload-specific testing.

## PDL workaround is diagnostic, not a complete fix

llama.cpp exposes:

```text
GGML_CUDA_PDL=0
```

This can bypass the PDL launch path and move execution back to a classic kernel launch. On the gfx1150 report, that moved the failure deeper into kernel execution rather than making recent llama.cpp fully work.

Use it to isolate the failing layer, not as evidence that the underlying kernel is compatible.

## Keep the application's real CUDA runtime

For the tested Windows llama.cpp packages, keep the stock NVIDIA `cudart` that ships with the CUDA build. ZLUDA replaces the CUDA **driver** surface through `nvcuda.dll`; renaming `nvcudart_hybrid64.dll` over the application's stock `cudart` can change behavior and was reported to break otherwise-working older llama.cpp builds.

## Stability warning

Issue #3 includes hard GPU/system lockups on gfx1150 with newer llama.cpp kernels. A process timeout is not guaranteed to recover a GPU after an invalid kernel has already damaged the driver state.

Until a specific modern llama.cpp build passes end-to-end on a given GPU:

- use a machine where a forced restart is acceptable;
- test `--list-devices` before model inference;
- prefer small diagnostic workloads first;
- keep PDL / Flash Attention tests separate;
- record the exact llama.cpp build, ZLUDA channel, HIP version, GPU `gfx` target and `ZLUDA_CC`;
- do not label the workload supported from a device-detection result alone.

## What counts as fixed

For a modern llama.cpp build, this project will treat the path as validated only when all of the following are true on the tested GPU:

1. CUDA device registration succeeds without CPU fallback.
2. Driver launch probes are clean.
3. The selected kernels launch without GPU reset/hang.
4. Prompt processing and token decode both complete.
5. Output is compared against a known-good native/reference path where practical.
6. Repeated runs exit cleanly with no delayed driver crash.

Until then, issue #3 remains intentionally open.

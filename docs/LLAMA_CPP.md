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
| b10978 | gfx1150: patched launch/PTX path is confirmed, SDPA silent corruption is fixed, but real kernel execution still fails later; gfx1200: patched v7 completes an end-to-end GPU smoke with all layers offloaded |

For the b10978 investigation, the pinned upstream source commit is `1e7bcf3da4b2741868d152fa47976fb2501c85e3`. The official Windows CUDA 12.4 package used for reproduction is:

```text
llama-b10978-bin-win-cuda-12.4-x64.zip
SHA-256 62D7478A88888574BFDE7A0B8FFDA4AABEC152E1C4DA69CC051CF1B7BA9DF043
```

The project does **not** currently claim modern llama.cpp support on every AMD GPU.

### 2026-09-18 reference validation on gfx1200

On the RX 9060 XT / `gfx1200` reference machine, a rebuilt v7-preview.10 runtime with the launch-attribute and PTX-metadata candidates now passes the driver preflight and a real b10978 GPU smoke:

- `driver_pci_bus_id`, `driver_launch_ex`, `driver_func_attributes`, and `driver_function_metadata`: **PASS**;
- the b10978 `.version 8.4 / .target sm_90` regression reports `PTX_VERSION=84`, so the false `ptxVersion >= 90` PDL gate stays **false**;
- `llama-cli --list-devices` detects `CUDA0: AMD Radeon RX 9060 XT [ZLUDA]` with no CPU-only fallback;
- a tiny GGUF smoke with `-ngl all -dev CUDA0 -fa off` and no `GGML_CUDA_PDL` override assigns **6/6 layers** to `CUDA0`, places the KV and compute buffers on CUDA0, completes CUDA Graph warmup, produces prompt/decode output, and exits cleanly;
- the smoke was repeated and did not produce a delayed GPU reset on the reference machine.

The same run still prints `no device code compatible` diagnostics from low-architecture fallback PTX embedded in the CUDA build. Those messages are not being hidden or counted as universal kernel support. The observed smoke proves that the previous registration/launch/metadata blockers are cleared on gfx1200; it does **not** prove every llama.cpp kernel shape or `gfx1150` is fixed.

### 2026-09-19 gfx1150 re-test

The issue #3 reporter re-tested the candidate `nvcuda.dll` from run `35398039575` on Radeon 890M / `gfx1150` with ROCm 7.2 and b10978.

The driver-level fixes reproduce there:

- `driver_pci_bus_id`: PASS;
- `driver_launch_ex`: PASS, including `COOPERATIVE=0`, while `PDL=1` still fails closed with 801;
- `driver_function_metadata`: PASS, including PTX 7.0 → 70 and PTX 8.4 → 84;
- the false b10978 `ptxVersion >= 90` PDL gate is therefore closed on gfx1150 too;
- memory-efficient SDPA changed from a finite but incorrect tensor to an explicit safe refusal, confirming the original silent-corruption bug is fixed on RDNA 3.5 as well.

Recent llama.cpp still does **not** pass end-to-end on that GPU. The candidate now gets past the old launch-attribute failure, spends time loading/compiling kernels, then fails later during real kernel execution around `ggml_backend_cuda_buffer_clear` / stream synchronization (and, for another model, `ggml_cuda_kernel_can_use_pdl`). The reporter also tested `ZLUDA_CC=8.0`, `8.6`, and `9.0`; the failure remained, so changing the advertised CUDA compute capability is not being treated as a fix.

That moves the remaining boundary from registration/launch metadata to application kernel loading/execution on gfx1150.

The CUDA 12.4 llama package also needs its matching stock NVIDIA `cudart64_12.dll` available on the process path. If only the main llama CUDA archive is extracted without the companion cudart package, b10978 can report `Available devices: (none)` even when the ZLUDA driver probes pass.

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

# Optional: add the focused CUDA driver metadata/launch preflight:
.\scripts\test-llama-registration.ps1 `
  -LlamaRoot C:\path\to\llama-b10978-bin-win-cuda-12.4-x64 `
  -PythonExe C:\path\to\cuda-pytorch-venv\Scripts\python.exe
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

The capability matrix also includes `driver_pdl_semantics`. It uses two PTX kernels in the same stream:

- a producer performs a deliberately non-trivial sequence of volatile global writes, stores `41`, and executes `griddepcontrol.launch_dependents`;
- a dependent kernel executes `griddepcontrol.wait`, reads the value, increments it and stores `42`.

If the backend returns `801` for `PDL=1`, the probe reports a normal unsupported/safe-refusal. If a future ZLUDA candidate accepts the launch attribute, success is only considered valid when the producer→consumer ordering remains numerically correct across repeated runs. A conservative backend that simply preserves ordinary same-stream serialization is allowed; a backend that accepts the attribute but breaks dependency ordering is not.

Run the focused driver probes with:

```powershell
.\scripts\test-capabilities.ps1 `
  -PythonExe C:\path\to\cuda-pytorch-venv\Scripts\python.exe `
  -Tests driver_pci_bus_id,driver_launch_ex,driver_pdl_semantics,driver_func_attributes,driver_function_metadata,driver_ptx_selection,driver_buffer_clear
```


### Multi-PTX target selection

Inspection of the tested b10978 `ggml-cuda.dll` found 143 CUDA fatbins. Every observed fatbin contains PTX 8.4 variants for `sm_50`, `sm_61`, `sm_70`, `sm_75`, `sm_80`, and `sm_90`, plus NVIDIA cubins for `sm_86` and `sm_89`.

The v7 loader previously reversed the PTX list and selected the first fully parsed entry, which makes these fatbins select `sm_90` regardless of `ZLUDA_CC`. A synthetic regression fatbin confirms the bug: with ZLUDA advertising compute capability 8.6, the old loader executes the `sm_90` kernel body instead of the highest compatible `sm_80` PTX.

`ptx-target-selection-candidate.patch` changes selection to the highest PTX target at or below the advertised CUDA compute capability. This is a candidate until the gfx1200 reference workload and the gfx1150 issue reporter both re-test it.

## Focused buffer-clear / kernel-loading probe

To isolate the new gfx1150 failure without requiring a model or llama.cpp fatbins, the capability matrix includes `driver_buffer_clear`.

It runs two tiny PTX clear kernels:

- PTX 7.0 targeting `sm_80`;
- PTX 8.4 targeting `sm_90`.

For each variant the probe checks:

```text
cuModuleLoadData
  -> cuModuleGetFunction
  -> non-default stream
  -> cuLaunchKernel
  -> cuStreamSynchronize
  -> D2H readback
  -> every element really became zero
```

Run it directly with:

```powershell
.\scripts\test-capabilities.ps1 `
  -PythonExe C:\path\to\cuda-pytorch-venv\Scripts\python.exe `
  -Tests driver_buffer_clear
```

On the RX 9060 XT / `gfx1200` reference runtime, both PTX 7.0/sm_80 and PTX 8.4/sm_90 variants pass all phases including non-default-stream synchronization and numerical readback.

This probe intentionally does **not** reproduce llama.cpp's embedded fatbins or every kernel it ships. If it also passes on gfx1150 while b10978 still fails in `ggml_backend_cuda_buffer_clear`, that is strong evidence that the remaining problem is application-specific device-code/kernel compatibility rather than generic PTX clear-kernel launch or stream synchronization.

## PTX metadata: target SM is not PTX ISA version

There is a second driver-level problem behind the unexpected PDL path.

Upstream ZLUDA currently answers `CU_FUNC_ATTRIBUTE_PTX_VERSION` with the parsed module's `sm_version` (the `.target sm_XX` value). CUDA defines `PTX_VERSION` differently: it is the PTX ISA major/minor encoded as `major * 10 + minor`.

That distinction matters directly to b10978. Its launcher only opts into PDL when:

```text
cudaFuncAttributes.ptxVersion >= 90
```

For a known module:

```text
.version 7.0
.target sm_80
```

the correct CUDA metadata is:

```text
PTX_VERSION = 70
```

not `80`.

CUDA 12.4 corresponds to PTX ISA 8.4, so treating an `sm_90` target as `PTX_VERSION=90` can make a CUDA 12.4 application appear PDL-capable when its PTX ISA version does not meet that gate.

The repository therefore includes:

- `ptx-version-candidate.patch`, which preserves the real parsed PTX version separately from the target SM without changing the on-disk ZLUDA cache format;
- `driver_function_metadata`, which loads a known `.version 7.0 / .target sm_80` module and requires `CU_FUNC_ATTRIBUTE_PTX_VERSION=70`;
- diagnostic reporting for `BINARY_VERSION`, because frameworks such as PyTorch can also use that value for architecture gating. The binary-version behavior is **not changed yet** by this candidate.

The rebuilt candidate passes the direct metadata probe on both the gfx1200 reference machine and the community-tested gfx1150 system. b10978 completes the end-to-end GPU smoke on gfx1200, while gfx1150 gets past the metadata/launch blockers and then fails later in real kernel execution. The metadata fix is therefore cross-checked on RDNA 3.5 without treating the full llama.cpp workload as supported there.

### Expected effect on b10978

The PTX metadata candidate does **not** implement PDL. For the official CUDA 12.4 b10978 package, the expected effect is that a PTX 8.4 kernel targeted at `sm_90` reports `ptxVersion=84`. The llama.cpp host check `ptxVersion >= 90` then stays false, so it should use the classic launch path without requiring `GGML_CUDA_PDL=0`.

That is separate from `launch-attributes-candidate.patch`, which fixes the benign `COOPERATIVE=0` driver case but deliberately keeps non-zero PDL unsupported.

On gfx1200, the PTX metadata fix plus the launch-attribute candidate advance b10978 through a real `-fa off` GPU smoke with all layers offloaded. On gfx1150, those same driver-level fixes are confirmed, but b10978 still fails later during real kernel loading/execution. That later failure is the remaining architecture-specific boundary; it is not folded into the metadata claim.

NVIDIA reference:
https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html

CUDA 12.4 / PTX 8.4 release note:
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/

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

### b10978 source-level Flash Attention boundary

For the exact llama.cpp b10978 release commit (`1e7bcf3da4b2741868d152fa47976fb2501c85e3`), the Ampere host configuration for `DKQ=DV=128, ncols=8` uses:

- `nbatch_fa=128`;
- `nbatch_K2=nbatch_V2=64`;
- a two-stage pipeline when `ncols2 >= 2`;
- `Q_in_reg=true`.

The b10978 source also contains real template instances for `(ncols1,ncols2)=(1,8),(2,4),(4,2)`.

Using b10978's own shared-memory formulas, those variants request:

| `ncols1,ncols2` | requested dynamic shared memory |
| --- | ---: |
| `1,8` | 65,808 bytes |
| `2,4` | 66,080 bytes |
| `4,2` | 66,624 bytes |

The manually measured gfx1200/ZLUDA opt-in boundary is 65,536 bytes. Therefore these concrete b10978 kernels exceed the advertised backend limit by only 272-1,088 bytes, which is enough for `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` to return `invalid argument`.

This is a **static source diagnosis**, not yet an end-to-end fix. Silently clamping the requested size is not valid because the kernel launch/layout was selected for the larger allocation. A correct workaround must select a compatible kernel/resource configuration and then pass numerical validation.

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

## Tracing the remaining classic-kernel failure

The original gfx1150 report reached an unspecified launch failure when `GGML_CUDA_PDL=0` and Flash Attention was disabled. The current gfx1200 candidate no longer needs that PDL override for the tested b10978 smoke, but the original architecture-specific failure remains useful context until gfx1150 is re-tested.

ZLUDA's Windows trace mode can record the driver calls, resolved kernel function names, PTX modules and compiler diagnostics:

```powershell
$env:GGML_CUDA_PDL = '0'
C:\path\to\zluda.exe --zluda-trace -- `
  C:\path\to\llama-cli.exe <normal arguments> -fa off
```

Trace output is written under:

```text
%TEMP%\zluda
```

The useful evidence is the final `log.txt` region around the first failed launch plus any `module_*.ptx` / `module_*.log` generated for that run. A `cuModuleGetFunction` record can be correlated with the function handle later passed to `cuLaunchKernel`, which lets us identify the actual failing kernel rather than attributing everything to the high-level `ggml_cuda_compute_forward` call.

The repository includes a trace summarizer for that correlation:

```powershell
python .\scripts\summarize-zluda-trace.py "$env:TEMP\zluda" `
  --json .\.runtime\llama-zluda-trace-summary.json
```

It selects the newest `log.txt` under the supplied trace directory, maps function handles back to kernel names, lists launch calls/non-success statuses, and surfaces any `module_*.log` compiler diagnostics.

Because gfx1150 has previously hard-locked during newer llama.cpp kernel experiments, trace mode does **not** make the test safe. Only collect this on a machine where a forced restart is acceptable, and prefer the smallest possible workload/output length.

## Capturing the first failing CUDA kernel

For failures that only surface as a generic `ggml_cuda_compute_forward` / `cudaGetLastError`, use ZLUDA's trace mode instead of guessing from the llama.cpp call site.

The repository wraps upstream `zluda.exe --zluda-trace -- ...` with:

```powershell
.\scripts\capture-zluda-trace.ps1 `
  -Program C:\path\to\llama-cli.exe `
  -ProgramArgs @(
    '-m', 'C:\path\to\small-test-model.gguf',
    '-fa', 'off'
  ) `
  -TimeoutSeconds 120 `
  -AcknowledgeGpuResetRisk
```

The helper:

- stages the configured runtime unless `-NoStage` is used;
- records stdout/stderr;
- detects only newly-created `%TEMP%\zluda\<app>[_N]` directories;
- copies `log.txt`, `module_*.ptx`, `module_*.elf` and related trace artifacts;
- records SHA-256 provenance for the program, launcher, `nvcuda.dll` and runtime config;
- writes `trace-report.json`;
- when Python is available, runs `scripts/summarize-zluda-trace.py` automatically and stores a per-trace summary JSON;
- produces a ZIP under `.runtime\traces`.

The summarizer correlates `cuModuleGetFunction` handles with `cuLaunchKernel` / `cuLaunchKernelEx` calls, reports non-success CUDA calls, and searches the captured `module_*.ptx` files for the resolved kernel name. That normally gives us the exact PTX module behind the final failing launch instead of only the generic llama.cpp call site.

It can also be run manually:

```powershell
python .\scripts\summarize-zluda-trace.py C:\path\to\trace-run --json trace-summary.json
```

> [!WARNING]
> The timeout only kills the process tree. It cannot restore a GPU/driver that has already entered an unrecoverable state. The helper therefore requires `-AcknowledgeGpuResetRisk` explicitly. Do not use it on a machine where a forced reboot would be unacceptable.

This helper remains useful for any architecture-specific modern llama.cpp kernel failure; a captured trace by itself is not evidence that the workload is supported.

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
4. Prompt processing and token decode both complete.5. Output is compared against a known-good native/reference path where practical.
6. Repeated runs exit cleanly with no delayed driver crash.

The gfx1200 reference satisfies device registration, driver launch probes, GPU kernel execution, prompt processing, token decode, and repeated clean exit for the diagnostic b10978 smoke. On gfx1150, the registration/launch/PTX-metadata/SDPA fixes are confirmed, while modern llama.cpp still has a separate application device-code/kernel-execution boundary. That application-specific work is tracked separately from the original issue #3 conv2d/SDPA report.

## gfx1150 conservative b10978 build

For the remaining gfx1150 page-fault investigation, the repository provides a separate GitHub Actions build of pinned llama.cpp b10978 (`1e7bcf3da4b2741868d152fa47976fb2501c85e3`) intended to reduce the amount of custom CUDA device code exercised before a kernel is proven safe.

The build uses:

- `GGML_CUDA_FORCE_CUBLAS=ON`;
- CUDA Graphs disabled;
- CUDA VMM disabled;
- peer-copy disabled;
- Flash Attention CUDA kernels not compiled;
- NCCL disabled;
- PTX-only CUDA targets `sm_75` and `sm_80`.

This does not make an unknown page-faulting kernel intrinsically safe. It reduces optional CUDA surfaces and routes supported matrix multiplication through the already-tested cuBLAS -> AMD backend path.

Use `scripts/run-llama-zluda-safe.ps1` for the first application smoke. Its defaults are intentionally small: one GPU layer and one generated token, no KV offload, no generic op offload, FA off, split mode `none`, CUDA Graphs disabled, `CUDA_LAUNCH_BLOCKING=1`, and a persistent numbered ZLUDA kernel trace.

The launcher refuses more than four GPU layers unless `-AllowMoreThanFourGpuLayers` is supplied. Escalate only after the smaller smoke exits cleanly. A matching stock CUDA 12.4 `cudart64_12.dll` is still required beside llama.cpp; the project artifact does not redistribute NVIDIA runtime DLLs.

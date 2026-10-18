# Compatibility

Support in this project is reported per **hardware + software stack + capability/workload**. A GPU being detected does not mean every CUDA application is supported.

## GPU status

| GPU | Native target | Status | Evidence |
| --- | --- | --- | --- |
| Radeon RX 9060 XT | `gfx1200` | **validated reference** | Main development machine; stable public stack and experimental v7 validation |
| Radeon RX 9070 XT | `gfx1201` | **validated external** | Separately tested Windows AMD/ZLUDA training setup; see [`RX9070XT_VALIDATION.md`](RX9070XT_VALIDATION.md) |
| Radeon RX 7900 XT | `gfx1100` | **community partial** | Stable v6.69 / HIP 6.4: GEMM and math SDPA verified; `run-zluda.ps1` math-only guard validated by reporter on real gfx1100 (`max_abs=2.384e-7`). Raw memory-efficient SDPA remains incorrect ([#9](https://github.com/Speedstu/CUDA-for-AMD-Windows/issues/9)); other capabilities unverified |
| Radeon 890M | `gfx1150` | **community partial** | HIP 7.2/GEMM, 6/6 focused driver preflight, PTX target selection, and guarded no-cuDNN conv2d are confirmed on hardware. Math-only PyTorch SDPA default verified by reporter 3/3 on stable v6 and latest v7; unsafe memory-efficient SDPA fails closed when guarded. Generic b10978 llama.cpp CUDA inference is **not supported yet**: a GPU page fault and Windows `0x119` bugcheck were reproduced; use the launch-blocking probe and conservative profile only |
| Other recognized AMD GPUs | architecture-dependent | **unverified candidate** | Scanner/runtime detection only until functional evidence is submitted |

`gfx1150` / RDNA 3.5 currently requires HIP SDK **7.2 or newer** in the project profile. The historical `gfx1200` reference uses HIP SDK 6.4.

The machine-readable architecture metadata lives in [`../manifests/windows-gpu-profiles.json`](../manifests/windows-gpu-profiles.json).

## Evidence levels

The repository uses the narrowest label supported by the evidence:

| Level | Meaning |
| --- | --- |
| detected | GPU/runtime can be enumerated |
| loadable | CUDA-facing DLL/API loads |
| safe refusal | unsupported behavior fails explicitly without wrong output or process damage |
| functional pass | operation completes and matches a CPU/native reference within tolerance |
| integration validated | real application/workload completes end-to-end and exits cleanly |

A higher level must not be inferred from a lower one.

## Stable reference path

The public stable reference is pinned around:

- Radeon RX 9060 XT / `gfx1200`
- ZLUDA `v6-preview.69`
- AMD HIP SDK `6.4`
- LibTorch `2.3.0 + cu118`

Smoke coverage includes the CUDA-facing driver plus cuBLAS, cuBLASLt, cuSPARSE and cuFFT loading. Numerical validation and the real PPO integration test are documented in [`VALIDATION.md`](VALIDATION.md).

### Runtime coverage on the validated stable reference

| CUDA-facing component | Current reference result |
| --- | --- |
| CUDA driver / `nvcuda` | ✅ validated path |
| cuBLAS | ✅ via rocBLAS |
| cuBLASLt | ✅ via hipBLASLt |
| cuSPARSE | ✅ via rocSPARSE |
| cuFFT | ✅ validated path |
| cuDNN | ⚠️ not available as a complete equivalent in the validated stable Windows HIP SDK path |

Dense/GEMM-heavy workloads can work without cuDNN. Convolution-heavy applications may need newer Windows ROCm components or additional compatibility work.

### CUDA-facing DLL map

This is a quick map of the main CUDA-facing Windows DLLs encountered by the project. A row marked experimental or partial is **not** a claim of complete library/API compatibility.

| CUDA-facing DLL / library | Typical role | AMD-side path in this project | Current status |
| --- | --- | --- | --- |
| `nvcuda.dll` | CUDA Driver API | ZLUDA → HIP | ✅ validated core path |
| `cudart64_12.dll` | CUDA Runtime API used by applications | Application-side NVIDIA CUDA Runtime; used alongside the compatibility stack when an application requires it | ⚠️ required by some apps; not an AMD reimplementation |
| `cublas64_12.dll` | BLAS / GEMM | ZLUDA → rocBLAS | ✅ validated operations |
| `cublasLt64_12.dll` | advanced GEMM / Lt API | ZLUDA → hipBLASLt / AMD backend path | 🟡 capability-specific |
| `cufft64_11.dll` | FFT | ZLUDA → rocFFT | ✅ validated operations |
| `cusparse64_12.dll` | sparse linear algebra | ZLUDA → rocSPARSE | ✅ validated operations |
| `cusolver64_11.dll` | dense solver / decompositions | project proxy → hipSOLVER | 🧪 experimental |
| `cudnn64_8.dll` | deep-learning primitives / convolution training | project cuDNN v8 bridge → MIOpen | 🧪 experimental; validated forward + dX + dW subset on gfx1200 |
| `nvml.dll` / NVML-facing path | GPU discovery / telemetry compatibility | ZLUDA/project NVML compatibility layer | ✅ validated reference path |

The exact DLL filenames depend on the CUDA major/minor version an application was built against. The table describes the **library role and tested compatibility route**, not a promise that every export in a given DLL is implemented.

For the cuDNN bridge specifically, the generated top-level `cudnn64_8.dll` preserves the original cuDNN v8 export surface and overrides only the subset listed in [`../native/cudnn_bridge/bridge-manifest.json`](../native/cudnn_bridge/bridge-manifest.json). See [`CUDNN_BRIDGE.md`](CUDNN_BRIDGE.md) for the validated scope and limitations.

## Experimental ZLUDA v7 path

The source patch set under [`../patches/zluda-v7-preview10/`](../patches/zluda-v7-preview10/) targets pinned upstream ZLUDA `v7-preview.10`.

It exists to develop newer CUDA-facing behavior without silently changing the stable installer.

Validated/reference-machine work includes:

- newer driver API coverage such as PCI bus ID and function metadata;
- launch-attribute probes with unsupported semantics kept fail-closed;
- PTX metadata/version handling used by newer applications;
- CUDA Graph stream-capture compatibility;
- cuFFT and cuSPARSE paths;
- NVML compatibility;
- experimental cuSOLVER → hipSOLVER bridging;
- experimental cuDNN v8 forward/backward convolution → MIOpen bridging on the gfx1200 reference system;
- SDPA guards for unsafe fallback kernels;
- llama.cpp registration and end-to-end GPU smoke on gfx1200;
- a focused PTX buffer-clear/non-default-stream regression probe for separating generic launch/sync behavior from application-specific kernel failures.

A source patch compiling successfully is not proof that it is functionally correct on every AMD GPU.

### Experimental cuDNN v8 convolution bridge

A separate source-built top-level `cudnn64_8.dll` compatibility proxy is validated on the RX 9060 XT / `gfx1200` reference system with PyTorch `2.3.0+cu118` and a TheRock 7.14.0 MIOpen backend.

Validated forward cases include FP32 and FP16 basic/padding/stride/dilation/grouped convolutions. The same matrix now validates autograd backward-data (input gradient) and backward-filter (weight gradient). `aten::cudnn_convolution`, public `torch.nn.functional.conv2d`, and the public autograd path matched CPU references for the fixed test tensors, with zero max absolute error in the 2026-09-19 gfx1200 reference run.

This is **not full cuDNN support**. The current bridge is limited to a narrow legacy cuDNN v8 2D forward + backward-data + backward-filter subset, default stream execution, cross-correlation mode, non-deterministic-algorithm reporting, and `alpha=1`, `beta=0`. cuDNN backward-bias, non-default streams, deterministic guarantees, fused APIs, and the cuDNN frontend are not claimed. See [`CUDNN_BRIDGE.md`](CUDNN_BRIDGE.md).

## Safe failure matters

The project does **not** count any of the following as support:

- device detection followed by CPU fallback;
- a kernel returning a numerically incorrect tensor;
- a timeout or GPU hang;
- a process that returns a result and then crashes during teardown;
- ignoring a CUDA launch attribute when that changes execution semantics;
- pretending an unavailable NVIDIA-cubin-only implementation is AMD-native.

For fused SDPA, a clean `UNSUPPORTED` result is preferred over silently accepting a fallback that does not contain the real compute path. The patched v7 candidate has now been independently re-tested on gfx1150: the previously finite-but-wrong memory-efficient SDPA result becomes an explicit safe refusal there as well.

For `gfx1150` PyTorch convolution, the legacy cuDNN/ZLUDA route is handled conservatively too: the launcher and `conv2d` capability probe avoid the known hanging path by disabling cuDNN unless a staged cuDNN→MIOpen bridge with an explicit MIOpen backend is detected. This is a safety fallback, not a claim that native cuDNN behavior is implemented. `-AllowUnsafeCudnnConv` exists only for deliberate A/B testing.

## Known limitations

- ZLUDA is not a complete CUDA implementation.
- Windows exposes only a subset of the full ROCm ecosystem.
- Some CUDA software depends on NVIDIA-specific cubins, PTX behavior, driver semantics or libraries.
- NCCL, TensorRT, unsupported custom CUDA extensions and architecture-specific kernels may fail.
- JIT compilation can make first-use latency look like a hang unless timeouts account for compilation.
- `ZLUDA_CC` is an emulated CUDA-facing compute capability; it is not the AMD `gfxXXXX` architecture.
- Hardware status can change as drivers, HIP SDKs, ZLUDA and applications evolve.

## Reporting another GPU

Run:

```powershell
.\scripts\gpu-scan.ps1 -OutputPath .\gpu-report.json
.\scripts\doctor.ps1
.\scripts\test-runtime.ps1
.\scripts\test-functional.ps1 -PythonExe C:\path\to\cuda-facing-venv\Scripts\python.exe
```

For broader bring-up:

```powershell
.\scripts\test-capabilities.ps1 -PythonExe C:\path\to\cuda-facing-venv\Scripts\python.exe
```

Then use the repository's **GPU compatibility report** issue template and include the exact GPU, `gfx` target, driver, HIP version, ZLUDA version/channel, application version, command, first useful error, and proof of GPU execution.
### Experimental llama.cpp F32 MMF compatibility (issue #6)

The Radeon 890M / gfx1150 reporter traced `MUL_MAT` missing-symbol crashes across six GGUF files to a single missing CUDA kernel, `mul_mat_f<float, 32, 4, 8, false>` (`hipModuleGetFunction` error 500). Switching the virtual compute capability from 8.6 to 7.5 made all six files launch, but that workaround affects *all* CUDA dispatch decisions and its results have not been compared for correctness against native ROCm. Do **not** silently change the project-wide `ZLUDA_CC` default.

A candidate patched pinned llama.cpp safe build now offers a narrower solution: `GGML_CUDA_ZLUDA_DISABLE_F32_MMF` disables only F32 MMF dispatch and uses the existing MMVF/cuBLAS fallbacks. It does not alter F16/BF16 MMF, graph/FA safety flags or other applications. See [`patches/llama-b10978/README.md`](../patches/llama-b10978/README.md). The gfx1150 contributor **confirmed all six files now launch** with the patched safe artifact (3/3 runs each), retaining `ZLUDA_CC=8.6`. On the 4B MTP file, 9/10 final answers matched native ROCm; the other five have not undergone equivalent correctness grading. This resolves the reported missing-symbol route at the application level, not the underlying F32 kernel implementation. The experimental `TOP_K` argsort candidate is a **separate** build; `top_k_cub` and Nemotron copy faults remain open until independently verified.
### Critical: older patched driver CUB corruption (issue #3)

Candidate drivers 0c2bee2, 07638a7, c810f91 and a2ac6e6 returned
PTX ISA version rather than target SM. This silently corrupted PyTorch sort,
cumsum and other CUB-backed operations on gfx1150 for large inputs. These
artifacts are not validated for numerical correctness. The stock v7-preview.10
driver passed the reporter's corresponding tests.

The corrected candidate preserves target SM and adds cub_sort_scan CPU-vs-GPU
tests at 8192 and 65536 elements. Build success alone is insufficient:
perform real GPU validation before replacing any existing runtime.
### Experimental TOP_K radix candidate (issue #6)

The previous experimental TOP_K argsort artifact failed at a subsequent CUB
DeviceRadixSort call on gfx1150. A separate opt-in radix candidate now
routes TOP_K through the existing plain-CUDA radix selection code. It is
not yet hardware-validated and must not replace the known-good F32 MMF
safe build. Nemotron copy errors are not addressed.
### gfx1150 MTP histogram and Nemotron copy candidates (October 13)

The last Radeon 890M trace identified independent, reproducible gaps:
TOP_K radix histogram PTX emits unsupported red.shared.add.u32, while the
Nemotron strided-copy path calls the missing cuMemcpy2DAsync_v2 driver entry.
The new opt-in llama build forces a supported atom.shared.add.u32 for the
histogram, and the independent driver candidate forwards async pitched
copies to HIP. **Neither candidate has passed complete gfx1150 inference
and numerical correctness testing yet.** The known-good F32 workaround,
older stock driver and validated f60a9d4 CUB driver remain separate.
### gfx1150 evidence update: async 2D copy and Nemotron (October 14)

Cary5031 confirmed the compile-only nvcuda-memcpy2d-async-candidate from
Actions run 38450929271, with loaded candidate hash 94D3B56D..., resolves
the Nemotron cpy.cu:478 CUDA 801 failure in the tested configuration.

- Nemotron-3-Nano-30B-A3B plain inference: six successful bounded runs
  (3 x 80 tokens, 3 x 200 tokens), plus a clean traced run with 230
  successful cuMemcpy2DAsync_v2 calls.
- At 80 tokens, output is byte-identical to native ROCm. At 200 tokens,
  wording differs after a common prefix; do not claim exact long-output
  numerical equivalence.
- Independent async 2D tests passed for exact Nemotron pitches,
  unequal pitches, offsets, 64-row copies, host copies and non-default
  and default streams. Stock and f60a9d4 returned CUDA 801.
- CUB sort/cumsum 24/24; wider PyTorch seven-op suite 35/35; three
  llama regression models 9/9, matched f60a9d4 on the same runtime.
  These are gfx1150-specific results reported by the contributor.

Our original driver_memcpy2d_async probe falsely failed all drivers with
CUDA_ERROR_INVALID_CONTEXT (201), because cuInit does not establish
the current context. The revised test initializes and verifies a context
before allocation. driver_pdl_semantics reports unsupported for compute
capability below 9.0 instead of treating sm_90 PTX rejection (209) as
a driver regression. This does not imply PDL support.

The llama histogram candidate run 38450933377 compiled but did NOT upload
an artifact: its PTX verifier was missing cuobjdump. Its replacement
workflow downloads a pinned NVIDIA cuobjdump 12.4.127 ZIP and checks the
official SHA-256 before inspecting compiled histogram PTX. TOP_K/MTP
remains unvalidated on gfx1150 until the new artifact and MTP test pass.
No known-good runtime is automatically replaced.
### October 16: reproducible gfx1150 MTP results and evidence gate

Cary5031 verified that the histogram candidate from run 38634153056
successfully runs draft-MTP on gfx1150 for Gemma-4-12B and
Qwen3.5-35B-A3B (three bounded runs per driver and model).
PTX verification reports two histogram variants with atom.shared.add
and no red.shared.add. The output is deterministic on each backend but
is not always byte-identical to native ROCm. Draft acceptance is near
native; see [full results and the 10-question correctness
protocol](GFX1150_MTP_CORRECTNESS.md). This is **hardware-tested candidate**
evidence, not a universal stable CUDA compatibility claim.

Candidate driver 94D3B56D... also passed the corrected async-copy and
PDL probe classification on the reported gfx1150 setup. Release evidence
now additionally requires a SHA-256 of the nvcuda.dll **actually loaded
inside each probe process**, preventing launcher redirection from
masking a different DLL than the staged one. The changes do not enable
unsupported upstream red instructions or resolve the separate conv2d
hangs and raw fused SDPA numerical errors.
### October 17: ten-question MTP grading and false-negative report fix

Cary5031 ran the fixed 10-question independent answer-key grading on
gfx1150 with Gemma-4-12B and Qwen3.5-35B-A3B. Each mode (native plain,
native MTP, ZLUDA plain and ZLUDA MTP) graded 30/30 for each model;
240/240 total successful exits and correct final answers, zero
incomplete. ZLUDA MTP gains over own plain mode averaged 2.18x Gemma
and 1.41x Qwen on this system. Wording sometimes differs between
backends; this is bounded correctness evidence, not complete numeric
equivalence. The raw 240 run logs remain with the contributor; see
[GFX1150_MTP_CORRECTNESS.md](GFX1150_MTP_CORRECTNESS.md).

The same contributor found a bug in test-functional.ps1: case-insensitive
PowerShell variable names caused the typed selection parameter $Tests
to alias the local $tests result array. Thus both selected GPU probes
printed PASS but the report serialized strings and falsely failed the
release gate. The results collection is now $probeEntries, covered by
a no-GPU execution test for both selected and default test sets. No
CUDA kernel, driver DLL or installed runtime was changed.

The original September gfx1150 conv2d failure was a bounded 25/180
second child process timeout, not a hard GPU lock. It happened in the
cuDNN/bridge path; the PyTorch no-cuDNN fallback passed the CPU
reference on 2026-09-20 (max_abs ~1.14e-05). A single carefully
bounded trace would be required to isolate the failing cuDNN path;
keep the existing fallback available.

### October 18: cuDNN phase 1 — activation ABI candidate

A versioned experimental cuDNN v8-to-MIOpen bridge now includes
sigmoid/ReLU/tanh descriptors and forward/backward translations (53
manifest overrides after tensor transform/set/scale additions).
The offline synthetic MIOpen ABI and numerical regression suite
passes, including a backend with no optional activation symbols.
**This is not yet a real AMD GPU functional pass** for the new
activation API; see [CUDNN_BRIDGE.md](CUDNN_BRIDGE.md) and the
[phase 1 checklist](CUDNN_PHASE1_CHECKLIST.md). Stable installers
and already validated driver binaries remain unchanged.


### October 18: cuDNN softmax ABI candidate

The experimental cuDNN v8 bridge now exposes accurate channel-wise
softmax forward/backward using optional MIOpen _V2 symbols, with **55
manifest overrides**. The CPU-backed test validates numerical
softmax values and analytical gradients; unsupported algorithms,
axes and missing symbols fail closed. Real AMD GPU FP32/FP16
qualification is pending and required before promotion to validated
status. The safe public runtime remains unchanged.


### October 18: 2D max-pooling ABI candidate

The cuDNN v8 proxy now has 60 overridden entry points with a
small forward-only 2D MAX pooling translation via optional MIOpen
symbols. Its unpadded, nonoverlapping 2x2 window/stride CPU-backed
ABI test passes against a 4x4 input and checks the exact numerical
max values and alpha/beta scaling. Ineligible modes and missing
backend symbols are deliberately unsupported. This remains an
**experimental, not yet real-GPU-validated** path; pooling backward,
cuDNN v9, padding semantics and non-default streams are not claimed.


### October 18: actual gfx1200 cuDNN/MIOpen FP32+FP16 hardware qualification

A separately staged, source-built 60-override cuDNN v8 compatibility
bridge ran **16/16 numeric GPU checks** on a physical RX 9060 XT,
using actual HIP and MIOpen DLLs from the official TheRock Windows
gfx120X-all 7.14.0 distribution. Eight FP32 and eight FP16
checks passed with no wrong outputs and the *loaded* driver/bridge
module paths and SHA-256 hashes independently verified. See
[raw numerical evidence](../benchmarks/cudnn-gfx1200-direct-hip-fp32-fp16-20261018.json).
FP16 tensor transform initially returned SUCCESS with zero output;
this silent numeric failure was fixed by strict cuDNN float32-host
to MIOpen binary16 tensor scalar conversion. Tests reject inexact
and non-finite scalar coefficients rather than silently rounding.

These are bounded direct-HIP device allocation / real MIOpen GPU tests,
**not CUDA-targeted PyTorch end-to-end evidence**. Only the declared
sigmoid/ReLU/tanh, accurate channel softmax, unpadded nonoverlapping
max pooling, transform/set/scale tensors and existing narrow legacy
cuDNN v8 convolution subset are within validated feature scope.
cuDNN v9/frontend, advanced fusions, unvalidated streams, many
normalizations, and other hardware remain open tasks.

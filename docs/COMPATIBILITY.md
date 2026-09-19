# Compatibility

Support in this project is reported per **hardware + software stack + capability/workload**. A GPU being detected does not mean every CUDA application is supported.

## GPU status

| GPU | Native target | Status | Evidence |
| --- | --- | --- | --- |
| Radeon RX 9060 XT | `gfx1200` | **validated reference** | Main development machine; stable public stack and experimental v7 validation |
| Radeon RX 9070 XT | `gfx1201` | **validated external** | Separately tested Windows AMD/ZLUDA training setup; see [`RX9070XT_VALIDATION.md`](RX9070XT_VALIDATION.md) |
| Radeon 890M | `gfx1150` | **community partial** | HIP 7.2/GEMM works; patched driver/PTX + SDPA safety fixes confirmed. The known legacy cuDNN conv2d hang is guarded by an automatic no-cuDNN PyTorch fallback; modern llama.cpp remains a separate application-kernel compatibility issue |
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

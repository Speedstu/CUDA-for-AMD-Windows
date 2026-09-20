# Scripts

The script filenames in this directory are part of the project's practical public interface.

**Existing entry points are intentionally kept in place.** Do not move or rename a public script just to reorganize the repository; old Reddit posts, issue comments, clones, automation and user notes may depend on these paths.

## Recommended user-facing commands

| Script | Purpose |
| --- | --- |
| `install.ps1` | Recommended end-to-end installer and validation entry point |
| `run-zluda.ps1` | Launch a CUDA-targeted application through the prepared runtime |
| `doctor.ps1` | Check HIP/runtime prerequisites |
| `gpu-scan.ps1` | Identify AMD GPU, native `gfx` target and project status |
| `test-runtime.ps1` | Runtime/library smoke test |
| `test-functional.ps1` | Focused numerical correctness checks |

## Compatibility entry points

| Script | Purpose |
| --- | --- |
| `setup.ps1` | Lower-level setup command used by the original project workflow; retained for compatibility |
| `stage-runtime.ps1` | Stage compatibility DLLs beside an application |
| `check-hip-compat.ps1` | Architecture-aware HIP SDK compatibility check |
| `verify.ps1` | Verify an already prepared runtime/configuration |

## Extended validation

| Script | Purpose |
| --- | --- |
| `test-capabilities.ps1` | Broad isolated CUDA-facing capability matrix |
| `capability_probe.py` | Per-capability Python probe implementation |
| `functional_probe.py` | Focused numerical functional probes |
| `test-llama-registration.ps1` | llama.cpp device/registration and optional driver preflight |
| `run-llama-zluda-safe.ps1` | Conservative gfx1150 llama.cpp smoke launcher with launch blocking, persistent kernel trace and guarded GPU-layer escalation |
| `test-velocityrl.ps1` | Optional real PPO integration smoke |
| `warmup-pytorch.ps1` | Prewarm expensive first-use PyTorch/ZLUDA compilation |
| `training_warmup.py` | Python workload used by the warmup helper |

## Maintainer / development tools

| Script | Purpose |
| --- | --- |
| `apply-zluda-patches.ps1` | Apply the machine-readable v7 patch series to pinned ZLUDA source |
| `capture-zluda-trace.ps1` | Capture an application ZLUDA trace for focused debugging |
| `summarize-zluda-trace.py` | Reduce trace output to useful diagnostics |
| `benchmark-gemm.ps1` | Same-GPU GEMM benchmark runner |
| `gemm_benchmark.py` | Python side of GEMM measurements |
| `build-cusolver-proxy.ps1` | Build the experimental cuSOLVER compatibility proxy |
| `stage-cusolver-proxy.ps1` | Stage that proxy reversibly for testing |
| `build-cudnn-bridge.ps1` | Build the experimental cuDNN v8 → MIOpen compatibility proxy from a user-supplied cuDNN DLL |
| `stage-cudnn-bridge.ps1` | Stage/restore that cuDNN proxy with SHA-256 state tracking |
| `test-cudnn-bridge.ps1` | Header-free ABI/correctness self-test for the cuDNN bridge |
| `test-cudnn-bridge-pytorch.py` | Reference PyTorch forward/backward training matrix for the experimental bridge |

## Stability rule

A cleanup/refactor must preserve the behavior and parameters of existing public commands unless there is a concrete compatibility reason to change them.

If an entry point ever needs replacement, keep a wrapper for the old path for at least one documented transition period.

Runtime behavior belongs in tests and source changes; repository organization should not silently alter compatibility semantics.

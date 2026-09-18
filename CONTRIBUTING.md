# Contributing

Thanks for helping improve CUDA compatibility on AMD GPUs under Windows.

This project is evidence-driven: a CUDA-facing call returning success is not enough by itself to label a capability or GPU as supported.

## Evidence levels

Use the narrowest status that the evidence actually proves.

| Level | Meaning | Example |
| --- | --- | --- |
| detected | The GPU/runtime can be enumerated | `gpu-scan.ps1`, `--list-devices` |
| loadable | The CUDA-facing DLL/API loads | `cuda_check.exe`, driver smoke |
| safe refusal | Unsupported work fails explicitly without wrong output or process damage | `NO_BINARY_FOR_GPU`, `NOT_SUPPORTED` |
| functional pass | The operation completes and matches a CPU/native reference within tolerance | capability/functional probes |
| integration validated | A real application/workload completes end-to-end and exits cleanly | PPO update, verified inference run |

Do not promote a result to a higher level without the corresponding evidence.

## Correctness first

The repository follows a fail-closed rule:

- a clean unsupported result is better than a silently wrong tensor;
- a timeout does not count as support;
- a correct result followed by a process crash does not count as a clean pass;
- an application silently falling back to CPU does not count as CUDA-on-AMD success;
- ignoring an unsupported CUDA launch attribute is not acceptable when doing so changes semantics;
- clamping a requested GPU resource beyond the advertised device limit is not a compatibility fix.

If a workaround changes execution semantics, document it as experimental and validate the numerical result independently.

## Before reporting a GPU

Run:

```powershell
.\scripts\gpu-scan.ps1
.\scripts\doctor.ps1
.\scripts\test-runtime.ps1
.\scripts\test-functional.ps1 -PythonExe C:\path\to\cuda-facing-venv\Scripts\python.exe
.\scripts\test-capabilities.ps1 -PythonExe C:\path\to\cuda-facing-venv\Scripts\python.exe
```

Attach the generated reports when practical. Current reports include runtime/probe hashes so results can be tied to the exact binaries that were tested.

For llama.cpp registration, use the model-free smoke first:

```powershell
.\scripts\test-llama-registration.ps1 -LlamaRoot C:\path\to\llama-cuda-build

# If a CUDA-facing PyTorch environment is available, include the focused
# driver metadata/launch preflight without loading a model:
.\scripts\test-llama-registration.ps1 `
  -LlamaRoot C:\path\to\llama-cuda-build `
  -PythonExe C:\path\to\cuda-facing-venv\Scripts\python.exe
```

Do not jump directly to risky inference kernels on an unvalidated GPU.

If a model run later fails only at a generic CUDA launch location, capture a ZLUDA trace as the final diagnostic step:

```powershell
.\scripts\capture-zluda-trace.ps1 `
  -Program C:\path\to\llama-cli.exe `
  -ProgramArgs @('-m','C:\path\to\small-test-model.gguf','-fa','off') `
  -AcknowledgeGpuResetRisk
```

Trace mode executes the application workload. The acknowledgement switch is intentional: killing the process cannot guarantee recovery from a GPU/driver hard lock.

## Compatibility reports

Please include:

- exact GPU model and `gfxXXXX` target;
- AMD driver;
- HIP/ROCm version;
- ZLUDA release/channel;
- repository commit;
- `ZLUDA_CC` when relevant;
- framework/application build;
- exact command;
- first useful error;
- whether the failure is unsupported, incorrect, timeout, process crash, or GPU reset;
- proof that the workload actually used the AMD/ZLUDA device rather than CPU fallback.

Use the GPU compatibility issue template so these fields are captured consistently.

## Numerical probes

A new capability probe should:

1. run in isolation;
2. use a deterministic seed where randomness is involved;
3. compare against a known-good CPU/native result when possible;
4. record finite/non-finite output;
5. use explicit tolerances;
6. return a machine-readable status;
7. avoid hiding unsupported behavior behind a fallback;
8. be small enough to diagnose but representative enough to exercise the real backend path.

The PowerShell runner is responsible for timeouts, child-process exit status, teardown hangs, and post-result crashes.

## Patch lifecycle

Experimental ZLUDA source changes follow this progression:

1. reproduce the upstream failure;
2. create a focused regression probe;
3. make the smallest semantic fix possible;
4. verify the patch applies to the pinned upstream commit;
5. compile the affected Windows target in CI;
6. test the rebuilt DLL on the reference GPU;
7. run the focused probe;
8. run the broader capability matrix;
9. run the real application that motivated the fix;
10. only then merge the change into the validated patch set and update support claims.

Candidate patches may live separately under `patches/` while steps 6-9 are pending.

## CI

The `verify` workflow checks:

- PowerShell syntax;
- Python probe syntax;
- required critical probes;
- JSON manifests;
- GPU scanner no-GPU behavior;
- patch applicability against the pinned ZLUDA source;
- pinned release hashes.

The separate candidate-build workflow recompiles the ZLUDA Windows driver for selected experimental source patches. A successful compile is necessary but **not** proof that the patch works correctly on an AMD GPU.

## Third-party binaries

Do not commit proprietary NVIDIA or AMD binaries unless their redistribution terms explicitly permit it and the repository has intentionally adopted that distribution model.

Prefer:

- source patches;
- reproducible build scripts;
- hashes of externally downloaded official assets;
- reversible local staging.

## Documentation claims

When changing a support table or score, state:

- hardware;
- software stack;
- date/version;
- number of tests;
- safe refusals separately from clean passes;
- whether the number is current or a historical snapshot.

If the probe list changes, an old percentage must not be presented as the score of the new matrix until the matrix has been rerun.

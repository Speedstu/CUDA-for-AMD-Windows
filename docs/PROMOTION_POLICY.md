# Candidate promotion policy

A green CI badge is **not** evidence that an AMD GPU calculated the right answer.
This project uses three separate levels:

1. **Compiled candidate** — pinned source compiled, patch series applied, artifacts hashed. No GPU correctness claim.
2. **Hardware tested** — identified GPU, driver SHA-256, versions, exact test commands and output. Scope is the specific workload tested, not all of CUDA.
3. **Eligible for manual review** — independent numerical tests meet the evidence gate below, with no silent wrong result, crash or timeout. A human maintainer still decides whether to publish.

The safeguard is **fail-closed and read-only**. It NEVER updates an installation, marks GitHub releases stable, or replaces existing runtime DLLs.

## Validation

Stage a candidate **in a fresh folder**, with a copy of the known-good runtime available for rollback. Run the available test suites against the same staged driver:

```powershell
.\scripts\test-capabilities.ps1 -RuntimeRoot C:\path\to\runtime -PythonExe C:\path\to\venv\Scripts\python.exe -Tests driver_function_metadata,driver_ptx_selection,cub_sort_scan,driver_buffer_clear -ReportPath C:\temp\capabilities.json
.\scripts\test-functional.ps1 -RuntimeRoot C:\path\to\runtime -PythonExe C:\path\to\venv\Scripts\python.exe -Tests matmul,sdpa_math -ReportPath C:\temp\functional.json
```

The targeted functional selection avoids raw fused-SDPA kernels that are already known to return incorrect results on some GPUs. Do not run the full experimental suite merely to qualify a release.

Then assess the evidence:

```powershell
python .\scripts\promotion_gate.py --capabilities C:\temp\capabilities.json --functional C:\temp\functional.json --driver C:\path\to\staged\nvcuda.dll --arch gfx1150 --revision <full-40-character-project-commit>
```

The gate verifies:

- Full Git revision and target AMD architecture match **both** reports.
- Measured SHA-256 matches the **actual supplied** driver DLL, for both reports.
- Reports have valid UTC timestamps no older than 7 days.
- Required capabilities are present, passed and exited cleanly: function metadata, fatbin target selection, **CUB sort/scan at 8192 and 65536 elements**, driver buffer clear, and a pitched non-default-stream cuMemcpy2DAsync_v2 check.
- The functional report has clean GPU numerical tests for at least **matmul** and **SDPA math**.
- Any reported incorrect result, timeout, crash, hang or missing core result blocks promotion even if a summary field says PASS.

Exit code **0** means *evidence is eligible for manual review*, **1** means failed tests or mismatched evidence, and **2** means invalid input.

The gate requires two independent forms of driver evidence for **every passing
GPU probe**:

- the SHA-256 of the staged driver recorded in the report, and
- the SHA-256 and actual file path of the nvcuda.dll resolved with
  GetModuleHandleW/GetModuleFileNameW from **inside the running probe**.
  A correct staged driver with a different loaded DLL is a failed gate.

A missing, unreadable or mismatching loaded module now **fails closed**,
even if the probe's numerical result says PASS. This does not, by itself,
validate a llama-cli process launched *outside* the probe runner: include
the child process loaded-module path and hash separately for llama MTP
tests, as shown in the issue #6 protocol.
Older evidence reports without per-probe loaded-module hashes are
**historical**, not automatically eligible for promotion under this
updated policy. Re-run the required safe, bounded probes when reviewing
a candidate; do not repeat previously crashing fused kernels.

Use a **new** report for every change in driver DLL, HIP SDK, GPU model/architecture, Python/PyTorch, ZLUDA version, compiler, or flags. An older passing report never validates a newer binary. Test other GPU architectures independently. For MTP, follow [the gfx1150 reproducible correctness plan](GFX1150_MTP_CORRECTNESS.md) and include dedicated token/output comparisons against native ROCm; a CUB pass alone cannot establish MTP correctness. For changes to GPU kernel compilation or policy, add regression tests on both sides of the failing threshold.

## Current precautions

- Candidate drivers built before the October 12 metadata correction (such as 0c2bee2, 07638a7, c810f91 and a2ac6e6) can corrupt PyTorch CUB-backed results without throwing. Do not promote or use them for correctness-sensitive work.
- A GitHub Actions artifact with successful Rust/CUDA compilation is *candidate-only*, regardless of its name.
- Do not turn off Windows Defender, replace system DLLs globally, or overwrite the known-good runtime to get a passing test.
- Do not mark issue #3 or #6 fixed merely because the new candidate compiles. Community gfx1150 results and correctness comparisons are required.

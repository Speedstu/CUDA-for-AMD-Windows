# Radeon RX 9070 XT / gfx1201 validation evidence

The project has archived validation evidence for **AMD Radeon RX 9070 XT (`gfx1201`)** on Windows. This is tracked separately from the current maintainer reference GPU, the RX 9060 XT (`gfx1200`).

## Evidence recovered from the archived test bundle

A historical AMD/ZLUDA GigaLearnCPP package was built specifically for a separately tested RX 9070 XT machine. The package records the following topology and launch configuration:

- discrete GPU: **AMD Radeon RX 9070 XT / `gfx1201`**;
- the RX 9070 XT was physical HIP device `1` on that machine;
- an integrated AMD GPU, when present, occupied physical HIP device `0`;
- launchers forced `HIP_VISIBLE_DEVICES=1`, `ROCR_VISIBLE_DEVICES=1` and `GPU_DEVICE_ORDINAL=1`;
- the workload was a CUDA-facing LibTorch/GigaLearnCPP training build launched through ZLUDA;
- the portable AMD runtime packaged for that setup contains `gfx1201`-specific hipBLASLt/Tensile code objects and libraries.

The archived bundle also contains a prebuilt CUDA-facing GigaLearnCPP executable, CUDA LibTorch libraries, ZLUDA runtime DLLs, and the portable AMD HIP math runtime used by that setup.

## Project status

The RX 9070 XT is therefore marked **`validated-external`** rather than `unverified-candidate`.

This status deliberately does **not** mean that every CUDA API is proven on the card. The archived package does not contain a complete machine-readable capability report from that separate machine, so the evidence is strongest for the packaged AMD/ZLUDA training setup and its `gfx1201` runtime path.

The RX 9060 XT remains the project's `validated-reference` device because it is the hardware currently available for repeatable local regression testing.

## Revalidation on another RX 9070 XT

On a current RX 9070 XT system, run:

```powershell
.\scripts\gpu-scan.ps1 -OutputPath .\gpu-report.json
.\scripts\test-runtime.ps1
.\scripts\test-functional.ps1 -PythonExe C:\path\to\venv\Scripts\python.exe
.\scripts\test-capabilities.ps1 -PythonExe C:\path\to\venv\Scripts\python.exe
```

A fresh capability report is still recommended because driver, HIP/ROCm, ZLUDA and PyTorch versions can change behavior independently of the GPU architecture.

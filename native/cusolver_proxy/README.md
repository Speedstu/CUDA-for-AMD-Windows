# cuSOLVER → hipSOLVER compatibility proxy

This directory contains a **source-only experimental cuSOLVER compatibility shim** for Windows.

The goal is to port cuSOLVER functionality incrementally to AMD hipSOLVER without pretending that the entire cuSOLVER API has already been reimplemented.

## Design

`build-cusolver-proxy.ps1` reads the export table from the user's existing `cusolver64_11.dll` and generates a proxy with the **same export surface**.

Validated functions are implemented locally and routed to `hipsolver.dll`:

- `cusolverDnCreate`
- `cusolverDnDestroy`
- `cusolverDnSetStream`
- `cusolverDn[S/D/C/Z]getrf_bufferSize`
- `cusolverDn[S/D/C/Z]getrf`
- `cusolverDn[S/D/C/Z]getrs`

Every other export is forwarded to a user-supplied copy of the original library named `cusolver64_11_nvidia.dll`. This keeps applications loadable while AMD-native coverage is expanded family by family.

No NVIDIA or AMD binary is stored in this repository.

## Build and stage

MinGW-w64 `gcc.exe` and GNU `objdump.exe` are currently required for the automatic builder.

For a CUDA PyTorch environment, point the staging script at the directory containing `cusolver64_11.dll`:

```powershell
.\scripts\stage-cusolver-proxy.ps1 `
  -TargetDir C:\path\to\venv\Lib\site-packages\torch\lib
```

The script will:

1. inspect all exports of the installed cuSOLVER DLL;
2. compile a matching proxy;
3. rename the user's original to `cusolver64_11_nvidia.dll`;
4. install the proxy as `cusolver64_11.dll`;
5. record SHA-256 state so the change can be reversed safely.

Restore the original DLL with:

```powershell
.\scripts\stage-cusolver-proxy.ps1 `
  -TargetDir C:\path\to\venv\Lib\site-packages\torch\lib `
  -Restore
```

## Validation

On the RX 9060 XT / `gfx1200` experimental ZLUDA v7 + TheRock stack, the generated proxy preserved **940/940 exports** of the tested CUDA 11 cuSOLVER DLL. The proxy also skips `hipsolverDnDestroy` only when Windows reports that DLL shutdown is already in progress; this prevents the same late-teardown `0xC0000409` fast-fail seen with other ROCm-backed cached handles while preserving normal runtime destruction.

`torch.linalg.solve` was validated against CPU references for:

- FP32
- FP64
- complex64
- complex128

The test can be repeated with:

```powershell
.\scripts\test-capabilities.ps1 `
  -RuntimeRoot C:\path\to\runtime `
  -PythonExe C:\path\to\venv\Scripts\python.exe `
  -Tests linalg_solve `
  -Strict
```

This currently validates the LU solve path only. QR, Cholesky, SVD, eigenvalue/eigenvector, batched and newer generic-X APIs still need AMD-native coverage and may fall through to the original library.

# cuSOLVER → hipSOLVER compatibility proxy

This directory contains a **source-only experimental cuSOLVER compatibility shim** for Windows.

The goal is to port cuSOLVER functionality incrementally to AMD hipSOLVER without pretending that the entire cuSOLVER API has already been reimplemented.

## Design

`build-cusolver-proxy.ps1` reads the export table from the user's existing `cusolver64_11.dll` and generates a proxy with the **same export surface**.

Validated functions are implemented locally and routed to `hipsolver.dll`:

- `cusolverDnCreate` / `cusolverDnDestroy` / `cusolverDnSetStream`
- generic parameter management: `cusolverDnCreateParams`, `cusolverDnDestroyParams`, and the mapped subset of `cusolverDnSetAdvOptions`
- LU: `cusolverDn[S/D/C/Z]getrf_bufferSize`, `getrf`, and `getrs`
- legacy Cholesky: `cusolverDn[S/D/C/Z]potrf_bufferSize`, `potrf`, `potrfBatched`, `potri_bufferSize`, `potri`, `potrs`, and `potrsBatched`
- generic-X Cholesky used by current PyTorch: `cusolverDnXpotrf_bufferSize`, `cusolverDnXpotrf`, and `cusolverDnXpotrs`
- legacy QR: `cusolverDn[S/D/C/Z]geqrf_bufferSize`, `geqrf`, `orgqr`/`ungqr`, and `ormqr`/`unmqr` families
- generic-X QR used by current PyTorch: `cusolverDnXgeqrf_bufferSize` and `cusolverDnXgeqrf`
- Jacobi SVD: `GesvdjInfo` management plus S/D/C/Z `gesvdj` and `gesvdjBatched` buffer/execute paths
- symmetric/Hermitian eigensolvers: S/D `syevd` + C/Z `heevd`, Jacobi `syevj`/`heevj` families, and generic-X `cusolverDnXsyevd[_bufferSize]`

Every other export is retained as a forwarder to a user-supplied copy of the original library named `cusolver64_11_nvidia.dll`. This preserves the original DLL's export/load surface while AMD-native coverage is expanded family by family. A forwarded symbol is **not** a claim that the original NVIDIA implementation can execute on AMD; handle-dependent unported routines should be treated as unvalidated until they receive an explicit hipSOLVER route.

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

On the RX 9060 XT / `gfx1200` experimental ZLUDA v7 + TheRock stack, the generated proxy preserved **940/940 exports** of the tested CUDA 11 cuSOLVER DLL and currently routes **131 entry points** to hipSOLVER. The proxy also skips `hipsolverDnDestroy` only when Windows reports that DLL shutdown is already in progress; this prevents the same late-teardown `0xC0000409` fast-fail seen with other ROCm-backed cached handles while preserving normal runtime destruction.

`torch.linalg.solve` was validated against CPU references for FP32, FP64, complex64 and complex128. Cholesky coverage was additionally validated for the same four dtypes, in both single-matrix and batched tensor cases, using:

- `torch.linalg.cholesky`
- `torch.cholesky_solve`
- `torch.cholesky_inverse`

QR coverage is additionally validated with `torch.linalg.qr`, `torch.geqrf`, `torch.orgqr`/`ungqr`, and `torch.ormqr`/`unmqr` across FP32, FP64, complex64 and complex128, including batched matrices. The same proxy also validates `torch.linalg.svd`, `torch.linalg.pinv`, `torch.linalg.eigh`, and `torch.linalg.eigvalsh`; dedicated probes cover both single and batched inputs, with deeper four-dtype sweeps used during development.

The test can be repeated with:

```powershell
.\scripts\test-capabilities.ps1 `
  -RuntimeRoot C:\path\to\runtime `
  -PythonExe C:\path\to\venv\Scripts\python.exe `
  -Tests @('linalg_solve','linalg_cholesky','linalg_qr','linalg_inv','linalg_lstsq','linalg_svd','linalg_pinv','linalg_eigh','linalg_eigvalsh') `
  -Strict
```

The generic-X bridge deliberately maps only the validated FP32, FP64, complex64 and complex128 data types; unknown generic-X data types return `NOT_SUPPORTED` instead of being passed through unchecked. Many less-common cuSOLVER families still remain forwarded and should be treated as unvalidated until they receive an explicit AMD-native route.

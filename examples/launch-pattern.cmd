@echo off
rem Generic launch pattern recovered from the working stack.
rem Prefer scripts\run-zluda.ps1 for real use.
set "ZLUDA_CC=8.6"
set "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1"
set "ROCBLAS_TENSILE_LIBPATH=%HIP_PATH%\bin\rocblas\library"
set "HIPBLASLT_TENSILE_LIBPATH=%HIP_PATH%\bin\hipblaslt\library"
rem Example:
rem C:\path\to\zluda.exe -- C:\path\to\app.exe

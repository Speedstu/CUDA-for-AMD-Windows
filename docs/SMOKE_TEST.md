# Smoke-test status

Date: 2026-09-13

A recovered ZLUDA `cuda_check.exe` was launched through the reconstructed runtime with:

- ZLUDA v6-preview.69 core
- recovered custom BLAS overlay
- ROCm 6.4 HIP root
- LibTorch 2.3.0+cu118 on PATH
- `ZLUDA_CC=8.6`

Observed stdout before the 20-second test timeout:

```text
cublaslt12: OK (C:\Program Files\AMD\ROCm\6.4\bin\hipblaslt.dll)
```

The probe did not exit within 20 seconds, so this is recorded as a **partial smoke**, not a full runtime pass. It proves the staged process reached the expected AMD hipBLASLt library through the reconstructed environment, but it does not establish complete CUDA API compatibility.

The historical benchmark/training logs remain the stronger evidence that this stack previously executed the intended LibTorch PPO workload successfully.

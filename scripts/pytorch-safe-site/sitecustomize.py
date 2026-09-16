"""Opt-in PyTorch compatibility hook for CUDA-on-AMD/ZLUDA.

Loaded only when scripts/run-zluda.ps1 is invoked with -PyTorchSafeSDPA.
It disables fused SDPA backends that require NVIDIA-only cubins on the
validated PyTorch 2.0.1+cu118 stack, leaving the numerically validated math
backend enabled.
"""
from __future__ import annotations

import os
import warnings

if os.environ.get("CUDAAMD_PYTORCH_SAFE_SDPA") == "1":
    try:
        import torch

        cuda_backends = getattr(torch.backends, "cuda", None)
        if cuda_backends is not None:
            disable_flash = getattr(cuda_backends, "enable_flash_sdp", None)
            disable_mem = getattr(cuda_backends, "enable_mem_efficient_sdp", None)
            if disable_flash is not None:
                disable_flash(False)
            if disable_mem is not None:
                disable_mem(False)
    except Exception as exc:  # Do not make unrelated Python programs unlaunchable.
        warnings.warn(f"CUDA-on-AMD PyTorch SDPA safe mode could not initialize: {exc!r}")
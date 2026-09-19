"""PyTorch compatibility hook for CUDA-on-AMD/ZLUDA.

Loaded by scripts/run-zluda.ps1 and selected capability probes when a tested
path needs to fail safe or avoid a known hanging backend.
"""
from __future__ import annotations

import os
import warnings

safe_sdpa = os.environ.get("CUDAAMD_PYTORCH_SAFE_SDPA") == "1"
safe_conv2d = os.environ.get("CUDAAMD_PYTORCH_SAFE_CONV2D") == "1"

if safe_sdpa or safe_conv2d:
    try:
        import torch

        if safe_sdpa:
            cuda_backends = getattr(torch.backends, "cuda", None)
            if cuda_backends is not None:
                disable_flash = getattr(cuda_backends, "enable_flash_sdp", None)
                disable_mem = getattr(cuda_backends, "enable_mem_efficient_sdp", None)
                if disable_flash is not None:
                    disable_flash(False)
                if disable_mem is not None:
                    disable_mem(False)

        if safe_conv2d:
            # gfx1150 community testing shows the legacy cuDNN/ZLUDA path can
            # hang indefinitely. PyTorch's no-cuDNN convolution fallback is
            # slower, but it avoids that unsafe path and remains numerically
            # verifiable by the capability probe.
            torch.backends.cudnn.enabled = False
    except Exception as exc:  # Do not make unrelated Python programs unlaunchable.
        warnings.warn(f"CUDA-on-AMD PyTorch safe mode could not initialize: {exc!r}")

"""Static regression for global missing-MIOpen conv2d safety fallbacks.

The gfx1150 performance policy stays conservative; no MIOpen dependency
on any other AMD architecture also disables cuDNN in wrapper-spawned
PyTorch processes. Hardware CUDA probe tests are independent.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
wrapper = (ROOT / "scripts/run-zluda.ps1").read_text(encoding="utf-8")
harness = (ROOT / "scripts/test-capabilities.ps1").read_text(encoding="utf-8")
site = (ROOT / "scripts/pytorch-safe-site/sitecustomize.py").read_text(encoding="utf-8")
assert "$miopenDllPresent" in wrapper
assert "-not $miopenDllPresent -or" in wrapper
assert "$gpuArch -eq 'gfx1150' -and -not $bridgeMiopenReady" in wrapper
assert "-not $AllowUnsafeCudnnConv" in wrapper
assert "$env:CUDAAMD_PYTORCH_SAFE_CONV2D" in wrapper
assert "Remove-Item Env:ROCR_VISIBLE_DEVICES" in wrapper
assert "$noMiopenBackend" in harness
assert "$Name -eq 'conv2d'" in harness
assert "($noMiopenBackend -or" in harness
assert "CUDAAMD_PYTORCH_SAFE_CONV2D" in harness
assert "torch.backends.cudnn.enabled = False" in site
print("PASS: missing MIOpen safety guard is architecture-independent, gfx1150 safety preserved")

"""Offline regression for CUDA -> HIP async pitched copy candidate."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
patch = (root / "patches/zluda-v7-preview10/memcpy-2d-async-candidate.patch").read_text("utf-8")
cap = (root / "scripts/capability_probe.py").read_text("utf-8")
assert "+            cuMemcpy2DAsync_v2," in patch, "Driver symbol not enabled"
assert "+pub(crate) unsafe fn copy_2d_async_v2(memcpy: hip_Memcpy2D, stream: hipStream_t)" in patch
assert "+    hipMemcpyParam2DAsync(&memcpy, stream)?;" in patch, "Not actually asynchronous"
assert "+    hipMemcpyParam2D(&memcpy)" not in patch, "Do not silently replace async with sync"
assert "def run_driver_memcpy2d_async(torch):" in cap
assert '"driver_memcpy2d_async": run_driver_memcpy2d_async' in cap
assert "src_pitch = 41216" in cap and "dst_pitch = 288" in cap
print("PASS: enabled CUDA async 2D symbol, HIP async forwarding, pitched GPU regression registered")

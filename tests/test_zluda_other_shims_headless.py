"""Bounded subprocess regression for non-modal missing-HIP dependencies.

This tests the *separately built* Rust shim candidates in an isolated
Windows process and never loads or changes the installed ZLUDA runtime.
A successful return from a function alone is not evidence of GPU math.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

SHIMS = (
    ("cublas.dll", "cublasCreate_v2"),
    ("cublaslt.dll", "cublasLtCreate"),
    ("cusparse.dll", "cusparseCreate"),
    ("cufft.dll", "cufftCreate"),
)

def single_child(path: Path, symbol: str) -> None:
    dll = ctypes.WinDLL(str(path))
    fn = getattr(dll, symbol)
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    fn.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    start = time.perf_counter()
    status = fn(ctypes.byref(handle))
    elapsed = time.perf_counter() - start
    print(json.dumps({
        "dll": path.name,
        "symbol": symbol,
        "status": int(status),
        "elapsed_s": round(elapsed, 5),
        "handle_created": bool(handle.value),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin-dir", type=Path)
    p.add_argument("--child", type=Path)
    p.add_argument("--symbol")
    args = p.parse_args()
    if args.child:
        single_child(args.child.resolve(strict=True), args.symbol)
        return
    if not args.bin_dir:
        p.error("--bin-dir is required")
    if os.name != "nt":
        raise RuntimeError("Windows-only DLL ABI tests")
    directory = args.bin_dir.resolve(strict=True)
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env.pop("ZLUDA_ALLOW_MISSING_DLL_DIALOG", None)
        env.pop("ZLUDA_NO_DIALOG", None)  # unused legacy variable must not control the fix
        env["HIP_PATH"] = str(Path(tmp) / "hip-missing")
        # Do not put any model/game/installed runtime next to these candidates.
        env["PATH"] = str(directory) + os.pathsep + os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32"
        )
        for name, symbol in SHIMS:
            dll = directory / name
            if not dll.is_file():
                raise AssertionError(f"Missing separately built candidate {dll}")
            start = time.perf_counter()
            try:
                child = subprocess.run(
                    [sys.executable, __file__, "--child", str(dll), "--symbol", symbol],
                    capture_output=True, text=True, timeout=8, env=env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )
            except subprocess.TimeoutExpired as exc:
                raise AssertionError(f"{name}/{symbol}: likely modal loader hang") from exc
            if child.returncode != 0:
                raise AssertionError(f"{name}/{symbol} exited {child.returncode}: {child.stderr[-1600:]}")
            data = json.loads(child.stdout.strip())
            assert data["dll"] == name and data["symbol"] == symbol
            assert len(data["sha256"]) == 64 and data["elapsed_s"] < 5
            print(f"PASS: {name}/{symbol} returned status {data['status']} "
                  f"in {data['elapsed_s']}s without a modal stall "
                  f"(elapsed subprocess {time.perf_counter()-start:.2f}s)")

if __name__ == "__main__":
    main()

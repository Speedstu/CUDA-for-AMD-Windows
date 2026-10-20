#!/usr/bin/env python3
"""Regression: missing HIP components must not open modal dialogs or stall.
The source check runs in ordinary CI; the optional Windows DLL probes test
real freshly built ZLUDA cuDNN v8/v9 candidate DLLs with missing MIOpen.
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
import time

REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "patches/zluda-v7-preview10/headless-loader-cudnn-property-candidate.patch"


def source_checks() -> None:
    p = PATCH.read_text("utf-8")
    manifest = json.loads((PATCH.parent / "series.json").read_text("utf-8"))
    assert any(item["file"] == PATCH.name and item["status"] == "candidate"
               for item in manifest["patches"]), "Candidate patch absent from pinned series"
    assert "ZLUDA_ALLOW_MISSING_DLL_DIALOG" in p
    assert "std::io::Write::write_all" in p
    assert "+            return None;" in p, "Must stop before TaskDialogIndirect by default"
    assert "cudnnGetProperty" in p and "get_property" in p
    assert "CUDNN_MAJOR" in p and "CUDNN_MINOR" in p and "CUDNN_PATCHLEVEL" in p
    assert "if value.is_null()" in p and "BAD_PARAM" in p
    print("PASS: pinned source candidate fails closed on missing HIP and writes cuDNN 8/9 properties")


def child_probe(dll: str, expected: tuple[int, int, int]) -> None:
    if not hasattr(ctypes, "WinDLL"):
        raise RuntimeError("Windows DLL child test requires Windows")
    target = Path(dll).resolve(strict=True)
    sdk = ctypes.WinDLL(str(target))
    sdk.cudnnGetProperty.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    sdk.cudnnGetProperty.restype = ctypes.c_int
    sdk.cudnnGetVersion.restype = ctypes.c_size_t
    ver = sdk.cudnnGetVersion()
    expected_ver = (1000*expected[0] + 100*expected[1] + expected[2]
                    if expected[0] < 9 else 10000*expected[0] + 100*expected[1] + expected[2])
    if ver != expected_ver:
        raise AssertionError(f"Version mismatched: reported={ver}, expected={expected_ver}")
    for name, index, value in zip(("major", "minor", "patch"), range(3), expected):
        result = ctypes.c_int(-123456)
        status = sdk.cudnnGetProperty(index, ctypes.byref(result))
        if status or result.value != value:
            raise AssertionError(f"{name}: status={status}, value={result.value}, expected={value}")
    for bad in (-1, 3, 999):
        result = ctypes.c_int(-123456)
        if sdk.cudnnGetProperty(bad, ctypes.byref(result)) == 0 or result.value != -123456:
            raise AssertionError("Invalid property accepted or wrote output")
    if sdk.cudnnGetProperty(0, None) == 0:
        raise AssertionError("NULL value pointer must not be accepted")
    sdk.cudnnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    sdk.cudnnCreate.restype = ctypes.c_int
    h = ctypes.c_void_p()
    t = time.perf_counter()
    status = sdk.cudnnCreate(ctypes.byref(h))
    elapsed = time.perf_counter() - t
    if status == 0 or h.value is not None:
        raise AssertionError(f"Missing MIOpen must fail initialization; status={status}, handle={h.value}")
    if elapsed > 5:
        raise AssertionError(f"Missing MIOpen returned too slowly after {elapsed:.2f}s")
    print(json.dumps({"version": ver, "property_values": list(expected),
                      "missing_miopen_status": status, "elapsed_seconds": round(elapsed, 4),
                      "dll_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "no_modal_block": True}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dnn8", type=Path)
    p.add_argument("--dnn9", type=Path)
    p.add_argument("--child")
    p.add_argument("--expected")
    args = p.parse_args()
    if args.child:
        child_probe(args.child, tuple(json.loads(args.expected)))
        return
    source_checks()
    if not args.dnn8 and not args.dnn9:
        return
    if not sys.platform.startswith("win"):
        raise RuntimeError("DLL subprocess check requires Windows")
    for filename, expected in ((args.dnn8, [8, 9, 7]), (args.dnn9, [9, 13, 1])):
        if not filename:
            continue
        dll = filename.resolve(strict=True)
        env = os.environ.copy()
        env.pop("ZLUDA_ALLOW_MISSING_DLL_DIALOG", None)
        env["HIP_PATH"] = str(dll.parent / "nonexistent-HIP-SDK")
        env["ZLUDA_NO_DIALOG"] = "1"
        t = time.perf_counter()
        try:
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--child", str(dll), "--expected", json.dumps(expected)],
                    text=True, capture_output=True, timeout=12, env=env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as e:
            raise AssertionError(f"{dll.name}: missing MIOpen caused blocking dialog") from e
        if child.returncode:
            raise AssertionError(f"{dll.name} child rc={child.returncode}\n"
                                 f"stdout={child.stdout[-1200:]}\nstderr={child.stderr[-1600:]}")
        if "[zluda-dependency]" not in child.stderr:
            raise AssertionError(f"{dll.name}: missing MIOpen diagnostic absent: {child.stderr}")
        print(f"PASS: {dll.name} properties and missing MIOpen no-dialog "
              f"elapsed={time.perf_counter()-t:.2f}s child={child.stdout.strip()}")


if __name__ == "__main__":
    main()

"""Identify the actual CUDA driver module loaded in the *current process*.

The launcher's DLL redirection can override a file staged next to a binary.
Therefore a staged-file hash alone does not prove which nvcuda.dll a GPU probe
used. This helper is read-only and does not load nvcuda.dll itself.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def loaded_module_identity(module: str = "nvcuda.dll") -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "unavailable", "reason": "windows_only"}
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        kernel.GetModuleFileNameW.argtypes = [
            ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint
        ]
        kernel.GetModuleFileNameW.restype = ctypes.c_uint
        handle = kernel.GetModuleHandleW(module)
        if not handle:
            return {"status": "not_loaded"}
        for capacity in (1024, 32768):
            buffer = ctypes.create_unicode_buffer(capacity)
            written = kernel.GetModuleFileNameW(handle, buffer, capacity)
            if not written:
                return {"status": "error", "reason": "GetModuleFileNameW_failed"}
            if written < capacity - 1:
                path = Path(buffer.value).resolve(strict=True)
                return {
                    "status": "loaded",
                    "path": str(path),
                    "sha256": _sha256(path),
                }
        return {"status": "error", "reason": "module_path_truncated"}
    except (OSError, ValueError) as exc:
        return {"status": "error", "reason": type(exc).__name__}

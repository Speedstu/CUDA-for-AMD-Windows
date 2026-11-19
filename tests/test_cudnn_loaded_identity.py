"""Verify that conv2d carries its loaded cuDNN DLL identity and hash."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("capability_probe_for_cudnn_identity", SCRIPTS / "capability_probe.py")
cap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap)

def fake_identity(name="nvcuda.dll"):
    return {"status": "loaded", "path": "H:/candidate/" + name, "sha256": "a" * 64}

def capture(name, status, **kw):
    out, err = io.StringIO(), io.StringIO()
    with patch.object(cap, "loaded_module_identity", side_effect=fake_identity):
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cap.emit(name, status, **kw)
    payload = json.loads(out.getvalue())
    assert err.getvalue().strip() == "CUDAAMD_RESULT:" + out.getvalue().strip()
    return payload

conv = capture("conv2d", "pass", path="cudnn_or_bridge")
assert conv["loaded_nvcuda"]["path"].endswith("nvcuda.dll")
assert conv["loaded_cudnn64_8"]["path"].endswith("cudnn64_8.dll")
assert conv["loaded_cudnn64_8"]["sha256"] == "a" * 64
assert conv["path"] == "cudnn_or_bridge"

failure = capture("conv2d", "unsupported", error="MIOpen.dll could not be found")
assert "loaded_cudnn64_8" in failure, "DLL identity must be captured for failure paths too"

other = capture("matmul", "pass")
assert "loaded_cudnn64_8" not in other, "Only conv2d carries cuDNN-specific loaded identity"
print("PASS: conv2d success/failure records in-process cuDNN shim identity")

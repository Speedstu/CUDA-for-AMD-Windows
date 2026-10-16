"""Read-only validation that we hash an actually loaded Windows DLL."""
import hashlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import loaded_driver_identity as identity
import capability_probe as capabilities
import functional_probe as functional


class LoadedModuleTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows loader only")
    def test_real_loaded_kernel32_path_matches_file_digest(self):
        evidence = identity.loaded_module_identity("kernel32.dll")
        self.assertEqual(evidence.get("status"), "loaded", evidence)
        path = Path(evidence["path"])
        self.assertTrue(path.is_file(), evidence)
        self.assertEqual(evidence["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(len(evidence["sha256"]), 64)

    def test_missing_module_is_not_loaded(self):
        evidence = identity.loaded_module_identity("__cudaamd_nonexistent__module__.dll")
        self.assertIn(evidence["status"], ("not_loaded", "unavailable"))

    def test_capability_probe_emits_loaded_driver_proof(self):
        import contextlib
        import io
        import json
        with patch.object(capabilities, "loaded_module_identity", return_value={
            "status": "loaded", "sha256": "c" * 64, "path": "C:\\ZLUDA\\nvcuda.dll"
        }):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                with contextlib.redirect_stderr(io.StringIO()):
                    capabilities.emit("test", "pass", ok=True)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["loaded_nvcuda"]["sha256"], "c" * 64)

    def test_functional_probe_emits_loaded_driver_proof(self):
        import contextlib
        import io
        import json
        with patch.object(functional, "loaded_module_identity", return_value={
            "status": "loaded", "sha256": "d" * 64, "path": "C:\\ZLUDA\\nvcuda.dll"
        }):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                with contextlib.redirect_stderr(io.StringIO()):
                    functional.emit("matmul", "pass", ok=True)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["loaded_nvcuda"]["sha256"], "d" * 64)


if __name__ == "__main__":
    unittest.main()

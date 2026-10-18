"""Fail-closed audit of checked-in *historical* gfx1200 cuDNN GPU evidence.

CI does not have AMD hardware, so this fixture asserts evidence integrity and
scope, not that the current binary executes correctly on every GPU.
"""
from collections import Counter
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks/cudnn-gfx1200-direct-hip-fp32-fp16-20261018.json"


class HardwareEvidenceSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(PATH.read_text(encoding="utf-8-sig"))

    def test_real_gpu_scope_explicit_and_not_pytorch(self):
        d = self.data
        self.assertEqual(d["schema"], 1)
        self.assertEqual(d["kind"], "real-gpu-direct-hip-miopen")
        self.assertIn("NOT PyTorch", d["backend_scope"])
        self.assertEqual(d["hardware"]["gpu"], "AMD Radeon RX 9060 XT")
        self.assertEqual(d["hardware"]["architecture"], "gfx1200")
        self.assertEqual(d["hardware"]["hip_device_count"], 1)

    def test_real_loaded_dll_provenance(self):
        runtime = self.data["runtime"]
        self.assertTrue(runtime["loaded_module_paths_and_digests_verified"])
        for key in ("hip_dll_sha256", "miopen_dll_sha256",
                    "cudnn_bridge_sha256", "package_sha256"):
            self.assertRegex(runtime[key], r"^[0-9a-fA-F]{64}$")

    def test_all_eight_distinct_operations_each_have_fp32_fp16_result(self):
        evidence = self.data["numeric"]
        self.assertEqual(evidence["correctness_tests"], 16)
        self.assertEqual(evidence["passed"], 16)
        self.assertEqual(evidence["failed"], 0)
        rows = evidence["results"]
        self.assertEqual(len(rows), 16)
        op_counts = Counter((row["operator"], row["dtype"]) for row in rows)
        self.assertEqual(len(op_counts), 16)
        self.assertTrue(all(v == 1 for v in op_counts.values()))
        self.assertEqual(Counter(row["dtype"] for row in rows),
                         {"float32": 8, "float16": 8})
        for row in rows:
            self.assertEqual(row["status"], "pass", row)
            tol = (evidence["fp32_tolerance"] if row["dtype"] == "float32"
                   else evidence["fp16_tolerance"])
            self.assertLessEqual(abs(row["forward_max_abs"]), tol, row)
            if "backward_max_abs" in row:
                self.assertLessEqual(abs(row["backward_max_abs"]), tol, row)

    def test_limitations_not_overclaimed(self):
        limitations = " ".join(self.data["limitations"]).lower()
        for needed in ("pytorch", "frontend/v9", "streams", "float32"):
            self.assertIn(needed, limitations)


if __name__ == "__main__":
    unittest.main()

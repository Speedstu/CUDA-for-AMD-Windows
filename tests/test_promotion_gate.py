"""Offline tests for the fail-closed AMD/ZLUDA promotion evidence gate."""
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("promotion_gate", ROOT / "scripts/promotion_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

NOW = datetime(2026, 10, 12, 20, 0, tzinfo=timezone.utc)
SHA = "a" * 64
REV = "b" * 40


def evidence():
    base = {
        "schema": 1,
        "project_revision": REV,
        "gpu": {"arch": "gfx1150"},
        "runtime_hashes": {"nvcuda_sha256": SHA},
        "generated_utc": (NOW - timedelta(minutes=3)).isoformat(),
    }
    required = list(gate.DRIVER_REQUIRED)
    cap = {
        **base,
        "selected_tests": required,
        "no_silent_corruption": True,
        "no_hangs": True,
        "no_process_crashes": True,
        "incorrect": 0,
        "timeouts": 0,
        "process_hangs": 0,
        "process_crashes": 0,
        "errors": 0,
        "timeout_retries": 0,
        "results": [
            {"test": name, "status": "pass", "result": {"ok": True},
             "process_exit": 0, "timed_out": False,
             "process_hung_after_result": False,
             "process_crashed_after_result": False}
            for name in required
        ]
    }
    fun = {
        **base, "core_correctness_ok": True, "hard_failures": 0,
        "tests": [
            {"test": name, "status": "pass", "result": {"ok": True},
             "process_exit": 0, "timed_out": False,
             "process_crashed_after_result": False}
            for name in gate.FUNCTIONAL_REQUIRED
        ]
    }
    return cap, fun


def failures(cap, fun):
    return gate.evaluate(cap, fun, driver_sha=SHA, arch="gfx1150",
                         revision=REV, now=NOW)


class TestPromotionGate(unittest.TestCase):
    def test_good_matching_evidence_is_eligible_for_review(self):
        self.assertEqual(failures(*evidence()), [])

    def test_bad_or_missing_driver_hash_fails_closed(self):
        cap, fun = evidence()
        cap["runtime_hashes"]["nvcuda_sha256"] = "0" * 64
        fun["runtime_hashes"] = {}
        self.assertTrue(any("SHA-256" in x for x in failures(cap, fun)))

    def test_missing_gpu_arch_or_revision_fails(self):
        cap, fun = evidence()
        cap["gpu"]["arch"] = "gfx1200"
        fun["project_revision"] = "c" * 40
        self.assertTrue(any("architecture" in x for x in failures(cap, fun)))
        self.assertTrue(any("revision" in x for x in failures(cap, fun)))

    def test_missing_required_sort_scan_fails(self):
        cap, fun = evidence()
        cap["results"] = [x for x in cap["results"] if x["test"] != "cub_sort_scan"]
        cap["selected_tests"] = [x["test"] for x in cap["results"]]
        self.assertTrue(any("cub_sort_scan" in x for x in failures(cap, fun)))

    def test_silent_numerical_failure_overrides_green_summary(self):
        cap, fun = evidence()
        cap["results"][2]["result"]["ok"] = False
        self.assertTrue(failures(cap, fun))

    def test_unsafe_timeout_or_gpu_crash_fails(self):
        cap, fun = evidence()
        cap["results"][0]["process_crashed_after_result"] = True
        cap["timeout_retries"] = 1
        self.assertTrue(any("clean numerical PASS" in x for x in failures(cap, fun)))
        self.assertTrue(any("timeout_retries" in x for x in failures(cap, fun)))

    def test_expired_and_future_reports_fail(self):
        cap, fun = evidence()
        cap["generated_utc"] = (NOW - timedelta(days=9)).isoformat()
        fun["generated_utc"] = (NOW + timedelta(minutes=1)).isoformat()
        self.assertEqual(sum("expired or from the future" in x for x in failures(cap, fun)), 2)

    def test_missing_functional_matmul_fails(self):
        cap, fun = evidence()
        fun["tests"] = [t for t in fun["tests"] if t["test"] != "matmul"]
        self.assertTrue(any("matmul" in x for x in failures(cap, fun)))

    def test_selected_test_spoof_or_duplicate_fails(self):
        cap, fun = evidence()
        cap["selected_tests"] = ["driver_buffer_clear", "driver_buffer_clear"]
        self.assertTrue(any("selected_tests" in x for x in failures(cap, fun)))

    def test_known_bad_capability_detected_even_if_not_required(self):
        cap, fun = evidence()
        cap["results"].append({
            "test": "random_operation", "status": "incorrect",
            "result": {"ok": False}, "process_exit": 2,
        })
        cap["selected_tests"].append("random_operation")
        self.assertTrue(any("random_operation" in x for x in failures(cap, fun)))

    def test_missing_timestamp_rejected(self):
        cap, fun = evidence()
        cap["generated_utc"] = None
        self.assertTrue(any("timestamp" in x for x in failures(cap, fun)))

    def test_cli_ties_report_to_actual_driver_bytes(self):
        cap, fun = evidence()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            driver = root / "nvcuda.dll"
            driver.write_bytes(b"fake driver bytes for deterministic offline test")
            digest = gate.sha256(driver)
            for report in (cap, fun):
                report["runtime_hashes"]["nvcuda_sha256"] = digest
                report["generated_utc"] = datetime.now(timezone.utc).isoformat()
            a = root / "cap.json"
            b = root / "fun.json"
            a.write_text(json.dumps(cap), encoding="utf-8")
            b.write_text(json.dumps(fun), encoding="utf-8")
            from unittest.mock import patch
            import io
            from contextlib import redirect_stdout
            with redirect_stdout(io.StringIO()) as output:
                code = gate.main([
                        "--capabilities", str(a), "--functional", str(b),
                        "--driver", str(driver), "--arch", "gfx1150",
                        "--revision", REV])
            self.assertEqual(code, 0, output.getvalue())
            driver.write_bytes(b"changed driver")
            with redirect_stdout(io.StringIO()) as output:
                code = gate.main([
                        "--capabilities", str(a), "--functional", str(b),
                        "--driver", str(driver), "--arch", "gfx1150",
                        "--revision", REV])
            self.assertEqual(code, 1, output.getvalue())


if __name__ == "__main__":
    unittest.main()

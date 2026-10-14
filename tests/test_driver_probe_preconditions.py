"""Offline regression: PDL capability and CUDA current-context preconditions."""
import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "capability_probe.py"
spec = importlib.util.spec_from_file_location("zluda_capabilities_for_test", MODULE_PATH)
cap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap)


class FakeCUDA:
    def __init__(self, major, minor):
        self.capability = (major, minor)

    def get_device_capability(self, index):
        assert index == 0
        return self.capability


class FakeTorch:
    def __init__(self, major, minor):
        self.cuda = FakeCUDA(major, minor)


class DriverProbePreconditionsTest(unittest.TestCase):
    def test_pdl_is_unsupported_below_sm90_without_loading_driver(self):
        from unittest.mock import patch

        for arch in ((7, 5), (8, 0), (8, 6), (8, 9)):
            with self.subTest(arch=arch):
                with patch.object(cap, "_load_win_dll", side_effect=AssertionError("driver should not load")):
                    with self.assertRaisesRegex(RuntimeError, "PDL unsupported.*below 9.0"):
                        cap.run_driver_pdl_semantics(FakeTorch(*arch))

    def test_pdl_sm90_keeps_real_validation_path(self):
        from unittest.mock import patch

        with patch.object(cap, "_load_win_dll", side_effect=RuntimeError("driver validation path reached")):
            with self.assertRaisesRegex(RuntimeError, "driver validation path reached"):
                cap.run_driver_pdl_semantics(FakeTorch(9, 0))

    def test_async_copy_must_create_current_context_before_allocation(self):
        text = MODULE_PATH.read_text("utf-8")
        start = text.index("def run_driver_memcpy2d_async(torch):")
        end = text.index("RUNNERS: dict", start)
        probe = text[start:end]
        self.assertIn('context_guard = torch.empty(1, device="cuda")', probe)
        self.assertIn('cuda.cuCtxGetCurrent(ctypes.byref(active_context))', probe)
        self.assertIn('if not active_context.value:', probe)
        self.assertLess(probe.index('context_guard = torch.empty'), probe.index('cuMemAlloc_v2(ctypes.byref(src_device)'))


if __name__ == "__main__":
    unittest.main()

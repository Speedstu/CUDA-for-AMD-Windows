import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_zluda_trace.py"
SPEC = importlib.util.spec_from_file_location("summarize_zluda_trace", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class TraceSummaryTests(unittest.TestCase):
    def test_maps_function_handle_to_failing_launch(self):
        text = """[ZLUDA_TRACE] cuModuleLoadData(module: 0x1, image: 0x2) -> CUDA_SUCCESS
[ZLUDA_TRACE] cuModuleGetFunction(hfunc: 0xABC, hmod: 0x1, name: "_Z12bad_kernelv") -> CUDA_SUCCESS
[ZLUDA_TRACE] cuLaunchKernel(f: 0xABC, gridDimX: 1, gridDimY: 1, gridDimZ: 1, blockDimX: 32, blockDimY: 1, blockDimZ: 1, sharedMemBytes: 0, hStream: 0x0, kernelParams: 0x0, extra: NULL) -> CUDA_ERROR_LAUNCH_FAILED
"""
        report = MODULE.parse_trace(text, 10)
        self.assertEqual(report["launch_count"], 1)
        self.assertEqual(report["failing_launch_count"], 1)
        self.assertEqual(report["first_failing_launch"]["function_name"], "_Z12bad_kernelv")
        self.assertEqual(report["first_failing_launch"]["status"], "CUDA_ERROR_LAUNCH_FAILED")

    def test_keeps_successful_last_launch(self):
        text = """[ZLUDA_TRACE] cuModuleGetFunction(hfunc: 0x10, hmod: 0x1, name: "ok_kernel") -> CUDA_SUCCESS
[ZLUDA_TRACE] cuLaunchKernel(f: 0x10, gridDimX: 1, gridDimY: 1, gridDimZ: 1, blockDimX: 1, blockDimY: 1, blockDimZ: 1, sharedMemBytes: 0, hStream: 0x0, kernelParams: 0x0, extra: NULL) -> CUDA_SUCCESS
"""
        report = MODULE.parse_trace(text, 10)
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["last_launch"]["function_name"], "ok_kernel")
        self.assertEqual(report["last_launch"]["status"], "CUDA_SUCCESS")


if __name__ == "__main__":
    unittest.main()

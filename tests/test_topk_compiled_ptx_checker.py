"""Offline tests for the PTX histogram artifact verifier.

Check that the verifier accepts a known-good atom.shared encoding and
rejects both red.shared and a missing compiled histogram.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "patches/llama-b10978/zluda-safe-topk-atom-histogram-candidate.patch"
CHECKER = ROOT / "tests/test_llama_topk_atom_histogram.py"

# Source verifier requires the *patched* pinned llama.cpp top-k.cu source,
# which is not vendored in the repository. We only validate --ptx logic
# here using a temporary synthetic copy of the relevant source fragments.
DUMMY_SOURCE = '''
static __global__ void top_k_radix_histogram(
    float * src) {
#ifdef GGML_CUDA_ZLUDA_ATOM_HISTOGRAM
    unsigned int value;
    unsigned int bin = 0;
    unsigned int address = (unsigned int) __cvta_generic_to_shared(&histogram[bin]);
    asm volatile("atom.shared.add.u32 %0, [%1], 1;" : "=r"(value) : "r"(address) : "memory");
#else
    atomicAdd(&histogram[bin], 1);
#endif
}
static __global__ void top_k_radix_select() {}
'''
GOOD_PTX = '''
.visible .entry _Z21top_k_radix_histogramILi256ELi8EEvPKfPK17top_k_radix_statePiiii() {
    atom.shared.add.u32 %r1, [%r2], 1;
}
'''
BAD_PTX = GOOD_PTX.replace("atom.shared.add.u32", "red.shared.add.u32")
NO_HISTOGRAM = ".visible .entry another_kernel() { ret; }"


class TestCompiledPTXGuard(unittest.TestCase):
    def test_patch_contains_opt_in_instruction(self):
        patch = SOURCE.read_text(encoding="utf-8")
        self.assertIn("GGML_CUDA_ZLUDA_ATOM_HISTOGRAM", patch)
        self.assertIn('asm volatile("atom.shared.add.u32', patch)

    def check_ptx(self, ptx):
        with TemporaryDirectory() as d:
            d = Path(d)
            source = d / "top-k.cu"
            src_ptx = d / "fatbin.ptx"
            source.write_text(DUMMY_SOURCE, encoding="utf-8")
            src_ptx.write_text(ptx, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(CHECKER), "--source", str(source), "--ptx", str(src_ptx)],
                capture_output=True, text=True
            )

    def test_valid_histogram_is_accepted(self):
        result = self.check_ptx(GOOD_PTX)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_red_shared_is_rejected(self):
        result = self.check_ptx(BAD_PTX)
        self.assertNotEqual(result.returncode, 0)

    def test_missing_kernel_is_rejected(self):
        result = self.check_ptx(NO_HISTOGRAM)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()

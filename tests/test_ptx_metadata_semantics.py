"""Source-level guard against silently corrupting CUB host launch policies.

Does not require torch or a GPU. The real CUB probe tests numerical
correctness on hardware, not just this source-level invariant.
"""
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
root = args.root
patch = (root / "patches/zluda-v7-preview10/ptx-version-candidate.patch").read_text("utf-8")
probe = (root / "scripts/capability_probe.py").read_text("utf-8")
assert "+            *pi = func.ptx_version as i32;" not in patch, "Unsafe PTX ISA function-attribute override"
assert "CU_FUNC_ATTRIBUTE_PTX_VERSION" not in patch, "Patch must preserve upstream CUDA function attribute"
assert "and int(ptx_version.value) == target_sm" in probe, "Metadata probe must expect target SM"
assert "and int(ptx_version.value) == expected" in probe, "Fatbin test must expect selected SM"
assert "def run_cub_sort_scan(torch):" in probe, "CUB runtime regression probe missing"
assert '"cub_sort_scan": run_cub_sort_scan' in probe, "CUB regression probe not registered"
print("PASS: CUDA PTX target semantics and CUB correctness gate definitions")
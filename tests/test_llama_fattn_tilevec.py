"""Offline regression for the opt-in non-MMA flash-attention candidate.

The actual source is compiled by the pinned Windows CUDA workflow; this
guard keeps its two critical selection predicates and macro default from
drifting silently. It does NOT claim real AMD GPU correctness.
"""
import argparse
from pathlib import Path
import re

def check(source: Path):
    s = source.read_text(encoding="utf-8")
    begin = s.index("static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(")
    dispatch = s[begin:s.index("return BEST_FATTN_KERNEL_", begin) + 5000]
    conditional = re.search(
        r"#ifdef\s+GGML_CUDA_ZLUDA_FATTN_NO_MMA\s+"
        r".*?const bool zluda_fattn_mma = false;\s+"
        r"#else\s+const bool zluda_fattn_mma = true;\s+"
        r"#endif", dispatch, re.S)
    assert conditional, "Candidate must select non-MMA by macro, with original MMA default"
    assert re.search(r"if\s*\(\s*zluda_fattn_mma\s*&&\s*turing_mma_available\(cc\)", dispatch)
    assert re.search(r"if\s*\(\s*zluda_fattn_mma\s*&&\s*volta_mma_available\(cc\)", dispatch)
    # Do not accidentally disable flash attention itself or make non-MMA
    # kernel selection dependent on unsupported MMA parser instructions.
    assert "BEST_FATTN_KERNEL_VEC" in dispatch
    assert "BEST_FATTN_KERNEL_TILE" in s[begin:begin+15000]
    assert re.search(r"#else\s+const bool zluda_fattn_mma = true;", dispatch)
    print("PASS: opt-in tile/vec source dispatch skips both MMA branches; default path unchanged")

def check_patch(path: Path):
    text=path.read_text(encoding="utf-8")
    for required in (
        "+#ifdef GGML_CUDA_ZLUDA_FATTN_NO_MMA",
        "+    const bool zluda_fattn_mma = false;",
        "+    const bool zluda_fattn_mma = true;",
        "+    if (zluda_fattn_mma && turing_mma_available(cc)",
        "+    if (zluda_fattn_mma && volta_mma_available(cc)",
    ):
        assert required in text, f"Missing patch guard: {required}"
    assert "-    if (turing_mma_available(cc)" in text
    assert "-    if (volta_mma_available(cc)" in text
    print("PASS: pinned flash-attention patch is opt-in and gates both MMA paths")

def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source",type=Path)
    mode.add_argument("--patch",type=Path)
    a=p.parse_args()
    if a.patch:
        check_patch(a.patch)
    else:
        check(a.source)

if __name__=="__main__":
    main()

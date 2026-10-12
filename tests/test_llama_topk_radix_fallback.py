"""No-GPU guard for the opt-in CUB-free TOP_K radix candidate."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
args = parser.parse_args()
text = args.source.read_text("utf-8")
guard = "#if (!defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)) || defined(GGML_CUDA_ZLUDA_TOPK_RADIX)"
assert guard in text, "Radix kernel definitions not compiled in CUDA candidate"
pos = text.index("void ggml_cuda_op_top_k(")
body = text[pos:]
start = body.index("#if defined(GGML_CUDA_ZLUDA_TOPK_RADIX)")
cub = body.index("#elif defined(CUB_TOP_K_AVAILABLE)")
assert start < cub, "Radix path must take precedence over CUB path"
radix = body[start:cub]
assert "top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);" in radix
assert "DeviceTopK::" not in radix and "argsort_f32_i32_cuda_cub" not in radix
assert "static void top_k_radix_cuda(" in text
assert "top_k_radix_histogram" in text and "top_k_radix_gather" in text
print("PASS: opt-in TOP_K dispatch uses existing plain-CUDA radix path without CUB")
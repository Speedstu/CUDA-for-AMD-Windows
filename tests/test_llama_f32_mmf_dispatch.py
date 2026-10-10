"""Generate a native, GPU-free dispatch check from the pinned llama.cpp mmf.cu source.

CI compiles the generated switch twice (with/without the ZLUDA-only macro).
This catches an accidentally widened guard while leaving CUDA kernel tests to the real GPU.
"""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--guarded', action='store_true')
args = p.parse_args()
source = args.source.read_text(encoding='utf-8')
needle = '    switch (type) {\n        case GGML_TYPE_F32:'
assert source.count(needle) == 1, 'Expected precisely one runtime F32 MMF dispatch'
start = source.index(needle)
end_marker = '\n    }'
end = source.index(end_marker, start) + len(end_marker)
switch = source[start:end]
assert '#ifdef GGML_CUDA_ZLUDA_DISABLE_F32_MMF' in switch
assert 'case GGML_TYPE_F16:' in switch and 'case GGML_TYPE_BF16:' in switch

harness = '''
enum ggml_type { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, OTHER };
static bool ampere_mma_available(int) { return true; }
static bool amd_mfma_available(int) { return true; }
static bool volta_mma_available(int) { return true; }
static bool turing_mma_available(int) { return true; }
static bool amd_wmma_available(int) { return true; }
static bool select_f32_mmf(ggml_type type, int cc) {
''' + switch + '''
}
int main() {
    // The ZLUDA safe build must avoid F32 MMF on Ampere-like CC, while
    // F16/BF16 MMF and the unmodified baseline F32 path stay available.
#ifdef GGML_CUDA_ZLUDA_DISABLE_F32_MMF
    if (select_f32_mmf(GGML_TYPE_F32, 86)) return 1;
#else
    if (!select_f32_mmf(GGML_TYPE_F32, 86)) return 2;
#endif
    if (!select_f32_mmf(GGML_TYPE_F16, 86)) return 3;
    if (!select_f32_mmf(GGML_TYPE_BF16, 86)) return 4;
    if (select_f32_mmf(OTHER, 86)) return 5;
    return 0;
}
'''
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(harness, encoding='utf-8')
print('Generated native dispatch test from ' + str(args.source))
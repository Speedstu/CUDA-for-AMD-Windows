"""Generate an MSVC-compilable, GPU-free TOP_K feature-gate regression.

The preprocessor condition is taken verbatim from pinned top-k.cu, not a
reimplementation. Run the emitted file with/without the candidate macro.
"""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
args = p.parse_args()
source = args.source.read_text(encoding='utf-8')
guards = [line for line in source.splitlines()
          if line.startswith('#    if (CCCL_MAJOR_VERSION')]
assert len(guards) == 1, f'Expected one CCCL TOP_K guard, got {guards!r}'
guard = guards[0]
assert guard.lstrip().startswith('#')
assert '!defined(GGML_CUDA_ZLUDA_DISABLE_CUB_TOPK)' in guard
assert '#elif defined(GGML_CUDA_USE_CUB)' in source, 'argsort fallback preprocessor branch missing'
assert 'argsort_f32_i32_cuda_cub(' in source, 'argsort fallback implementation missing'
assert 'DeviceTopK::MaxPairs(' in source, 'original CUB path missing'
assert 'cudaMemcpy2DAsync(' in source, 'TOP_K output transfer missing'

harness = '''
#define GGML_CUDA_USE_CUB 1
#define CCCL_MAJOR_VERSION 3
#define CCCL_MINOR_VERSION 2
''' + guard + '''
#define CUB_TOP_K_AVAILABLE
#endif
int main() {
#if defined(GGML_CUDA_ZLUDA_DISABLE_CUB_TOPK)
#ifdef CUB_TOP_K_AVAILABLE
    return 1;
#else
    return 0;
#endif
#else
#ifdef CUB_TOP_K_AVAILABLE
    return 0;
#else
    return 2;
#endif
#endif
}
'''
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(harness, encoding='utf-8')
print('Generated TOP_K gate regression from pinned CUDA source')
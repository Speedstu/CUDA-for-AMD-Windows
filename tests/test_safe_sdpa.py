import json
import os
import sys
import torch
import torch.nn.functional as F

safe = os.environ.get('CUDAAMD_PYTORCH_SAFE_SDPA') == '1'
backends = {
    'flash': bool(torch.backends.cuda.flash_sdp_enabled()),
    'mem_efficient': bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
    'math': bool(torch.backends.cuda.math_sdp_enabled()),
}
report = {'safe_env': safe, 'backends': backends}
if not safe and '--flags-only' not in sys.argv:
    raise SystemExit('Refusing to run SDPA without the math-only safety guard')
if '--flags-only' not in sys.argv:
    torch.manual_seed(3407)
    q = torch.randn(1, 2, 12, 32)
    k = torch.randn(1, 2, 12, 32)
    v = torch.randn(1, 2, 12, 32)
    ref = F.scaled_dot_product_attention(q, k, v)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device is not available (possible CPU fallback)')
    report['device'] = torch.cuda.get_device_name(0)
    out = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).float().cpu()
    report['max_abs'] = float((ref - out).abs().max().item())
    report['numeric_match'] = bool(torch.allclose(ref, out, atol=5e-3, rtol=5e-3))
    if not report['numeric_match']:
        raise RuntimeError('Math SDPA does not match CPU reference')
print('CUDAAMD_SAFE_SDPA_TEST:' + json.dumps(report, sort_keys=True), flush=True)
if safe and (backends['flash'] or backends['mem_efficient'] or not backends['math']):
    raise SystemExit(2)

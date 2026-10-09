"""Static guard: the high-risk llama smoke must bound host KV cache by default."""
from pathlib import Path
import re

source = (Path(__file__).resolve().parents[1] / 'scripts' / 'run-llama-zluda-safe.ps1').read_text(encoding='utf-8')
assert re.search(r'\[ValidateRange\(128,\s*262144\)\]\[int\]\$ContextSize\s*=\s*4096', source), 'Missing bounded 4096 default'
assert re.search(r"'-c',\s*\[string\]\$ContextSize", source), 'Context not passed through to llama.cpp'
assert "'-nkvo'" in source, 'Safe CPU KV fallback must not be relaxed implicitly'
assert "'--no-op-offload'" in source, 'Safe op offload guard must remain'
assert "'-fa', 'off'" in source, 'Safe FA guard must remain'
assert "CUDA_LAUNCH_BLOCKING = '1'" in source, 'Launch blocking must remain enabled'
print('Safe llama context defaults to 4096, explicit -c is wired, safety flags retained')
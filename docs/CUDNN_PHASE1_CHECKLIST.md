# Deep learning compatibility — Phase 1 acceptance checklist

Goal: make **CUDA-facing deep learning applications on AMD Windows**
work reliably with well-defined correctness evidence, rather than claiming
CUDA/cuDNN compatibility based only on DLL loading or a successful build.

This is an engineering backlog and release gate, not a support promise.
Every entry must have a versioned Windows ABI test, GPU numerical CPU/native
reference, loaded DLL hashes, failure-path tests, and no regression on an
existing supported workload before being called *validated*.

## Phase 1 status (October 18, 2026)

| Component / scope | State | Blocking evidence |
| --- | --- | --- |
| cuDNN v8 legacy 2D convolution forward, backward-data/filter | Validated on gfx1200 reference, September snapshot | More streams/layouts and GPU generations |
| cuDNN v8 transform, set and scale tensor FP32/FP16 on default stream | **New experimental candidate** | Dedicated real-MIOpen GPU test in scripts/test-cudnn-transform-pytorch.py |
| cuDNN tensor descriptors + algorithm heuristics | Partial | Broader shape/dtype/algorithm conformance |
| cuDNN backward bias | **Not implemented** | Recent MIOpen deprecates its equivalent and can return NotImplemented |
| cuDNN activation (sigmoid, ReLU, tanh) fwd/back | **Experimental ABI-verified candidate** | Real AMD GPU FP32/FP16 correctness; other modes and NaN semantics |
| cuDNN softmax accurate+channel fwd/back | **Experimental ABI-verified candidate** | Real AMD GPU FP32/FP16 correctness; additional modes/algorithms |
| cuDNN 2D max pooling forward (nonoverlapping, pad=0) | **Experimental ABI-verified candidate** | Real GPU MIOpen validation, backward, average/padded/overlapping variants |
| cuDNN normalization | **Not broadly implemented in this proxy** | ABI and GPU tests per op; batching/gradient tests |
| cuDNN frontend/backend graph and cuDNN v9 | **Not implemented** | Separate modern ABI design and graph execution |
| Non-default CUDA stream mapping to HIP streams | **Not supported** | Verified ZLUDA stream handle conversion and ordering tests |
| Fused SDPA / FlashAttention | Unsafe modes explicitly refused | Correct AMD kernel or verified fallback with speed testing |
| gfx1150 historical cuDNN/bridge conv2d timeout | Safe no-cuDNN fallback exists | One bounded, opt-in file-only trace on reporter GPU |
| Packaging/runtime coexistence | Reversible staging + deterministic DLL path | Real app matrix, user-facing checks and rollback audits |

## Required engineering sequence

1. **ABI foundation.** Versioned export inventory for cuDNN v8 and v9,
   strict symbol-resolution rules, optional feature detection, safe
   default/no-backend behavior, and reversible install/staging. Refuse
   unsupported operations, dtype/layout mismatches and unsafe streams.
2. **Convolution completeness.** Validate forward/backward input and weight
   gradients, bias, algorithm/workspace selection, groups, dilation,
   large shapes, FP16/BF16/FP32, non-contiguous tensors and stream use.
   Implement missing families using current supported AMD primitives
   rather than dead/deprecated MIOpen entry points.
3. **Additional DNN ops.** Implement and test pooling, activation,
   normalization, softmax, tensor operation, and fused graph families.
4. **cuDNN v9/frontend.** Design separate ABI-compatible interface,
   front-end descriptors and operation graph mapping. Do not pretend
   cuDNN v8 forwarders provide v9 support.
5. **Attention safety/performance.** Replace unsafe NVIDIA-cubin-only
   fused kernels with verified AMD implementations; require numerical
   parity against FP32 math reference under long sequence lengths,
   masks, causal mode and multiple heads before enabling by default.
6. **Device-specific evidence.** Run named PyTorch models, convolution
   training, inference and CUDA-extensions tests on multiple real
   RDNA generations and representative Instinct GPUs. Capture all
   process loaded DLL SHA-256 values and forbid silent wrong outputs.
7. **Release promotion.** Versioned compatibility table with supported,
   unsupported and known-incorrect paths. All exact versions/flags
   reproducible, zero GPU hang/regression in the declared scope, manual
   reviewer signoff. Never overwrite the stable user installation from
   an experimental build.

## Verification commands

Use a **fresh isolated folder**; avoid staging into an active game's or
user's global library directory.

- Offline ABI: `pwsh scripts/test-cudnn-bridge.ps1`
- CUDA bridge PyTorch training matrix (reference staging required):
  `python scripts/test-cudnn-bridge-pytorch.py`
- New transform GPU correctness (reference staging required):
  `python scripts/test-cudnn-transform-pytorch.py --bridge-dll C:\staged\cudnn64_8.dll`
- CI: `.github/workflows/cudnn-bridge.yml` on Windows, plus verify
  and multi-architecture hardware evidence.

## Non-goals for current candidate

No claim of universal CUDA compatibility, no cuDNN v9 frontend, no
working upstream PTX `red.*` parser, no full FlashAttention support,
and no promise that an NVIDIA-only cubin executes without an equivalent
PTX path or a verified AMD implementation.

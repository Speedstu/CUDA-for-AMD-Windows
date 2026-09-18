# Experimental cuDNN v8 -> MIOpen bridge (Windows)

This directory contains an **experimental, narrow compatibility bridge** for the legacy cuDNN v8 convolution API on AMD GPUs under Windows.

It is not a reimplementation of cuDNN and it does not claim full cuDNN compatibility.

## Current validated scope

Reference validation on 2026-09-19:

- GPU: AMD Radeon RX 9060 XT
- architecture: `gfx1200`
- ZLUDA: v7-preview.10 based runtime
- PyTorch: 2.3.0+cu118
- cuDNN ABI expected by PyTorch: 8.7 / `cudnn64_8.dll`
- backend used for GPU validation: MIOpen from TheRock 7.14.0 `gfx120X-all`

Validated CUDA-facing path:

```text
PyTorch / cuDNN v8 legacy API
        |
        v
cudnn64_8.dll compatibility bridge
        |
        v
MIOpen solution API / Immediate execution
        |
        v
HIP / AMD Radeon RX 9060 XT
```

The versioned GPU matrix validates forward convolution, backward-data (input gradient), and backward-filter (weight gradient) against CPU PyTorch references.

| Case | FP32 forward | FP32 dx/dw | FP16 forward | FP16 dx/dw |
|---|---|---|---|---|
| basic 2D convolution | PASS | PASS | PASS | PASS |
| padding | PASS | PASS | PASS | PASS |
| stride | PASS | PASS | PASS | PASS |
| dilation | PASS | PASS | PASS | PASS |
| groups=2 | PASS | PASS | PASS | PASS |

The 2026-09-19 reference run reported zero max absolute error for every forward output, input gradient, and weight gradient in this fixed matrix.

Forward validation covers both direct `aten::cudnn_convolution` and public `torch.nn.functional.conv2d`. Backward validation uses the public autograd path with both input and weight requiring gradients.

A separate public `torch.nn.functional.conv2d(..., bias=...)` training check also passes on the reference stack with CPU-reference output, input-gradient, weight-gradient and bias-gradient comparisons. PyTorch computes that bias gradient outside this bridge; `cudnnConvolutionBackwardBias` itself is still not implemented or claimed.

## Implemented surface

The generated proxy currently overrides 44 exports when built against the PyTorch 2.3.0+cu118 top-level cuDNN v8 DLL. The implemented convolution-related surface includes:

- handle create/destroy
- default-stream get/set
- tensor descriptors
- filter descriptors
- 2D convolution descriptors
- convolution group count
- convolution math-type round-trip
- convolution output shape
- `cudnnGetConvolutionForwardAlgorithmMaxCount`
- `cudnnGetConvolutionForwardAlgorithm_v7`
- `cudnnGetConvolutionForwardWorkspaceSize`
- `cudnnConvolutionForward`
- `cudnnGetConvolutionBackwardDataAlgorithmMaxCount`
- `cudnnGetConvolutionBackwardDataAlgorithm_v7`
- `cudnnGetConvolutionBackwardDataWorkspaceSize`
- `cudnnConvolutionBackwardData`
- `cudnnGetConvolutionBackwardFilterAlgorithmMaxCount`
- `cudnnGetConvolutionBackwardFilterAlgorithm_v7`
- `cudnnGetConvolutionBackwardFilterWorkspaceSize`
- `cudnnConvolutionBackwardFilter`
- `cudnnGetErrorString`

When built against the real PyTorch 2.3.0+cu118 `cudnn64_8.dll`, the proxy preserves the complete observed top-level export surface: 268/268 exports. Unported exports are forwarded to a preserved user-owned `cudnn64_8_nvidia.dll` for ABI preservation; those forwarded APIs are **not** compatibility claims.

The project does **not** redistribute NVIDIA cuDNN binaries or AMD MIOpen binaries.

## Fail-closed policy

Implemented translations reject semantics that have not been validated instead of silently approximating them.

Current intentional restrictions:

- only 2D convolution descriptors are implemented
- only cuDNN cross-correlation mode is implemented
- non-default CUDA streams return `CUDNN_STATUS_NOT_SUPPORTED`
- forward/backward translated execution requires cuDNN scaling `alpha=1`, `beta=0`
- deterministic execution is **not advertised** because the selected MIOpen solution records do not provide a proven equivalent cuDNN determinism guarantee
- unsupported algorithm families are rejected
- `cudnnConvolutionBackwardBias` is not implemented by this bridge
- fused convolution/activation, pooling, normalization, RNN, attention, and cuDNN frontend graph APIs are not claimed

The forward and backward PyTorch validation therefore uses `torch.backends.cudnn.deterministic = False`.

## Exact solver/workspace binding

The bridge does not map a cuDNN algorithm family to an arbitrary MIOpen implementation.

For forward, backward-data, and backward-filter it follows the same rule:

1. ask MIOpen for applicable solution IDs
2. translate only backend algorithm families that have a compatible legacy cuDNN family
3. query the exact workspace requirement for that solution ID
4. return that workspace through the corresponding cuDNN workspace query
5. select the same family again when the execution call arrives
6. execute the selected MIOpen solution through its Immediate API

The three execution paths use:

- `miopenConvolutionForwardImmediate`
- `miopenConvolutionBackwardDataImmediate`
- `miopenConvolutionBackwardWeightsImmediate`

This avoids using one backend solver during the heuristic query and a different solver with a different workspace requirement during execution.

### Backward family mapping

The legacy cuDNN backward algorithm names are not one-to-one with MIOpen's families. The bridge therefore exposes a deliberately small mapping:

- MIOpen GEMM / implicit-GEMM -> cuDNN legacy `ALGO_1`
- MIOpen direct -> cuDNN legacy `ALGO_0`
- backward-data FFT -> cuDNN backward-data FFT
- backward-data Winograd -> cuDNN backward-data Winograd
- backward-weights Winograd -> cuDNN backward-filter Winograd-nonfused

Other legacy algorithm values remain unsupported rather than being guessed.

## Build

The bridge is generated locally from the user's own cuDNN top-level DLL:

```powershell
.\scripts\build-cudnn-bridge.ps1 `
  -OriginalCudnn "C:\path\to\cudnn64_8.dll"
```

The output includes:

- generated `cudnn64_8.dll`
- generated export-definition file
- `cudnn-bridge-build.json` with SHA-256 hashes and export counts

## Reversible staging

```powershell
.\scripts\stage-cudnn-bridge.ps1 `
  -TargetDir "C:\path\to\application\dll-directory" `
  -MiopenDll "C:\path\to\MIOpen.dll"
```

Restore the exact original binary:

```powershell
.\scripts\stage-cudnn-bridge.ps1 `
  -TargetDir "C:\path\to\application\dll-directory" `
  -Restore
```

The staging script records hashes and refuses to overwrite an unexpected modified target.

## Header-free CI self-test

```powershell
.\scripts\test-cudnn-bridge.ps1
```

The CI self-test does not require or download cuDNN or MIOpen. It creates:

- a synthetic cuDNN export-contract DLL
- the generated bridge
- a test-only MIOpen ABI stub
- forward v7 heuristic/workspace/execution checks
- backward-data v7 heuristic/workspace/execution checks
- backward-filter v7 heuristic/workspace/execution checks
- CPU-reference numerical checks for forward, input gradient, and weight gradient
- fail-closed stream/convolution-mode checks
- a staging/restoration hash test

This validates the bridge's ABI plumbing and translation logic without redistributing vendor binaries.

## Real GPU validation

`native/cudnn_bridge/tests/bridge_gpu_smoke.c` provides a header-free forward GPU smoke for a locally supplied MIOpen/HIP runtime.

`scripts/test-cudnn-bridge-pytorch.py` is the broader reference integration matrix. It validates:

- direct legacy cuDNN forward via `aten::cudnn_convolution`
- public `torch.nn.functional.conv2d`
- public PyTorch autograd backward-data
- public PyTorch autograd backward-filter
- FP32 and FP16
- padding, stride, dilation, and grouped convolution

For PyTorch 2.3's tested legacy cuDNN path, the reference run sets:

```text
TORCH_CUDNN_V8_API_DISABLED=1
```

That forces the legacy v7 algorithm-query surface exercised by this bridge rather than implying cuDNN frontend/v8 graph API support.

## Status

This bridge is **experimental**. The validated claim is currently limited to the 2D forward + backward-data + backward-filter cases above on the reference `gfx1200` system.

Other GPUs, cuDNN versions, MIOpen versions, non-default streams, deterministic guarantees, backward-bias, fused APIs, and the cuDNN frontend remain unverified or unsupported until separately implemented and tested.
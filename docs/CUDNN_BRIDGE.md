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

The generated proxy currently overrides 60 exports when built against the PyTorch 2.3.0+cu118 top-level cuDNN v8 DLL. The implemented convolution-related surface includes:

- handle create/destroy
- experimental cudnnTransformTensor (FP32/FP16, 4D, optional MIOpen backend symbol; default stream; real-GPU correctness test pending)
- experimental cudnnSetTensor and cudnnScaleTensor (FP32/FP16 4D, optional backend symbols; default stream; real-GPU validation pending)
- experimental cuDNN v8 activation descriptors, forward and backward for sigmoid, ReLU and tanh, default stream only (real GPU evidence pending)
- experimental cudnnSoftmaxForward and cudnnSoftmaxBackward for ACCURATE+CHANNEL FP32/FP16 4D; optional MIOpen _V2 translations (real GPU evidence pending)
- experimental cuDNN v8 max pooling 2D forward without overlap or padding (MIOpen pooling; real GPU evidence pending)
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
- cudnnTransformTensor is limited to equal-shaped 4D FP32/FP16 tensors with positive strides; it returns NOT_SUPPORTED when miopenTransformTensor is absent, preserving existing convolution support
- only cuDNN cross-correlation mode is implemented
- non-default CUDA streams return `CUDNN_STATUS_NOT_SUPPORTED`
- forward/backward translated execution requires cuDNN scaling `alpha=1`, `beta=0`
- deterministic execution is **not advertised** because the selected MIOpen solution records do not provide a proven equivalent cuDNN determinism guarantee
- unsupported algorithm families are rejected
- `cudnnConvolutionBackwardBias` is not implemented by this bridge
- fused convolution/activation, pooling, normalization, RNN, attention, and cuDNN frontend graph APIs are not claimed

The forward and backward PyTorch validation therefore uses `torch.backends.cudnn.deterministic = False`.

## Phase 1 extension: tensor transforms (2026-10-18)

The cuDNN v8 bridge now implements three narrow, opt-in tensor operations. The first
is cudnnTransformTensor -> miopenTransformTensor. It supports
4D float32 and float16 tensors with matching logical dimensions and
non-overlapping positive strides on the default stream. MIOpen performs
Y = alpha * X + beta * Y. Invalid shapes are rejected, and
unsupported data types, ranks and backend symbols return NOT_SUPPORTED.

The other two APIs, cudnnSetTensor and cudnnScaleTensor, call optional miopenSetTensor and miopenScaleTensor respectively, with the same restricted, non-overlapping 4D FP32/FP16 descriptor policy. Their test uses non-contiguous strides, verifies the filled/scaled values and confirms that untouched padding is preserved.

The ABI self-test uses a synthetic, CPU-side MIOpen implementation to
exercise noncontiguous strides, alpha/beta scaling and output numerics. Overlapping strides are refused to avoid silently corrupting GPU output.
It also tests a backend with no miopenTransformTensor symbol, where
older convolutions must still pass, and a deliberately wrong explicit
CUDA_AMD_MIOPEN_DLL path, which must no longer fall back silently to
an unrelated DLL on Windows PATH. The test verifies that staging and
restoration preserve the original cuDNN DLL hash.

Run the isolated tests without GPU:

    pwsh scripts/test-cudnn-bridge.ps1

For real GPU qualification, stage an intentionally selected matching
ZLUDA/HIP/MIOpen runtime in a separate directory and run:

    python scripts/test-cudnn-transform-pytorch.py --bridge-dll C:\path\to\staged\cudnn64_8.dll

The latter compares FP32/FP16 strided transform, set and scale GPU results against a CPU
reference and fails on any nonzero cuDNN status. **That real GPU test
has not yet passed on a compatible MIOpen installation**, so this
new tensor APIs remain experimental, not production support.

The older cuDNN v8 convolution, backward-data and backward-filter
reference results from September are preserved; none of these new
operations is installed globally or silently promoted to the stable
v6 path.
## Phase 1 extension: cuDNN v8 activations (October 18, 2026)

The optional translation now exposes six cuDNN entry points:
cudnnCreateActivationDescriptor, cudnnDestroyActivationDescriptor,
cudnnSetActivationDescriptor, cudnnGetActivationDescriptor,
cudnnActivationForward and cudnnActivationBackward.

Only sigmoid, ReLU and tanh are mapped to the documented MIOpen
LOGISTIC, RELU and TANH modes. The bridge translates these enum values
rather than assuming NVIDIA and AMD enumerations match.
The supported configuration is 4D FP32/FP16 matching tensors,
non-overlapping positive strides, default CUDA stream, coefficient 0,
and CUDNN_NOT_PROPAGATE_NAN. Other modes/coefficients or propagation
semantics explicitly return NOT_SUPPORTED rather than a silently wrong
kernel. The MIOpen descriptors and forward/backward symbols are optional;
a backend missing them retains the pre-existing convolution functionality.

An offline ABI/numerical test (CPU-backed synthetic MIOpen) covers
sigmoid, ReLU, and tanh forward/backward, descriptor getters/setters,
unsupported modes and omission of the optional activation symbols:
\`pwsh scripts/test-cudnn-bridge.ps1\`.

Real-GPU qualification is **still required** before claiming supported
hardware. On a deliberately staged ZLUDA/HIP/MIOpen runtime with real
MIOpen activation functions, run:

\`python scripts/test-cudnn-activation-pytorch.py --bridge-dll C:\\path\\to\\cudnn64_8.dll\`

That script checks six FP32/FP16 forward/backward cases against CPU
references and includes the SHA-256 of the exact bridge DLL used.
The offline MIOpen stub is never to be reported as a physical GPU test.

## Phase 1 extension: accurate channel softmax (October 18, 2026)

cudnnSoftmaxForward and cudnnSoftmaxBackward now translate the cuDNN
ACCURATE (1) / CHANNEL (1) pair to MIOpen's documented
miopenSoftmaxForward_V2 and miopenSoftmaxBackward_V2 pair.
Only 4D FP32/FP16 tensors with matching logical dimensions,
non-overlapping strides and default CUDA streams are eligible.
Other algorithms/modes return NOT_SUPPORTED instead of silently changing
normalization axes or logarithmic behaviour. The symbols are optional,
so missing MIOpen softmax functions cannot disable the validated
convolution or activation APIs.

The CPU-backed MIOpen-stub smoke suite validates FP32 channel-wise
softmax normalizers, forward and backward analytical gradients, and
safe refusal when symbols or modes are missing.
A separate, deliberately opt-in real GPU probe runs FP32 and FP16
against a CPU PyTorch reference (not yet passed on real MIOpen):

\`python scripts/test-cudnn-softmax-pytorch.py --bridge-dll C:\\staged\\cudnn64_8.dll\`

No stable installer or NVIDIA DLL is replaced by this candidate.

## Optional real GPU validation through the HIP runtime

For an independent GPU check that does not require a CUDA-targeted PyTorch
wheel, \`scripts/test-cudnn-direct-hip-gpu.py\` allocates genuinely
device-resident FP32 tensors with \`hipMalloc\`, runs cuDNN activation
and accurate channel softmax forward/backward against a separately loaded
actual MIOpen backend, copies the results to host with \`hipMemcpy\`, and
compares against a CPU mathematical reference. It records actual
loaded cuDNN/HIP/MIOpen DLL module paths and SHA-256 values.

Example for a correctly extracted, isolated ROCm TheRock distribution:

    python scripts/test-cudnn-direct-hip-gpu.py --bridge-dll H:\\isolated\\cudnn64_8.dll --hip-dll H:\\therock\\bin\\amdhip64_7.dll --miopen-dll H:\\therock\\bin\\MIOpen.dll --report H:\\isolated\\gpu-evidence.json

Run only under a separate short-lived process with a bounded timeout
and no other active GPU workloads. It MUST NOT be executed against the
CPU-only MIOpen stub and does not install system DLLs.
This direct-HIP device-allocation test is **not the same as complete
CUDA-facing PyTorch/ZLUDA application validation**. Hardware results
must be recorded separately from CPU-backed ABI tests.

## Phase 1 extension: 2D max pooling forward (October 18, 2026)

The optional legacy cuDNN v8 pooling API subset includes
cudnnCreatePoolingDescriptor, cudnnDestroyPoolingDescriptor,
cudnnSetPooling2dDescriptor, cudnnGetPooling2dForwardOutputDim,
and cudnnPoolingForward. It maps to MIOpen's documented
miopenCreatePoolingDescriptor, miopenSet2dPoolingDescriptor and
miopenPoolingForward functions.

To avoid silent disagreements in padding/rounding or backward
argmax workspace semantics, the current subset deliberately accepts
only 4D FP32/FP16 MAX pooling with no padding, nonoverlapping
window-size strides, evenly divisible spatial dimensions, and
the default CUDA stream. Average pooling, padding, overlapping
windows, deterministic max claims and backward gradient APIs are
**not** implemented. These paths return NOT_SUPPORTED rather than
producing plausible but incorrect tensors.

The offline ABI regression suite verifies 4x4 to 2x2 max pooling,
alpha/beta scaling, dimension queries, invalid algorithm refusal,
missing MIOpen pooling symbol refusal, unchanged convolution, and
reversible DLL staging.
The opt-in direct HIP/MIOpen GPU test now includes max-pooling FP32
comparison against an exact CPU reference; it must run with real
MIOpen and a device, not the synthetic test stub.

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
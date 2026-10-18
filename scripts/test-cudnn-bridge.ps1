[CmdletBinding()]
param(
    [string]$OriginalCudnn,
    [string]$OutputDir,
    [string]$Gcc,
    [string]$Objdump
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$manifestPath = Join-Path $repo 'native\cudnn_bridge\bridge-manifest.json'
$stubSource = Join-Path $repo 'native\cudnn_bridge\tests\miopen_stub.c'
$smokeSource = Join-Path $repo 'native\cudnn_bridge\tests\bridge_smoke.c'
if (-not $OutputDir) { $OutputDir = Join-Path $repo '.runtime\cudnn-bridge-selftest' }
$OutputDir = [IO.Path]::GetFullPath($OutputDir)

if (-not $Gcc) {
    $cmd = Get-Command gcc.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { $Gcc = $cmd.Path }
}
if (-not $Gcc -or -not (Test-Path $Gcc)) { throw 'MinGW-w64 gcc.exe not found.' }
if (-not $Objdump) {
    $cmd = Get-Command objdump.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { $Objdump = $cmd.Path }
}
if (-not $Objdump -or -not (Test-Path $Objdump)) {
    $candidate = Join-Path (Split-Path $Gcc -Parent) 'objdump.exe'
    if (Test-Path $candidate) { $Objdump = $candidate }
}
if (-not $Objdump -or -not (Test-Path $Objdump)) { throw 'GNU objdump.exe not found.' }

if (Test-Path $OutputDir) { Remove-Item $OutputDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$fakeDir = Join-Path $OutputDir 'fake-original'
New-Item -ItemType Directory -Force -Path $fakeDir | Out-Null

if ($OriginalCudnn) {
    $OriginalCudnn = [IO.Path]::GetFullPath($OriginalCudnn)
    if (-not (Test-Path $OriginalCudnn)) { throw "Original cuDNN not found: $OriginalCudnn" }
    $sourceCudnn = $OriginalCudnn
    $sourceKind = 'real-user-supplied'
} else {
    $fakeC = Join-Path $fakeDir 'fake_cudnn.c'
    $fakeDef = Join-Path $fakeDir 'fake_cudnn.def'
    'int cudnn_bridge_fake_placeholder(void) { return 0; }' | Set-Content -Encoding ascii $fakeC
    $def = New-Object System.Collections.Generic.List[string]
    [void]$def.Add('LIBRARY cudnn64_8')
    [void]$def.Add('EXPORTS')
    foreach ($name in @($manifest.overrides)) {
        [void]$def.Add("    $name=cudnn_bridge_fake_placeholder")
    }
    1..40 | ForEach-Object {
        [void]$def.Add(("    cudnnBridgeSyntheticExport{0:D2}=cudnn_bridge_fake_placeholder" -f $_))
    }
    $def | Set-Content -Encoding ascii $fakeDef
    $sourceCudnn = Join-Path $fakeDir 'cudnn64_8.dll'
    & $Gcc -shared -O2 -s -o $sourceCudnn $fakeC $fakeDef
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $sourceCudnn)) { throw 'Synthetic cuDNN DLL build failed.' }
    $sourceKind = 'synthetic-export-contract'
}

$bridgeOut = Join-Path $OutputDir 'bridge'
$buildArgs = @{
    OriginalCudnn = $sourceCudnn
    OutputDir = $bridgeOut
    Gcc = $Gcc
    Objdump = $Objdump
}
& (Join-Path $PSScriptRoot 'build-cudnn-bridge.ps1') @buildArgs

$bridgeDll = Join-Path $bridgeOut 'cudnn64_8.dll'
$miopenStub = Join-Path $bridgeOut 'MIOpen.dll'
$smokeExe = Join-Path $bridgeOut 'bridge_smoke.exe'
$activationSmoke = Join-Path $bridgeOut 'bridge_activation_smoke.exe'
$activationSource = Join-Path $repo 'native\cudnn_bridge\tests\bridge_activation_smoke.c'
$softmaxSmoke = Join-Path $bridgeOut 'bridge_softmax_smoke.exe'
$softmaxSource = Join-Path $repo 'native\cudnn_bridge\tests\bridge_softmax_smoke.c'
$poolSmoke = Join-Path $bridgeOut 'bridge_pool_smoke.exe'
$poolSource = Join-Path $repo 'native\cudnn_bridge\tests\bridge_pool_smoke.c'
& $Gcc -std=gnu17 -shared -O2 -Wall -Wextra -Werror -s -o $miopenStub $stubSource
if ($LASTEXITCODE -ne 0) { throw 'MIOpen test stub build failed.' }
& $Gcc -std=gnu17 -O2 -Wall -Wextra -Werror -s -o $smokeExe $smokeSource -lm
if ($LASTEXITCODE -ne 0) { throw 'cuDNN bridge smoke executable build failed.' }
& $Gcc -std=gnu17 -O2 -Wall -Wextra -Werror -s -o $activationSmoke $activationSource -lm
if ($LASTEXITCODE -ne 0) { throw 'cuDNN activation ABI smoke executable build failed.' }
& $Gcc -std=gnu17 -O2 -Wall -Wextra -Werror -s -o $softmaxSmoke $softmaxSource -lm
if ($LASTEXITCODE -ne 0) { throw 'cuDNN softmax ABI executable compilation failed.' }
& $Gcc -std=gnu17 -O2 -Wall -Wextra -Werror -s -o $poolSmoke $poolSource -lm
if ($LASTEXITCODE -ne 0) { throw 'cuDNN max pooling ABI test compilation failed.' }

Copy-Item $sourceCudnn (Join-Path $bridgeOut 'cudnn64_8_nvidia.dll') -Force
$previousMiopen = $env:CUDA_AMD_MIOPEN_DLL
try {
    $env:CUDA_AMD_MIOPEN_DLL = $miopenStub
    & $smokeExe $bridgeDll
    if ($LASTEXITCODE -ne 0) { throw "cuDNN bridge ABI/correctness smoke failed: exit $LASTEXITCODE" }
    & $activationSmoke $bridgeDll
    if ($LASTEXITCODE -ne 0) { throw "cuDNN activation forward/backward test failed: exit $LASTEXITCODE" }
    & $softmaxSmoke $bridgeDll
    if ($LASTEXITCODE -ne 0) { throw 'cuDNN softmax forward/backward numerical check failed.' }
    & $poolSmoke $bridgeDll
    if ($LASTEXITCODE -ne 0) { throw 'cuDNN 2D max pooling numerical check failed.' }
    $stubNoPool = Join-Path $bridgeOut 'MIOpen-without-pooling.dll'
    & $Gcc -std=gnu17 -shared -O2 -Wall -Wextra -Werror -DCUDAAMD_STUB_NO_POOLING -s -o $stubNoPool $stubSource -lm
    if ($LASTEXITCODE -ne 0) { throw 'No-pooling MIOpen stub compilation failed.' }
    $env:CUDA_AMD_MIOPEN_DLL = $stubNoPool
    & $poolSmoke $bridgeDll '--without-pooling'
    if ($LASTEXITCODE -ne 0) { throw 'Missing optional max pooling symbol was not safely refused.' }
    $env:CUDA_AMD_MIOPEN_DLL = $miopenStub
    $stubNoSoftmax = Join-Path $bridgeOut 'MIOpen-without-softmax.dll'
    & $Gcc -std=gnu17 -shared -O2 -Wall -Wextra -Werror -DCUDAAMD_STUB_NO_SOFTMAX -s -o $stubNoSoftmax $stubSource -lm
    if ($LASTEXITCODE -ne 0) { throw 'Missing-softmax MIOpen stub compilation failed.' }
    $env:CUDA_AMD_MIOPEN_DLL = $stubNoSoftmax
    & $softmaxSmoke $bridgeDll '--without-softmax'
    if ($LASTEXITCODE -ne 0) { throw 'Missing optional softmax symbol failed to refuse safely.' }
    $env:CUDA_AMD_MIOPEN_DLL = $miopenStub
    $stubNoActivation = Join-Path $bridgeOut 'MIOpen-without-activation.dll'
    & $Gcc -std=gnu17 -shared -O2 -Wall -Wextra -Werror -DCUDAAMD_STUB_NO_ACTIVATION -s -o $stubNoActivation $stubSource -lm
    if ($LASTEXITCODE -ne 0) { throw 'Old MIOpen no-activation stub build failed.' }
    $env:CUDA_AMD_MIOPEN_DLL = $stubNoActivation
    & $activationSmoke $bridgeDll '--without-activation'
    if ($LASTEXITCODE -ne 0) { throw 'Optional activation symbols fail-closed probe failed.' }
    $env:CUDA_AMD_MIOPEN_DLL = $miopenStub
    # Optional tensor transform must not be mandatory for other convolutions.
    $stubNoTransform = Join-Path $bridgeOut 'MIOpen-without-transform.dll'
    & $Gcc -std=gnu17 -shared -O2 -Wall -Wextra -Werror -DCUDAAMD_STUB_NO_TRANSFORM -s -o $stubNoTransform $stubSource
    if ($LASTEXITCODE -ne 0) { throw 'Old-MIOpen backend stub build failed.' }
    $env:CUDA_AMD_MIOPEN_DLL = $stubNoTransform
    & $smokeExe $bridgeDll '--without-transform'
    if ($LASTEXITCODE -ne 0) { throw 'Missing optional symbol regressed convolution support.' }

    # A wrong explicitly configured backend must not silently load from PATH.
    $env:CUDA_AMD_MIOPEN_DLL = Join-Path $bridgeOut 'intentionally-missing-miopen.dll'
    & $smokeExe $bridgeDll '--bad-backend'
    if ($LASTEXITCODE -ne 0) { throw 'Bad explicit MIOpen path did not fail closed.' }
} finally {
    $env:CUDA_AMD_MIOPEN_DLL = $previousMiopen
}

$stageDir = Join-Path $OutputDir 'stage-contract'
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null
Copy-Item $sourceCudnn (Join-Path $stageDir 'cudnn64_8.dll') -Force
$originalHash = (Get-FileHash (Join-Path $stageDir 'cudnn64_8.dll') -Algorithm SHA256).Hash
$stageArgs = @{
    TargetDir = $stageDir
    BridgeDll = $bridgeDll
    MiopenDll = $miopenStub
}
& (Join-Path $PSScriptRoot 'stage-cudnn-bridge.ps1') @stageArgs
if (-not (Test-Path (Join-Path $stageDir 'cudnn64_8_nvidia.dll'))) { throw 'Staging did not preserve original cuDNN.' }
if (-not (Test-Path (Join-Path $stageDir '.cudaamd-cudnn-bridge.json'))) { throw 'Staging state file missing.' }
& (Join-Path $PSScriptRoot 'stage-cudnn-bridge.ps1') -TargetDir $stageDir -Restore
$restoredHash = (Get-FileHash (Join-Path $stageDir 'cudnn64_8.dll') -Algorithm SHA256).Hash
if ($restoredHash -ne $originalHash) { throw 'Staging restore did not reproduce the exact original DLL hash.' }

$buildMetadata = Get-Content (Join-Path $bridgeOut 'cudnn-bridge-build.json') -Raw | ConvertFrom-Json
$report = [ordered]@{
    schema = 1
    source_kind = $sourceKind
    source_sha256 = (Get-FileHash $sourceCudnn -Algorithm SHA256).Hash
    bridge_sha256 = (Get-FileHash $bridgeDll -Algorithm SHA256).Hash
    source_exports = [int]$buildMetadata.original_export_count
    bridge_exports = [int]$buildMetadata.proxy_export_count
    amd_overrides = [int]$buildMetadata.amd_native_override_count
    abi_smoke = 'pass'
    cpu_reference_forward_cross_correlation = 'pass'
    cpu_reference_backward_data = 'pass'
    cpu_reference_backward_filter = 'pass'
    transform_tensor_strided_fp32 = 'pass'
    set_scale_tensor_strided_fp32 = 'pass'
    activation_sigmoid_relu_tanh_fp32 = 'pass'
    activation_backward_cpu_reference = 'pass'
    activation_without_backend_symbol = 'safe-refusal'
    softmax_channel_accurate_forward_backward_fp32 = 'pass'
    softmax_unimplemented_modes_and_backend = 'safe-refusal'
    max_pooling_2d_fp32 = 'pass'
    pooling_unsupported_modes_or_backend = 'safe-refusal'
    transform_tensor_missing_backend_symbol = 'safe-refusal'
    invalid_explicit_miopen_path = 'safe-refusal'
    v7_forward_algorithm_query = 'pass'
    v7_backward_data_algorithm_query = 'pass'
    v7_backward_filter_algorithm_query = 'pass'
    nondefault_stream_policy = 'safe-refusal'
    true_convolution_mode_policy = 'safe-refusal'
    stage_restore_hash = 'pass'
}
$reportPath = Join-Path $OutputDir 'cudnn-bridge-selftest.json'
$report | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $reportPath

Write-Host ''
Write-Host 'cuDNN bridge self-test PASS'
Write-Host "Source kind : $sourceKind"
Write-Host "Exports     : $($report.source_exports)/$($report.bridge_exports)"
Write-Host "Overrides   : $($report.amd_overrides)"
Write-Host "Report      : $reportPath"
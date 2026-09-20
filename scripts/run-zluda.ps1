[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Program,
    [string[]]$ProgramArgs = @(),
    [string]$RuntimeRoot,
    [switch]$NoStage,
    [switch]$PyTorchSafeSDPA,
    [switch]$AllowUnsafeFusedSDPA,
    [switch]$AllowUnsafeCudnnConv
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$Program = (Resolve-Path $Program).Path
$targetDir = Split-Path $Program -Parent
$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
if (-not (Test-Path $configPath)) { throw 'Missing runtime-config.json. Run scripts/setup.ps1 first.' }
$config = Get-Content $configPath -Raw | ConvertFrom-Json

function Normalize-PathEntry([string]$Value) {
    if (-not $Value) { return $null }
    try { return [System.IO.Path]::GetFullPath($Value).TrimEnd('\') }
    catch { return $Value.Trim().TrimEnd('\') }
}

if (-not $NoStage) {
    & (Join-Path $PSScriptRoot 'stage-runtime.ps1') -TargetDir $targetDir -RuntimeRoot $RuntimeRoot
}

$zluda = $config.zluda_root
$hip = $config.hip_root
$libtorch = $config.libtorch_root
$env:ZLUDA_CC = if ($config.zluda_cc) { $config.zluda_cc } else { '8.6' }
$env:TORCH_ALLOW_TF32_CUBLAS_OVERRIDE = '1'
if ($config.gpu -and $null -ne $config.gpu.index) {
    # Avoid double-remapping the selected device through both HIP and ROCR
    # visibility variables on mixed iGPU/dGPU systems.
    $env:HIP_VISIBLE_DEVICES = [string]$config.gpu.index
    Remove-Item Env:ROCR_VISIBLE_DEVICES -ErrorAction SilentlyContinue
}

$latestChannel = [bool](
    $config.upstream -and
    [string]$config.upstream.zluda_channel -eq 'latest'
)
$programLeaf = [System.IO.Path]::GetFileName($Program)
$pythonProgram = [bool]($programLeaf -match '^python(?:w|[0-9.]*)?\.exe$')
$gpuArch = if ($config.gpu -and $config.gpu.arch) { [string]$config.gpu.arch } else { '' }

# Detect an explicitly staged cuDNN -> MIOpen bridge. If it has a concrete
# MIOpen backend, gfx1150 does not need the no-cuDNN safety fallback.
$bridgeMiopenReady = [bool]($env:CUDA_AMD_MIOPEN_DLL -and (Test-Path $env:CUDA_AMD_MIOPEN_DLL))
$bridgeStateCandidates = @(
    (Join-Path $targetDir '.cudaamd-cudnn-bridge.json')
)
if ($pythonProgram -and ([System.IO.Path]::GetFileName($targetDir) -ieq 'Scripts')) {
    $venvRoot = Split-Path $targetDir -Parent
    $bridgeStateCandidates += (Join-Path $venvRoot 'Lib\site-packages\torch\lib\.cudaamd-cudnn-bridge.json')
}
foreach ($statePath in $bridgeStateCandidates) {
    if ($bridgeMiopenReady -or -not (Test-Path $statePath)) { continue }
    try {
        $state = Get-Content $statePath -Raw | ConvertFrom-Json
        if ($state.miopen_dll -and (Test-Path ([string]$state.miopen_dll))) {
            $env:CUDA_AMD_MIOPEN_DLL = [string]$state.miopen_dll
            $bridgeMiopenReady = $true
        }
    } catch {}
}

# The unpatched upstream latest channel can enter unsafe fused SDPA paths.
$autoSafeSDPA = [bool]($latestChannel -and $pythonProgram -and -not $AllowUnsafeFusedSDPA)
$useSafeSDPA = [bool]($PyTorchSafeSDPA -or $autoSafeSDPA)

# Community gfx1150 testing shows the legacy cuDNN/ZLUDA convolution path can
# hang indefinitely. Until a bridge-backed path is staged and validated, use
# PyTorch's non-cuDNN convolution fallback instead of letting that hang.
$autoSafeConv2D = [bool](
    $pythonProgram -and
    $gpuArch -eq 'gfx1150' -and
    -not $bridgeMiopenReady -and
    -not $AllowUnsafeCudnnConv
)
$useSafeConv2D = $autoSafeConv2D

$safeSite = Join-Path $PSScriptRoot 'pytorch-safe-site'
$safeSiteKey = Normalize-PathEntry $safeSite
$useSafeSite = [bool]($useSafeSDPA -or $useSafeConv2D)

if ($useSafeSDPA) { $env:CUDAAMD_PYTORCH_SAFE_SDPA = '1' }
else { Remove-Item Env:CUDAAMD_PYTORCH_SAFE_SDPA -ErrorAction SilentlyContinue }

if ($useSafeConv2D) { $env:CUDAAMD_PYTORCH_SAFE_CONV2D = '1' }
else { Remove-Item Env:CUDAAMD_PYTORCH_SAFE_CONV2D -ErrorAction SilentlyContinue }

# Always normalize this project's hook out of inherited PYTHONPATH first, then
# add exactly one copy back when at least one safety policy requires it.
$keptPythonPath = @(
    foreach ($part in ($env:PYTHONPATH -split ';')) {
        if (-not $part) { continue }
        $partKey = Normalize-PathEntry $part
        if (-not [string]::Equals($partKey, $safeSiteKey, [System.StringComparison]::OrdinalIgnoreCase)) {
            $part
        }
    }
)
if ($useSafeSite) {
    if (-not (Test-Path (Join-Path $safeSite 'sitecustomize.py'))) {
        throw "Missing PyTorch safe-site hook: $safeSite"
    }
    $env:PYTHONPATH = $safeSite + $(if ($keptPythonPath.Count -gt 0) { ';' + ($keptPythonPath -join ';') } else { '' })
} elseif ($keptPythonPath.Count -gt 0) {
    $env:PYTHONPATH = $keptPythonPath -join ';'
} else {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
}

if ($hip) {
    $env:HIP_PATH = $hip
    $env:ROCBLAS_TENSILE_LIBPATH = Join-Path $hip 'bin\rocblas\library'
    $hipblasltLib = Join-Path $hip 'bin\hipblaslt\library'
    if ($config.gpu -and $config.gpu.arch) {
        $archLib = Join-Path $hipblasltLib ([string]$config.gpu.arch)
        if (Test-Path $archLib) { $hipblasltLib = $archLib }
    }
    $env:HIPBLASLT_TENSILE_LIBPATH = $hipblasltLib
}

$parts = @($targetDir, $zluda)
if ($hip) { $parts += (Join-Path $hip 'bin') }
if ($libtorch) { $parts += (Join-Path $libtorch 'lib'); $parts += (Join-Path $libtorch 'bin') }
$env:PATH = (($parts | Where-Object { $_ -and (Test-Path $_) }) -join ';') + ';' + $env:PATH

$launcher = Join-Path $zluda 'zluda.exe'
if (-not (Test-Path $launcher)) { throw "Missing ZLUDA launcher: $launcher" }
Write-Host "[run] ZLUDA_CC=$env:ZLUDA_CC"
if ($env:HIP_VISIBLE_DEVICES) { Write-Host "[run] HIP_VISIBLE_DEVICES=$env:HIP_VISIBLE_DEVICES" }
if ($useSafeSDPA) {
    $reason = if ($PyTorchSafeSDPA) { 'explicit' } else { 'automatic for unpatched latest channel' }
    Write-Host "[run] PyTorch safe SDPA: flash=off memory-efficient=off math=on ($reason)"
}
if ($useSafeConv2D) {
    Write-Host '[run] PyTorch safe conv2d: cuDNN=off (automatic gfx1150 hang guard)'
} elseif ($gpuArch -eq 'gfx1150' -and $bridgeMiopenReady) {
    Write-Host '[run] PyTorch conv2d: staged cuDNN->MIOpen bridge detected'
}
Write-Host "[run] $launcher -- $Program $($ProgramArgs -join ' ')"
& $launcher '--' $Program @ProgramArgs
exit $LASTEXITCODE

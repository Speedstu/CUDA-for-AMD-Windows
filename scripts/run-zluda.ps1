[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Program,
    [string[]]$ProgramArgs = @(),
    [string]$RuntimeRoot,
    [switch]$NoStage,
    [switch]$PyTorchSafeSDPA
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

if (-not $NoStage) {
    & (Join-Path $PSScriptRoot 'stage-runtime.ps1') -TargetDir $targetDir -RuntimeRoot $RuntimeRoot
}

$zluda = $config.zluda_root
$hip = $config.hip_root
$libtorch = $config.libtorch_root
$env:ZLUDA_CC = if ($config.zluda_cc) { $config.zluda_cc } else { '8.6' }
$env:TORCH_ALLOW_TF32_CUBLAS_OVERRIDE = '1'
if ($PyTorchSafeSDPA) {
    $safeSite = Join-Path $PSScriptRoot 'pytorch-safe-site'
    if (-not (Test-Path (Join-Path $safeSite 'sitecustomize.py'))) {
        throw "Missing PyTorch safe-site hook: $safeSite"
    }
    $env:CUDAAMD_PYTORCH_SAFE_SDPA = '1'
    $env:PYTHONPATH = $safeSite + $(if ($env:PYTHONPATH) { ';' + $env:PYTHONPATH } else { '' })
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
if ($PyTorchSafeSDPA) { Write-Host '[run] PyTorch safe SDPA: flash=off memory-efficient=off math=on' }
Write-Host "[run] $launcher -- $Program $($ProgramArgs -join ' ')"
& $launcher '--' $Program @ProgramArgs
exit $LASTEXITCODE

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Program,
    [string[]]$ProgramArgs = @(),
    [string]$RuntimeRoot,
    [switch]$NoStage
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
if ($config.gpu -and $null -ne $config.gpu.hip_visible_device -and [string]$config.gpu.hip_visible_device -ne '') {
    $env:HIP_VISIBLE_DEVICES = [string]$config.gpu.hip_visible_device
    $env:ROCR_VISIBLE_DEVICES = [string]$config.gpu.hip_visible_device
    Write-Host "[run] HIP_VISIBLE_DEVICES=$env:HIP_VISIBLE_DEVICES"
}
if ($hip) {
    $env:HIP_PATH = $hip
    $env:ROCBLAS_TENSILE_LIBPATH = Join-Path $hip 'bin\rocblas\library'
    $env:HIPBLASLT_TENSILE_LIBPATH = Join-Path $hip 'bin\hipblaslt\library'
}

$parts = @($targetDir, $zluda)
if ($hip) { $parts += (Join-Path $hip 'bin') }
if ($libtorch) { $parts += (Join-Path $libtorch 'lib'); $parts += (Join-Path $libtorch 'bin') }
$env:PATH = (($parts | Where-Object { $_ -and (Test-Path $_) }) -join ';') + ';' + $env:PATH

$launcher = Join-Path $zluda 'zluda.exe'
if (-not (Test-Path $launcher)) { throw "Missing ZLUDA launcher: $launcher" }
Write-Host "[run] ZLUDA_CC=$env:ZLUDA_CC"
Write-Host "[run] $launcher -- $Program $($ProgramArgs -join ' ')"
& $launcher '--' $Program @ProgramArgs
exit $LASTEXITCODE

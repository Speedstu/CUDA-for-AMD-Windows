[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$TargetDir,
    [string]$RuntimeRoot
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
if (-not (Test-Path $configPath)) { throw "Missing runtime config. Run scripts/setup.ps1 first: $configPath" }
$config = Get-Content $configPath -Raw | ConvertFrom-Json
$zluda = $config.zluda_root
$overlay = $config.custom_overlay_root
if (-not $zluda -or -not (Test-Path (Join-Path $zluda 'zluda.exe'))) { throw 'Configured ZLUDA core is missing.' }
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

$coreFiles = @(
    'nvcuda.dll','nvml.dll','zluda_redirect.dll','zluda_precompile.exe','nvcudart_hybrid64.dll',
    'cufft64_10.dll','cufft64_11.dll','cufft64_12.dll',
    'cusparse64_10.dll','cusparse64_11.dll','cusparse64_12.dll'
)
foreach ($name in $coreFiles) {
    $src = Join-Path $zluda $name
    if (Test-Path $src) { Copy-Item $src (Join-Path $TargetDir $name) -Force }
}

# Preserve upstream ZLUDA BLAS implementations under fallback names before overlaying custom wrappers.
foreach ($ver in @('11','12','13')) {
    foreach ($base in @('cublas64','cublasLt64')) {
        $name = "${base}_${ver}.dll"
        $src = Join-Path $zluda $name
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $TargetDir $name) -Force
            Copy-Item $src (Join-Path $TargetDir "${base}_${ver}_zluda.dll") -Force
        }
    }
}

if ($overlay -and (Test-Path $overlay)) {
    foreach ($name in @(
        'amdhip64.dll','amdhip64_7.dll','rocm_kpack.dll',
        'cublas64_11.dll','cublas64_12.dll','cublas64_13.dll',
        'cublasLt64_11.dll','cublasLt64_12.dll','cublasLt64_13.dll'
    )) {
        $src = Join-Path $overlay $name
        if (Test-Path $src) { Copy-Item $src (Join-Path $TargetDir $name) -Force }
    }
}

Write-Host "[stage] Runtime staged into: $TargetDir"
Write-Host "[stage] Core: $zluda"
if ($overlay) { Write-Host "[stage] Overlay: $overlay" }

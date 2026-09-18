[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OriginalCusolver,
    [string]$OutputDir,
    [string]$Gcc,
    [string]$Objdump
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$source = Join-Path $repo 'native\cusolver_proxy\cusolver_proxy.c'
if (-not (Test-Path $source)) { throw "Missing proxy source: $source" }

$OriginalCusolver = [IO.Path]::GetFullPath($OriginalCusolver)
if (-not (Test-Path $OriginalCusolver)) { throw "Original cuSOLVER DLL not found: $OriginalCusolver" }
if (-not $OutputDir) { $OutputDir = Join-Path $repo '.runtime\cusolver-proxy' }
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $Gcc) {
    $cmd = Get-Command gcc.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { $Gcc = $cmd.Source }
}
if (-not $Gcc -or -not (Test-Path $Gcc)) {
    throw 'MinGW-w64 gcc.exe was not found. Pass -Gcc <path-to-gcc.exe>.'
}

if (-not $Objdump) {
    $cmd = Get-Command objdump.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { $Objdump = $cmd.Source }
}
if (-not $Objdump -or -not (Test-Path $Objdump)) {
    $candidate = Join-Path (Split-Path $Gcc -Parent) 'objdump.exe'
    if (Test-Path $candidate) { $Objdump = $candidate }
}
if (-not $Objdump -or -not (Test-Path $Objdump)) {
    throw 'GNU objdump.exe was not found. It is used to preserve the original DLL export surface.'
}

$overrideNames = @(
    'cusolverDnCreate', 'cusolverDnDestroy', 'cusolverDnSetStream',
    'cusolverDnCreateParams', 'cusolverDnDestroyParams', 'cusolverDnSetAdvOptions',
    'cusolverDnXpotrf_bufferSize', 'cusolverDnXpotrf', 'cusolverDnXpotrs',
    'cusolverDnXgeqrf_bufferSize', 'cusolverDnXgeqrf',
    'cusolverDnXsyevd_bufferSize', 'cusolverDnXsyevd',
    'cusolverDnSgeqrf_bufferSize', 'cusolverDnDgeqrf_bufferSize', 'cusolverDnCgeqrf_bufferSize', 'cusolverDnZgeqrf_bufferSize',
    'cusolverDnSgeqrf', 'cusolverDnDgeqrf', 'cusolverDnCgeqrf', 'cusolverDnZgeqrf',
    'cusolverDnSorgqr_bufferSize', 'cusolverDnDorgqr_bufferSize', 'cusolverDnCungqr_bufferSize', 'cusolverDnZungqr_bufferSize',
    'cusolverDnSorgqr', 'cusolverDnDorgqr', 'cusolverDnCungqr', 'cusolverDnZungqr',
    'cusolverDnSormqr_bufferSize', 'cusolverDnDormqr_bufferSize', 'cusolverDnCunmqr_bufferSize', 'cusolverDnZunmqr_bufferSize',
    'cusolverDnSormqr', 'cusolverDnDormqr', 'cusolverDnCunmqr', 'cusolverDnZunmqr',
    'cusolverDnSgetrf_bufferSize', 'cusolverDnDgetrf_bufferSize', 'cusolverDnCgetrf_bufferSize', 'cusolverDnZgetrf_bufferSize',
    'cusolverDnSgetrf', 'cusolverDnDgetrf', 'cusolverDnCgetrf', 'cusolverDnZgetrf',
    'cusolverDnSgetrs', 'cusolverDnDgetrs', 'cusolverDnCgetrs', 'cusolverDnZgetrs',
    'cusolverDnSpotrf_bufferSize', 'cusolverDnDpotrf_bufferSize', 'cusolverDnCpotrf_bufferSize', 'cusolverDnZpotrf_bufferSize',
    'cusolverDnSpotrf', 'cusolverDnDpotrf', 'cusolverDnCpotrf', 'cusolverDnZpotrf',
    'cusolverDnSpotrfBatched', 'cusolverDnDpotrfBatched', 'cusolverDnCpotrfBatched', 'cusolverDnZpotrfBatched',
    'cusolverDnSpotri_bufferSize', 'cusolverDnDpotri_bufferSize', 'cusolverDnCpotri_bufferSize', 'cusolverDnZpotri_bufferSize',
    'cusolverDnSpotri', 'cusolverDnDpotri', 'cusolverDnCpotri', 'cusolverDnZpotri',
    'cusolverDnSpotrs', 'cusolverDnDpotrs', 'cusolverDnCpotrs', 'cusolverDnZpotrs',
    'cusolverDnSpotrsBatched', 'cusolverDnDpotrsBatched', 'cusolverDnCpotrsBatched', 'cusolverDnZpotrsBatched',
    'cusolverDnCreateGesvdjInfo', 'cusolverDnDestroyGesvdjInfo',
    'cusolverDnXgesvdjSetMaxSweeps', 'cusolverDnXgesvdjSetSortEig', 'cusolverDnXgesvdjSetTolerance',
    'cusolverDnXgesvdjGetResidual', 'cusolverDnXgesvdjGetSweeps',
    'cusolverDnSsyevd_bufferSize', 'cusolverDnDsyevd_bufferSize', 'cusolverDnCheevd_bufferSize', 'cusolverDnZheevd_bufferSize',
    'cusolverDnSsyevd', 'cusolverDnDsyevd', 'cusolverDnCheevd', 'cusolverDnZheevd',
    'cusolverDnSgesvdj_bufferSize', 'cusolverDnDgesvdj_bufferSize', 'cusolverDnCgesvdj_bufferSize', 'cusolverDnZgesvdj_bufferSize',
    'cusolverDnSgesvdj', 'cusolverDnDgesvdj', 'cusolverDnCgesvdj', 'cusolverDnZgesvdj',
    'cusolverDnSgesvdjBatched_bufferSize', 'cusolverDnDgesvdjBatched_bufferSize', 'cusolverDnCgesvdjBatched_bufferSize', 'cusolverDnZgesvdjBatched_bufferSize',
    'cusolverDnSgesvdjBatched', 'cusolverDnDgesvdjBatched', 'cusolverDnCgesvdjBatched', 'cusolverDnZgesvdjBatched',
    'cusolverDnCreateSyevjInfo', 'cusolverDnDestroySyevjInfo',
    'cusolverDnXsyevjSetMaxSweeps', 'cusolverDnXsyevjSetSortEig', 'cusolverDnXsyevjSetTolerance',
    'cusolverDnXsyevjGetResidual', 'cusolverDnXsyevjGetSweeps',
    'cusolverDnSsyevj_bufferSize', 'cusolverDnDsyevj_bufferSize', 'cusolverDnCheevj_bufferSize', 'cusolverDnZheevj_bufferSize',
    'cusolverDnSsyevj', 'cusolverDnDsyevj', 'cusolverDnCheevj', 'cusolverDnZheevj',
    'cusolverDnSsyevjBatched_bufferSize', 'cusolverDnDsyevjBatched_bufferSize', 'cusolverDnCheevjBatched_bufferSize', 'cusolverDnZheevjBatched_bufferSize',
    'cusolverDnSsyevjBatched', 'cusolverDnDsyevjBatched', 'cusolverDnCheevjBatched', 'cusolverDnZheevjBatched'
)
$overrideSet = @{}
foreach ($name in $overrideNames) { $overrideSet[$name] = $true }

$objdumpText = @(& $Objdump -p $OriginalCusolver 2>&1 | ForEach-Object { [string]$_ })
if ($LASTEXITCODE -ne 0) { throw "objdump failed for $OriginalCusolver" }
$exports = New-Object System.Collections.Generic.List[string]
foreach ($line in $objdumpText) {
    if ($line -match '^\s*\[\s*\d+\]\s+\+base\[.*?\]\s+[0-9A-Fa-f]+\s+(\S+)\s*$') {
        [void]$exports.Add($Matches[1])
    }
}
$exports = @($exports | Sort-Object -Unique)
if ($exports.Count -lt 100) {
    throw "Only $($exports.Count) exports were parsed. Refusing to build a proxy with an incomplete export surface."
}

$missingOverrides = @($overrideNames | Where-Object { $_ -notin $exports })
if ($missingOverrides.Count -gt 0) {
    Write-Warning ("The original DLL does not export some hipSOLVER overrides: " + ($missingOverrides -join ', '))
}
$allExports = @($exports + $missingOverrides | Sort-Object -Unique)

$defPath = Join-Path $OutputDir 'cusolver_proxy.generated.def'
$defLines = New-Object System.Collections.Generic.List[string]
[void]$defLines.Add('LIBRARY cusolver64_11')
[void]$defLines.Add('EXPORTS')
foreach ($name in $allExports) {
    if ($overrideSet.ContainsKey($name)) {
        [void]$defLines.Add("    $name")
    } else {
        [void]$defLines.Add("    $name=cusolver64_11_nvidia.$name")
    }
}
$defLines | Set-Content -Encoding ascii $defPath

$outDll = Join-Path $OutputDir 'cusolver64_11.dll'
& $Gcc -shared -O2 -s -o $outDll $source $defPath
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $outDll)) {
    throw 'cuSOLVER proxy compilation failed.'
}

$proxyDump = @(& $Objdump -p $outDll 2>&1 | ForEach-Object { [string]$_ })
$proxyExports = @($proxyDump | ForEach-Object {
    if ($_ -match '^\s*\[\s*\d+\]\s+\+base\[.*?\]\s+[0-9A-Fa-f]+\s+(\S+)\s*$') { $Matches[1] }
} | Where-Object { $_ } | Sort-Object -Unique)

$missingFromProxy = @($allExports | Where-Object { $_ -notin $proxyExports })
if ($missingFromProxy.Count -gt 0) {
    throw "Built proxy is missing $($missingFromProxy.Count) exports; first missing: $($missingFromProxy[0])"
}

$metadata = [ordered]@{
    schema = 1
    generated_utc = [DateTime]::UtcNow.ToString('o')
    original_cusolver = $OriginalCusolver
    original_sha256 = (Get-FileHash $OriginalCusolver -Algorithm SHA256).Hash
    proxy = $outDll
    proxy_sha256 = (Get-FileHash $outDll -Algorithm SHA256).Hash
    original_export_count = $exports.Count
    proxy_export_count = $proxyExports.Count
    hip_overrides = $overrideNames
    forwarded_export_count = $allExports.Count - $overrideNames.Count
    gcc = $Gcc
    objdump = $Objdump
}
$metadataPath = Join-Path $OutputDir 'cusolver-proxy-build.json'
$metadata | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $metadataPath

Write-Host 'cuSOLVER compatibility proxy built'
Write-Host "Original exports : $($exports.Count)"
Write-Host "Proxy exports    : $($proxyExports.Count)"
Write-Host "hipSOLVER paths  : $($overrideNames.Count)"
Write-Host "DLL              : $outDll"
Write-Host "Metadata         : $metadataPath"

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OriginalCudnn,
    [string]$OutputDir,
    [string]$Gcc,
    [string]$Objdump
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$source = Join-Path $repo 'native\cudnn_bridge\cudnn_bridge.c'
if (-not (Test-Path $source)) { throw "Missing bridge source: $source" }

$OriginalCudnn = [IO.Path]::GetFullPath($OriginalCudnn)
if (-not (Test-Path $OriginalCudnn)) { throw "Original cuDNN DLL not found: $OriginalCudnn" }
$baseName = [IO.Path]::GetFileNameWithoutExtension($OriginalCudnn)
if ($baseName -notmatch '^cudnn64_[0-9]+$') {
    throw "Expected the top-level cuDNN DLL (for example cudnn64_8.dll), got: $baseName"
}
if (-not $OutputDir) { $OutputDir = Join-Path $repo '.runtime\cudnn-bridge' }
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $Gcc) {
    $cmd = Get-Command gcc.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { $Gcc = $cmd.Path }
}
if (-not $Gcc -or -not (Test-Path $Gcc)) {
    throw 'MinGW-w64 gcc.exe was not found. Pass -Gcc <path-to-gcc.exe>.'
}

if (-not $Objdump) {
    $cmd = Get-Command objdump.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { $Objdump = $cmd.Path }
}
if (-not $Objdump -or -not (Test-Path $Objdump)) {
    $candidate = Join-Path (Split-Path $Gcc -Parent) 'objdump.exe'
    if (Test-Path $candidate) { $Objdump = $candidate }
}
if (-not $Objdump -or -not (Test-Path $Objdump)) {
    throw 'GNU objdump.exe was not found.'
}

$manifestPath = Join-Path $repo 'native\cudnn_bridge\bridge-manifest.json'
if (-not (Test-Path $manifestPath)) { throw "Missing bridge manifest: $manifestPath" }
$bridgeManifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
if ([int]$bridgeManifest.schema -ne 1) { throw 'Unsupported cuDNN bridge manifest schema.' }
$overrideNames = @($bridgeManifest.overrides | ForEach-Object { [string]$_ })
if ($overrideNames.Count -lt 1) { throw 'cuDNN bridge manifest did not define overrides.' }

$overrideSet = @{}
foreach ($name in $overrideNames) { $overrideSet[$name] = $true }

$objdumpText = @(& $Objdump -p $OriginalCudnn 2>&1 | ForEach-Object { [string]$_ })
if ($LASTEXITCODE -ne 0) { throw "objdump failed for $OriginalCudnn" }
$exports = New-Object System.Collections.Generic.List[string]
foreach ($line in $objdumpText) {
    if ($line -match '^\s*\[\s*\d+\]\s+\+base\[.*?\]\s+[0-9A-Fa-f]+\s+(\S+)\s*$') {
        [void]$exports.Add($Matches[1])
    }
}
$exports = @($exports | Sort-Object -Unique)
if ($exports.Count -lt 50) {
    throw "Only $($exports.Count) exports were parsed. Refusing to build an incomplete cuDNN proxy."
}

$implemented = @($overrideNames | Where-Object { $_ -in $exports })
$missingOverrides = @($overrideNames | Where-Object { $_ -notin $exports })
if ($missingOverrides.Count) {
    Write-Warning ("Original DLL does not export some bridge entry points: " + ($missingOverrides -join ', '))
}

$forwardModule = $baseName + '_nvidia'
$defPath = Join-Path $OutputDir 'cudnn_bridge.generated.def'
$defLines = New-Object System.Collections.Generic.List[string]
[void]$defLines.Add("LIBRARY $baseName")
[void]$defLines.Add('EXPORTS')
foreach ($name in $exports) {
    if ($overrideSet.ContainsKey($name)) {
        [void]$defLines.Add("    $name")
    } else {
        [void]$defLines.Add("    $name=$forwardModule.$name")
    }
}
$defLines | Set-Content -Encoding ascii $defPath

$outDll = Join-Path $OutputDir ($baseName + '.dll')
& $Gcc -std=gnu17 -shared -O2 -Wall -Wextra -Werror -s -o $outDll $source $defPath
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $outDll)) {
    throw 'cuDNN compatibility bridge compilation failed.'
}

$proxyDump = @(& $Objdump -p $outDll 2>&1 | ForEach-Object { [string]$_ })
$proxyExports = @($proxyDump | ForEach-Object {
    if ($_ -match '^\s*\[\s*\d+\]\s+\+base\[.*?\]\s+[0-9A-Fa-f]+\s+(\S+)\s*$') { $Matches[1] }
} | Where-Object { $_ } | Sort-Object -Unique)

$missingFromProxy = @($exports | Where-Object { $_ -notin $proxyExports })
$extraInProxy = @($proxyExports | Where-Object { $_ -notin $exports })
if ($missingFromProxy.Count -or $extraInProxy.Count) {
    throw "Proxy export mismatch: missing=$($missingFromProxy.Count), extra=$($extraInProxy.Count)"
}

$metadata = [ordered]@{
    schema = 1
    generated_utc = [DateTime]::UtcNow.ToString('o')
    original_cudnn = $OriginalCudnn
    original_sha256 = (Get-FileHash $OriginalCudnn -Algorithm SHA256).Hash
    original_dll_name = [IO.Path]::GetFileName($OriginalCudnn)
    forward_dll_name = $forwardModule + '.dll'
    proxy = $outDll
    proxy_sha256 = (Get-FileHash $outDll -Algorithm SHA256).Hash
    original_export_count = $exports.Count
    proxy_export_count = $proxyExports.Count
    amd_native_override_count = $implemented.Count
    amd_native_overrides = $implemented
    forwarded_export_count = $exports.Count - $implemented.Count
    backend = 'MIOpen.dll (runtime-loaded; not redistributed)'
    stream_policy = 'default CUDA stream only until ZLUDA stream-handle translation is validated'
    gcc = $Gcc
    objdump = $Objdump
}
$metadataPath = Join-Path $OutputDir 'cudnn-bridge-build.json'
$metadata | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $metadataPath

Write-Host 'cuDNN -> MIOpen experimental bridge built'
Write-Host "Original exports : $($exports.Count)"
Write-Host "Proxy exports    : $($proxyExports.Count)"
Write-Host "AMD overrides    : $($implemented.Count)"
Write-Host "Forwarded        : $($exports.Count - $implemented.Count)"
Write-Host "DLL              : $outDll"
Write-Host "Metadata         : $metadataPath"
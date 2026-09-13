[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [switch]$DownloadZluda,
    [switch]$DownloadLibTorch,
    [string]$LibTorchRoot,
    [string]$ZludaRoot,
    [string]$HipRoot,
    [switch]$UseRecoveredCustomOverlay,
    [string]$RecoveredOverlayRoot
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
if (-not $RecoveredOverlayRoot) { $RecoveredOverlayRoot = Join-Path $repo 'local-artifacts\custom' }
New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null

if (-not $HipRoot) {
    if ($env:HIP_PATH) { $HipRoot = $env:HIP_PATH.TrimEnd('\\') }
    elseif (Test-Path 'C:\Program Files\AMD\ROCm\6.4') { $HipRoot = 'C:\Program Files\AMD\ROCm\6.4' }
}

if ($DownloadZluda) {
    $zludaPkg = Join-Path $RuntimeRoot 'zluda-v6-preview69'
    $zludaZip = Join-Path $RuntimeRoot 'zluda-windows-87531d3.zip'
    $zludaUrl = 'https://github.com/vosen/ZLUDA/releases/download/v6-preview.69/zluda-windows-87531d3.zip'
    if (-not (Test-Path (Join-Path $zludaPkg 'zluda\zluda.exe'))) {
        Write-Host '[setup] Downloading ZLUDA v6-preview.69 (Windows, commit 87531d3)...'
        Invoke-WebRequest -Uri $zludaUrl -OutFile $zludaZip
        if (Test-Path $zludaPkg) { Remove-Item $zludaPkg -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $zludaPkg | Out-Null
        Expand-Archive -Path $zludaZip -DestinationPath $zludaPkg -Force
    }
    if (Test-Path (Join-Path $zludaPkg 'zluda\zluda.exe')) { $ZludaRoot = Join-Path $zludaPkg 'zluda' }
    elseif (Test-Path (Join-Path $zludaPkg 'zluda.exe')) { $ZludaRoot = $zludaPkg }
    else { throw 'Downloaded ZLUDA archive did not contain zluda.exe in the expected location.' }
}

if ($DownloadLibTorch) {
    $dest = Join-Path $RuntimeRoot 'libtorch-2.3.0-cu118'
    $zip = Join-Path $RuntimeRoot 'libtorch-2.3.0+cu118.zip'
    $url = 'https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.3.0%2Bcu118.zip'
    if (-not (Test-Path (Join-Path $dest 'libtorch\lib\torch_cuda.dll'))) {
        Write-Host '[setup] Downloading LibTorch 2.3.0+cu118...'
        Invoke-WebRequest -Uri $url -OutFile $zip
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Expand-Archive -Path $zip -DestinationPath $dest -Force
    }
    $LibTorchRoot = Join-Path $dest 'libtorch'
}

$zludaRuntime = $null
if ($ZludaRoot) {
    if (-not (Test-Path (Join-Path $ZludaRoot 'zluda.exe'))) { throw "ZLUDA root does not contain zluda.exe: $ZludaRoot" }
    $zludaRuntime = Join-Path $RuntimeRoot 'zluda-core'
    if ((Resolve-Path $ZludaRoot).Path -ne (Resolve-Path $zludaRuntime -ErrorAction SilentlyContinue).Path) {
        if (Test-Path $zludaRuntime) { Remove-Item $zludaRuntime -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $zludaRuntime | Out-Null
        Copy-Item (Join-Path $ZludaRoot '*') $zludaRuntime -Recurse -Force
    }
} elseif (Test-Path (Join-Path $RuntimeRoot 'zluda-core\zluda.exe')) {
    $zludaRuntime = Join-Path $RuntimeRoot 'zluda-core'
}

$overlayRuntime = $null
if ($UseRecoveredCustomOverlay) {
    if (-not (Test-Path (Join-Path $RecoveredOverlayRoot 'cublas64_11.dll'))) {
        throw "Recovered custom overlay not found at $RecoveredOverlayRoot"
    }
    $overlayRuntime = Join-Path $RuntimeRoot 'custom-overlay'
    if (Test-Path $overlayRuntime) { Remove-Item $overlayRuntime -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $overlayRuntime | Out-Null
    Copy-Item (Join-Path $RecoveredOverlayRoot '*.dll') $overlayRuntime -Force
}

$config = [ordered]@{
    schema = 1
    recovered_profile = 'gfx1200-zluda-v6-preview69-hip713-libtorch230-cu118'
    zluda_root = $zludaRuntime
    hip_root = $HipRoot
    libtorch_root = $LibTorchRoot
    custom_overlay_root = $overlayRuntime
    zluda_cc = '8.6'
    torch_allow_tf32_cublas_override = '1'
    upstream = [ordered]@{
        zluda_release = 'v6-preview.69'
        zluda_windows_asset = 'zluda-windows-87531d3.zip'
        libtorch = '2.3.0+cu118'
    }
    notes = 'Historical profile: ZLUDA v6-preview.69 core + recovered HIP 7.13/custom BLAS overlay; ROCm 6.4 toolchain/libraries.'
}
$config | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $RuntimeRoot 'runtime-config.json')

Write-Host "[setup] Runtime config: $(Join-Path $RuntimeRoot 'runtime-config.json')"
if ($zludaRuntime) { Write-Host "[setup] ZLUDA: $zludaRuntime" } else { Write-Warning 'No ZLUDA root staged yet. Use -DownloadZluda or -ZludaRoot.' }
if ($HipRoot) { Write-Host "[setup] HIP: $HipRoot" } else { Write-Warning 'No HIP root detected/provided.' }
if ($LibTorchRoot) { Write-Host "[setup] LibTorch: $LibTorchRoot" } else { Write-Warning 'No LibTorch root configured. Use -DownloadLibTorch or -LibTorchRoot.' }
if ($overlayRuntime) { Write-Host "[setup] Custom overlay: $overlayRuntime" }

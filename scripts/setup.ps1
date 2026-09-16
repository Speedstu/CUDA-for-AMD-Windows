[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [switch]$AutoDetectGpu,
    [int]$GpuIndex = 0,
    [string]$ZludaCc = '8.6',
    [switch]$DownloadZluda,
    [ValidateSet('stable','latest')]
    [string]$ZludaChannel = 'stable',
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
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
if (-not $RecoveredOverlayRoot) { $RecoveredOverlayRoot = Join-Path $repo 'local-artifacts\custom' }
New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null

$scan = $null
$scanPath = $null
if ($AutoDetectGpu) {
    $scanPath = Join-Path $RuntimeRoot 'gpu-report.json'
    $scan = & (Join-Path $PSScriptRoot 'gpu-scan.ps1') -HipRoot $HipRoot -GpuIndex $GpuIndex -OutputPath $scanPath -Quiet -PassThru
    if (-not $scan.selected_gpu) { throw 'AutoDetectGpu could not find an AMD GPU. Run scripts/gpu-scan.ps1 for diagnostics.' }
    if (-not $HipRoot -and $scan.hip_root) { $HipRoot = ([string]$scan.hip_root).TrimEnd('\') }
    Write-Host "[setup] GPU: $($scan.selected_gpu.name) / $($scan.selected_gpu.gfx) [$($scan.selected_gpu.project_status)]"
}

if (-not $HipRoot) {
    if ($env:HIP_PATH -and (Test-Path $env:HIP_PATH)) {
        $HipRoot = $env:HIP_PATH.TrimEnd('\')
    } else {
        $rocmBase = Join-Path $env:ProgramFiles 'AMD\ROCm'
        if (Test-Path $rocmBase) {
            $candidate = Get-ChildItem $rocmBase -Directory -ErrorAction SilentlyContinue | Sort-Object {
                try { [version]$_.Name } catch { [version]'0.0' }
            } -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'bin\hipInfo.exe') } | Select-Object -First 1
            if ($candidate) { $HipRoot = $candidate.FullName }
        }
    }
}

if ($DownloadZluda) {
    $releaseManifest = Get-Content (Join-Path $repo 'manifests\zluda-releases.json') -Raw | ConvertFrom-Json
    $channel = $releaseManifest.channels.$ZludaChannel
    if (-not $channel) { throw "Unknown ZLUDA channel: $ZludaChannel" }
    $release = [string]$channel.release
    $asset = [string]$channel.windows_asset
    $expectedZluda = ([string]$channel.sha256).ToUpperInvariant()
    $slug = ($release -replace '[^A-Za-z0-9._-]', '-')
    $zludaPkg = Join-Path $RuntimeRoot "zluda-$slug"
    $zludaZip = Join-Path $RuntimeRoot $asset
    $zludaUrl = "https://github.com/vosen/ZLUDA/releases/download/$release/$asset"
    if (-not (Test-Path (Join-Path $zludaPkg 'zluda\zluda.exe'))) {
        Write-Host "[setup] Downloading ZLUDA $release ($ZludaChannel channel)..."
        Invoke-WebRequest -Uri $zludaUrl -OutFile $zludaZip
        $actualZluda = (Get-FileHash $zludaZip -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($actualZluda -ne $expectedZluda) { Remove-Item $zludaZip -Force; throw "ZLUDA SHA-256 mismatch: $actualZluda" }
        Write-Host "[setup] ZLUDA SHA-256 verified: $actualZluda"
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
        Write-Host '[setup] Downloading LibTorch 2.3.0+cu118 (~2.66 GB)...'
        Invoke-WebRequest -Uri $url -OutFile $zip
        $expectedTorch = 'E7D57EE5052996E1A9AAEAD5ECC3C491BA7C0DB21316FB1FA8A4A8136005C6CC'
        $actualTorch = (Get-FileHash $zip -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($actualTorch -ne $expectedTorch) { Remove-Item $zip -Force; throw "LibTorch SHA-256 mismatch: $actualTorch" }
        Write-Host "[setup] LibTorch SHA-256 verified: $actualTorch"
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Expand-Archive -Path $zip -DestinationPath $dest -Force
    }
    $LibTorchRoot = Join-Path $dest 'libtorch'
}

if (-not $release) { $release = if ($ZludaChannel -eq 'latest') { 'v7-preview.10' } else { 'v6-preview.69' } }
if (-not $asset) { $asset = if ($ZludaChannel -eq 'latest') { 'zluda-windows-9c8b43f.zip' } else { 'zluda-windows-87531d3.zip' } }
if (-not $slug) { $slug = ($release -replace '[^A-Za-z0-9._-]', '-') }

$zludaRuntime = $null
if ($ZludaRoot) {
    if (-not (Test-Path (Join-Path $ZludaRoot 'zluda.exe'))) { throw "ZLUDA root does not contain zluda.exe: $ZludaRoot" }
    $zludaRuntime = Join-Path $RuntimeRoot 'zluda-core'
    $existingRuntime = Resolve-Path $zludaRuntime -ErrorAction SilentlyContinue
    if (-not $existingRuntime -or (Resolve-Path $ZludaRoot).Path -ne $existingRuntime.Path) {
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
    if ($scan -and -not $scan.selected_gpu.project_tested) {
        Write-Warning 'The recovered custom BLAS/HIP overlay was only tested on RX 9060 XT / gfx1200. Using it on this GPU is experimental.'
    }
    $overlayRuntime = Join-Path $RuntimeRoot 'custom-overlay'
    if (Test-Path $overlayRuntime) { Remove-Item $overlayRuntime -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $overlayRuntime | Out-Null
    Copy-Item (Join-Path $RecoveredOverlayRoot '*.dll') $overlayRuntime -Force
}

$gpuName = if ($scan) { [string]$scan.selected_gpu.name } else { $null }
$gpuArch = if ($scan) { [string]$scan.selected_gpu.gfx } else { $null }
$gpuStatus = if ($scan) { [string]$scan.selected_gpu.project_status } else { 'not-scanned' }
$isReference = [bool]($scan -and $scan.selected_gpu.project_tested)
$profileName = if ($isReference -and $overlayRuntime) {
    'reference-gfx1200-zluda-v6-preview69-custom-overlay-libtorch230-cu118'
} elseif ($gpuArch) {
    "auto-$gpuArch-zluda-$slug-libtorch230-cu118"
} else {
    "manual-zluda-$slug-libtorch230-cu118"
}

$config = [ordered]@{
    schema = 2
    profile = $profileName
    gpu = [ordered]@{
        name = $gpuName
        arch = $gpuArch
        index = if ($scan) { $scan.selected_gpu.index } else { $null }
        project_status = $gpuStatus
        tested_reference = $isReference
        scanner_report = $scanPath
    }
    zluda_root = $zludaRuntime
    hip_root = $HipRoot
    libtorch_root = $LibTorchRoot
    custom_overlay_root = $overlayRuntime
    zluda_cc = $ZludaCc
    torch_allow_tf32_cublas_override = '1'
    upstream = [ordered]@{
        zluda_release = if ($release) { $release } else { $null }
        zluda_windows_asset = if ($asset) { $asset } else { $null }
        zluda_channel = $ZludaChannel
        libtorch = '2.3.0+cu118'
    }
    validation = [ordered]@{
        reference_gpu = 'AMD Radeon RX 9060 XT'
        reference_arch = 'gfx1200'
        other_gpus = 'unverified until community-tested'
    }
}
$config | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $RuntimeRoot 'runtime-config.json')

Write-Host "[setup] Runtime config: $(Join-Path $RuntimeRoot 'runtime-config.json')"
if ($zludaRuntime) { Write-Host "[setup] ZLUDA: $zludaRuntime" } else { Write-Warning 'No ZLUDA root staged yet. Use -DownloadZluda or -ZludaRoot.' }
if ($HipRoot) { Write-Host "[setup] HIP: $HipRoot" } else { Write-Warning 'No HIP root detected/provided.' }
if ($LibTorchRoot) { Write-Host "[setup] LibTorch: $LibTorchRoot" } else { Write-Warning 'No LibTorch root configured. Use -DownloadLibTorch or -LibTorchRoot.' }
if ($overlayRuntime) { Write-Host "[setup] Custom overlay: $overlayRuntime" }
if ($scanPath) { Write-Host "[setup] GPU report: $scanPath" }

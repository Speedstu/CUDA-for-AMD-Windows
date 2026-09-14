[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [switch]$AutoDetectGpu,
    [int]$GpuIndex = -1,
    [string]$ZludaCc = '8.6',
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
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
if (-not $RecoveredOverlayRoot) { $RecoveredOverlayRoot = Join-Path $repo 'local-artifacts\custom' }
New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null

if (-not $LibTorchRoot) {
    $existingConfigPath = Join-Path $RuntimeRoot 'runtime-config.json'
    if (Test-Path $existingConfigPath) {
        $existingConfig = Get-Content $existingConfigPath -Raw | ConvertFrom-Json
        if ($existingConfig.libtorch_root -and (Test-Path (Join-Path ([string]$existingConfig.libtorch_root) 'lib\torch_cuda.dll'))) {
            $LibTorchRoot = [string]$existingConfig.libtorch_root
        }
    }
    if (-not $LibTorchRoot) {
        $defaultLibTorchRoot = Join-Path $RuntimeRoot 'libtorch-2.3.0-cu118\libtorch'
        if (Test-Path (Join-Path $defaultLibTorchRoot 'lib\torch_cuda.dll')) {
            $LibTorchRoot = $defaultLibTorchRoot
        }
    }
}

function Download-File {
    param(
        [Parameter(Mandatory=$true)][string]$Uri,
        [Parameter(Mandatory=$true)][string]$Destination
    )

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        Write-Host '[setup] Downloading with curl (resume/retry enabled)...'
        & $curl.Source --fail --location --retry 5 --retry-delay 3 --retry-all-errors --continue-at - --output $Destination $Uri
        if ($LASTEXITCODE -eq 0) { return }
        throw "curl failed with exit code $LASTEXITCODE"
    }

    Write-Warning 'curl.exe was not found; falling back to Invoke-WebRequest (no resume support).'
    Invoke-WebRequest -Uri $Uri -OutFile $Destination
}

function Get-Sha256WithProgress {
    param([Parameter(Mandatory=$true)][string]$Path)

    $stream = $null
    $sha256 = $null
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        $buffer = New-Object byte[] (4MB)
        $total = $stream.Length
        $readTotal = [int64]0
        $lastPercent = -1
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        Write-Host ("[setup] Verifying SHA-256: {0:N2} GB" -f ($total / 1GB))

        while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            [void]$sha256.TransformBlock($buffer, 0, $read, $buffer, 0)
            $readTotal += $read
            $percent = [int][Math]::Floor(($readTotal * 100) / $total)
            if ($percent -ge ($lastPercent + 5) -or $readTotal -eq $total) {
                $rate = if ($timer.Elapsed.TotalSeconds -gt 0) { $readTotal / 1MB / $timer.Elapsed.TotalSeconds } else { 0 }
                Write-Host ("[setup] Hash progress: {0,3}% ({1:N1}/{2:N1} GB, {3:N1} MB/s)" -f $percent, ($readTotal / 1GB), ($total / 1GB), $rate)
                $lastPercent = $percent
            }
        }
        [void]$sha256.TransformFinalBlock([byte[]]::new(0), 0, 0)
        return ([BitConverter]::ToString($sha256.Hash) -replace '-', '').ToUpperInvariant()
    } finally {
        if ($sha256) { $sha256.Dispose() }
        if ($stream) { $stream.Dispose() }
    }
}

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
    $zludaPkg = Join-Path $RuntimeRoot 'zluda-v6-preview69'
    $zludaZip = Join-Path $RuntimeRoot 'zluda-windows-87531d3.zip'
    $zludaUrl = 'https://github.com/vosen/ZLUDA/releases/download/v6-preview.69/zluda-windows-87531d3.zip'
    if (-not (Test-Path (Join-Path $zludaPkg 'zluda\zluda.exe'))) {
        Write-Host '[setup] Downloading ZLUDA v6-preview.69 (Windows, commit 87531d3)...'
        Download-File -Uri $zludaUrl -Destination $zludaZip
        $expectedZluda = 'E2959ED17C8DDAE6BF2BF76CF3A7348F577A2548754685D776389DF2220E5D9A'
        $actualZluda = Get-Sha256WithProgress -Path $zludaZip
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
        Download-File -Uri $url -Destination $zip
        $expectedTorch = 'E7D57EE5052996E1A9AAEAD5ECC3C491BA7C0DB21316FB1FA8A4A8136005C6CC'
        $actualTorch = Get-Sha256WithProgress -Path $zip
        if ($actualTorch -ne $expectedTorch) { Remove-Item $zip -Force; throw "LibTorch SHA-256 mismatch: $actualTorch" }
        Write-Host "[setup] LibTorch SHA-256 verified: $actualTorch"
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Expand-Archive -Path $zip -DestinationPath $dest -Force
    }
    $LibTorchRoot = Join-Path $dest 'libtorch'
}

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
        Write-Warning 'The recovered custom BLAS/HIP overlay was only tested on RX 9060 XT / gfx1200. Using it on gfx1201 is experimental.'
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
    'reference-gfx1201-zluda-v6-preview69-custom-overlay-libtorch230-cu118'
} elseif ($gpuArch) {
    "auto-$gpuArch-zluda-v6-preview69-libtorch230-cu118"
} else {
    'manual-zluda-v6-preview69-libtorch230-cu118'
}

$config = [ordered]@{
    schema = 2
    profile = $profileName
    gpu = [ordered]@{
        name = $gpuName
        arch = $gpuArch
        index = if ($scan) { $scan.selected_gpu.index } else { $null }
        hip_visible_device = if ($scan) { [string]$scan.selected_gpu.index } else { $null }
        project_status = $gpuStatus
        tested_reference = $isReference
        scanner_report = $scanPath
    }
    zluda_root = $zludaRuntime
    hip_root = $HipRoot
    hip_sdk_target = '7.2'
    libtorch_root = $LibTorchRoot
    custom_overlay_root = $overlayRuntime
    zluda_cc = $ZludaCc
    torch_allow_tf32_cublas_override = '1'
    upstream = [ordered]@{
        zluda_release = 'v6-preview.69'
        zluda_windows_asset = 'zluda-windows-87531d3.zip'
        libtorch = '2.3.0+cu118'
    }
    validation = [ordered]@{
        reference_gpu = 'AMD Radeon RX 9070'
        reference_arch = 'gfx1201'
        hip_sdk = '7.2'
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

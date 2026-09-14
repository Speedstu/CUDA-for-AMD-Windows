[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [string]$HipRoot,
    [string]$LibTorchRoot,
    [int]$GpuIndex = -1,
    [switch]$SkipLibTorch,
    [switch]$SkipRuntimeTest
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)

Write-Host 'CUDA for AMD on Windows - installer'
Write-Host '==================================='
Write-Host '1/4 Checking AMD/HIP prerequisites...'
$doctor = & (Join-Path $PSScriptRoot 'doctor.ps1') -RuntimeRoot $RuntimeRoot -HipRoot $HipRoot -PassThru
if (-not $doctor.ok) {
    throw 'HIP SDK prerequisites are missing. Install AMD HIP SDK with HIP Libraries, then rerun install.ps1.'
}
$HipRoot = [string]$doctor.hip_root

Write-Host ''
Write-Host '2/4 Preparing pinned ZLUDA + CUDA-facing runtime...'
$setupArgs = @{
    RuntimeRoot = $RuntimeRoot
    AutoDetectGpu = $true
    GpuIndex = $GpuIndex
    HipRoot = $HipRoot
    DownloadZluda = $true
}
if ($LibTorchRoot) { $setupArgs.LibTorchRoot = $LibTorchRoot }
elseif (-not $SkipLibTorch) { $setupArgs.DownloadLibTorch = $true }
& (Join-Path $PSScriptRoot 'setup.ps1') @setupArgs

Write-Host ''
Write-Host '3/4 Verifying prepared files...'
& (Join-Path $PSScriptRoot 'verify.ps1') -RuntimeRoot $RuntimeRoot

if (-not $SkipRuntimeTest) {
    Write-Host ''
    Write-Host '4/4 Running ZLUDA/HIP runtime validation...'
    & (Join-Path $PSScriptRoot 'test-runtime.ps1') -RuntimeRoot $RuntimeRoot
} else {
    Write-Host ''
    Write-Host '4/4 Runtime test skipped by request.'
}

$config = Get-Content (Join-Path $RuntimeRoot 'runtime-config.json') -Raw | ConvertFrom-Json
Write-Host ''
Write-Host 'READY'
Write-Host "GPU     : $($config.gpu.name) / $($config.gpu.arch)"
Write-Host "ZLUDA   : $($config.zluda_root)"
Write-Host "HIP SDK : $($config.hip_root)"
if ($config.libtorch_root) { Write-Host "LibTorch: $($config.libtorch_root)" }
Write-Host ''
Write-Host 'To run a CUDA-targeted .exe:'
Write-Host '  .\scripts\run-zluda.ps1 -Program C:\path\to\app.exe'

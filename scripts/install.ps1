[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [string]$HipRoot,
    [string]$LibTorchRoot,
    [int]$GpuIndex = -1,
    [string]$FunctionalPython,
    [ValidateSet('stable','latest')]
    [string]$ZludaChannel = 'stable',
    [switch]$SkipLibTorch,
    [switch]$SkipRuntimeTest,
    [switch]$SkipFunctionalTest
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)

Write-Host 'CUDA for AMD on Windows - installer'
Write-Host '==================================='
Write-Host '1/5 Checking AMD/HIP prerequisites...'
$doctor = & (Join-Path $PSScriptRoot 'doctor.ps1') -RuntimeRoot $RuntimeRoot -HipRoot $HipRoot -PassThru
if (-not $doctor.ok) {
    throw 'HIP SDK prerequisites are missing. Install AMD HIP SDK with HIP Libraries, then rerun install.ps1.'
}
$HipRoot = [string]$doctor.hip_root
$hipCompat = & (Join-Path $PSScriptRoot 'check-hip-compat.ps1') -HipRoot $HipRoot -GpuIndex $GpuIndex -PassThru
if (-not $hipCompat.compatible) { throw $hipCompat.reason }
Write-Host "[PASS] HIP version gate: $($hipCompat.hip_version) / $($hipCompat.gfx)"

Write-Host ''
Write-Host '2/5 Preparing pinned ZLUDA + CUDA-facing runtime...'
$setupArgs = @{
    RuntimeRoot = $RuntimeRoot
    AutoDetectGpu = $true
    GpuIndex = $GpuIndex
    HipRoot = $HipRoot
    DownloadZluda = $true
    ZludaChannel = $ZludaChannel
}
if ($LibTorchRoot) { $setupArgs.LibTorchRoot = $LibTorchRoot }
elseif (-not $SkipLibTorch) { $setupArgs.DownloadLibTorch = $true }
& (Join-Path $PSScriptRoot 'setup.ps1') @setupArgs

Write-Host ''
Write-Host '3/5 Verifying prepared files...'
& (Join-Path $PSScriptRoot 'verify.ps1') -RuntimeRoot $RuntimeRoot

if (-not $SkipRuntimeTest) {
    Write-Host ''
    Write-Host '4/5 Running ZLUDA/HIP runtime smoke validation...'
    & (Join-Path $PSScriptRoot 'test-runtime.ps1') -RuntimeRoot $RuntimeRoot
} else {
    Write-Host ''
    Write-Host '4/5 Runtime smoke test skipped by request.'
}

$functionalReport = $null
if (-not $SkipFunctionalTest) {
    Write-Host ''
    Write-Host '5/5 Running numerical functional validation when available...'
    $functionalArgs = @{ RuntimeRoot = $RuntimeRoot }
    if ($FunctionalPython) { $functionalArgs.PythonExe = $FunctionalPython }
    & (Join-Path $PSScriptRoot 'test-functional.ps1') @functionalArgs
    $functionalPath = Join-Path $RuntimeRoot 'functional-test.json'
    if (Test-Path $functionalPath) { $functionalReport = Get-Content $functionalPath -Raw | ConvertFrom-Json }
    if ($functionalReport -and $functionalReport.available -and -not $functionalReport.core_correctness_ok) {
        throw 'Core dense/GEMM numerical validation failed. This runtime must not be treated as safe for the validated PPO/GEMM workload profile.'
    }
} else {
    Write-Host ''
    Write-Host '5/5 Functional test skipped by request.'
}

$config = Get-Content (Join-Path $RuntimeRoot 'runtime-config.json') -Raw | ConvertFrom-Json
$referenceGpu = [bool]($config.gpu -and $config.gpu.tested_reference)
$functionalPass = [bool]($functionalReport -and $functionalReport.available -and $functionalReport.core_correctness_ok)

Write-Host ''
if ($functionalPass) {
    Write-Host 'READY - runtime smoke + dense/GEMM core numerical checks passed'
    if ($functionalReport -and -not $functionalReport.correctness_ok) {
        Write-Warning 'One or more extended CUDA-facing capabilities failed or are unavailable. Review .runtime\functional-test.json before using convolution/attention-heavy workloads.'
    }
} elseif ($referenceGpu) {
    Write-Host 'READY - reference runtime prepared; numerical probe was not completed in this install run'
    Write-Warning 'For changes to the stack, run scripts/test-functional.ps1 before trusting new workload results.'
} else {
    Write-Warning 'RUNTIME READY, FUNCTIONAL VALIDATION INCOMPLETE.'
    Write-Warning 'This GPU is not a validated reference. Do not interpret cuda_check/doctor success as proof of numerical correctness.'
    Write-Host 'Run: .\scripts\test-functional.ps1 -PythonExe <cuda-pytorch-venv\Scripts\python.exe> -Strict'
}
Write-Host "GPU     : $($config.gpu.name) / $($config.gpu.arch) [$($config.gpu.project_status)]"
Write-Host "ZLUDA   : $($config.zluda_root)"
Write-Host "HIP SDK : $($config.hip_root)"
if ($config.libtorch_root) { Write-Host "LibTorch: $($config.libtorch_root)" }
Write-Host ''
Write-Host 'To run a CUDA-targeted .exe:'
Write-Host '  .\scripts\run-zluda.ps1 -Program C:\path\to\app.exe'

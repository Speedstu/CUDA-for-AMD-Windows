[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [string]$HipRoot,
    [int]$GpuIndex = 0,
    [switch]$Strict,
    [switch]$PassThru
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)

$scan = & (Join-Path $PSScriptRoot 'gpu-scan.ps1') -HipRoot $HipRoot -GpuIndex $GpuIndex -Quiet -PassThru
if (-not $HipRoot -and $scan.hip_root) { $HipRoot = [string]$scan.hip_root }

$driverRuntime = $false
foreach ($name in @('amdhip64_7.dll','amdhip64_6.dll')) {
    if (Test-Path (Join-Path $env:WINDIR "System32\$name")) { $driverRuntime = $true }
}

$checks = [ordered]@{
    amd_driver_hip_runtime = [bool]$driverRuntime
    hip_sdk = [bool]($HipRoot -and (Test-Path $HipRoot))
    hip_info = [bool]($HipRoot -and (Test-Path (Join-Path $HipRoot 'bin\hipInfo.exe')))
    rocblas = [bool]($HipRoot -and (Test-Path (Join-Path $HipRoot 'bin\rocblas.dll')))
    hipblaslt = [bool]($HipRoot -and ((Test-Path (Join-Path $HipRoot 'bin\hipblaslt.dll')) -or (Test-Path (Join-Path $HipRoot 'bin\libhipblaslt.dll'))))
    rocsparse = [bool]($HipRoot -and (Test-Path (Join-Path $HipRoot 'bin\rocsparse.dll')))
}

$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
$config = if (Test-Path $configPath) { Get-Content $configPath -Raw | ConvertFrom-Json } else { $null }
$checks.zluda = [bool]($config -and $config.zluda_root -and (Test-Path (Join-Path $config.zluda_root 'zluda.exe')))
$checks.cuda_check = [bool]($config -and $config.zluda_root -and (Test-Path (Join-Path $config.zluda_root 'cuda_check.exe')))
$checks.libtorch_cuda = [bool]($config -and $config.libtorch_root -and (Test-Path (Join-Path $config.libtorch_root 'lib\torch_cuda.dll')))

$compat = $null
if ($checks.hip_sdk -and $scan.selected_gpu) {
    try { $compat = & (Join-Path $PSScriptRoot 'check-hip-compat.ps1') -HipRoot $HipRoot -GpuIndex $GpuIndex -PassThru } catch {}
}
$versionOk = [bool](-not $compat -or $compat.compatible)
$checks.hip_version_compatible = $versionOk

Write-Host 'CUDA for AMD - doctor'
Write-Host '---------------------'
if ($scan.selected_gpu) {
    Write-Host "GPU       : $($scan.selected_gpu.name) / $($scan.selected_gpu.gfx) [$($scan.selected_gpu.project_status)]"
} else { Write-Warning 'GPU       : no AMD GPU detected' }
if ($HipRoot) { Write-Host "HIP SDK   : $HipRoot" } else { Write-Warning 'HIP SDK   : not found' }
if ($compat -and $compat.minimum_hip_sdk) { Write-Host "HIP floor : $($compat.minimum_hip_sdk)" }

foreach ($entry in $checks.GetEnumerator()) {
    $tag = if ($entry.Value) { 'PASS' } else { 'MISS' }
    Write-Host ("[{0}] {1}" -f $tag, $entry.Key)
}
if ($compat -and -not $compat.compatible) { Write-Warning $compat.reason }

$coreOk = $checks.amd_driver_hip_runtime -and $checks.hip_sdk -and $checks.hip_version_compatible -and $checks.rocblas -and $checks.hipblaslt -and $checks.rocsparse
$functionalRequired = [bool]($scan.selected_gpu -and $scan.selected_gpu.functional_validation_required)

if (-not $coreOk) {
    Write-Warning 'AMD HIP prerequisites are incomplete or incompatible with this GPU profile.'
}
if ($scan.selected_gpu -and $scan.selected_gpu.project_status -eq 'community-partial') {
    Write-Warning 'Known partial compatibility report for this GPU. doctor PASS means prerequisites only; it is NOT a numerical correctness result.'
    Write-Host 'Run .\scripts\test-functional.ps1 -PythonExe <cuda-pytorch-venv\Scripts\python.exe> -Strict'
} elseif ($functionalRequired) {
    Write-Warning 'This is not a validated reference GPU. Numerical functional validation is required before trusting workload output.'
}

$result = [pscustomobject][ordered]@{
    ok = [bool]$coreOk
    prerequisites_ok = [bool]$coreOk
    numerical_correctness_tested = $false
    functional_validation_required = $functionalRequired
    hip_root = $HipRoot
    hip_version = if ($scan) { $scan.hip_version } else { $null }
    gpu = if ($scan) { $scan.selected_gpu } else { $null }
    checks = [pscustomobject]$checks
}
if ($PassThru) { return $result }
if ($Strict -and -not $coreOk) { throw 'AMD HIP SDK prerequisites are incomplete or incompatible.' }

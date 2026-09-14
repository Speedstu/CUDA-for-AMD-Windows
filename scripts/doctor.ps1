[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [string]$HipRoot,
    [switch]$Strict,
    [switch]$PassThru
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)

function Find-HipRoot {
    param([string]$Explicit)
    if ($Explicit -and (Test-Path $Explicit)) { return (Resolve-Path $Explicit).Path.TrimEnd('\') }
    if ($env:HIP_PATH -and (Test-Path $env:HIP_PATH)) { return (Resolve-Path $env:HIP_PATH).Path.TrimEnd('\') }
    $base = Join-Path $env:ProgramFiles 'AMD\ROCm'
    if (Test-Path $base) {
        $candidates = Get-ChildItem $base -Directory -ErrorAction SilentlyContinue | Sort-Object {
            try { [version]$_.Name } catch { [version]'0.0' }
        } -Descending
        foreach ($candidate in $candidates) {
            if (Test-Path (Join-Path $candidate.FullName 'bin\rocblas.dll')) { return $candidate.FullName.TrimEnd('\') }
        }
    }
    return $null
}

$hip = Find-HipRoot $HipRoot
$driverRuntime = @('amdhip64_7.dll','amdhip64_6.dll') | ForEach-Object {
    $path = Join-Path $env:WINDIR "System32\$_"
    if (Test-Path $path) {
        $item = Get-Item $path
        [ordered]@{ name=$_.ToString(); path=$path; version=$item.VersionInfo.FileVersion }
    }
}

$gpu = $null
try { $gpu = & (Join-Path $PSScriptRoot 'gpu-scan.ps1') -HipRoot $hip -Quiet -PassThru } catch {}

$checks = [ordered]@{}
$checks.amd_driver_hip_runtime = [bool]($driverRuntime.Count -gt 0)
$checks.hip_sdk = [bool]$hip
$checks.hip_info = [bool]($hip -and (Test-Path (Join-Path $hip 'bin\hipInfo.exe')))
$checks.rocblas = [bool]($hip -and (Test-Path (Join-Path $hip 'bin\rocblas.dll')))
$checks.hipblaslt = [bool]($hip -and ((Test-Path (Join-Path $hip 'bin\hipblaslt.dll')) -or (Test-Path (Join-Path $hip 'bin\libhipblaslt.dll'))))
$checks.rocsparse = [bool]($hip -and (Test-Path (Join-Path $hip 'bin\rocsparse.dll')))

$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
$config = if (Test-Path $configPath) { Get-Content $configPath -Raw | ConvertFrom-Json } else { $null }
$checks.zluda = [bool]($config -and $config.zluda_root -and (Test-Path (Join-Path $config.zluda_root 'zluda.exe')))
$checks.cuda_check = [bool]($config -and $config.zluda_root -and (Test-Path (Join-Path $config.zluda_root 'cuda_check.exe')))
$checks.libtorch_cuda = [bool]($config -and $config.libtorch_root -and (Test-Path (Join-Path $config.libtorch_root 'lib\torch_cuda.dll')))

Write-Host 'CUDA for AMD - doctor'
Write-Host '---------------------'
if ($gpu -and $gpu.selected_gpu) {
    Write-Host "GPU       : $($gpu.selected_gpu.name) / $($gpu.selected_gpu.gfx) [$($gpu.selected_gpu.project_status)]"
} else { Write-Warning 'GPU       : no AMD GPU detected' }
if ($hip) { Write-Host "HIP SDK   : $hip" } else { Write-Warning 'HIP SDK   : not found' }
if ($driverRuntime) {
    foreach ($r in $driverRuntime) { Write-Host "Driver HIP: $($r.name) $($r.version)" }
} else { Write-Warning 'Driver HIP: amdhip64_6/7.dll not found in System32' }

foreach ($entry in $checks.GetEnumerator()) {
    $tag = if ($entry.Value) { 'PASS' } else { 'MISS' }
    Write-Host ("[{0}] {1}" -f $tag, $entry.Key)
}

$coreOk = $checks.amd_driver_hip_runtime -and $checks.hip_sdk -and $checks.rocblas -and $checks.hipblaslt -and $checks.rocsparse
if (-not $coreOk) {
    Write-Host ''
    Write-Warning 'AMD HIP SDK prerequisites are incomplete. Install the Windows HIP SDK (including HIP Libraries), then rerun this script.'
    Write-Host 'AMD installation guide: https://rocm.docs.amd.com/projects/HIP/en/latest/hip-sdk/windows/install.html'
}

$result = [pscustomobject][ordered]@{
    ok = [bool]$coreOk
    hip_root = $hip
    gpu = if ($gpu) { $gpu.selected_gpu } else { $null }
    driver_runtime = $driverRuntime
    checks = [pscustomobject]$checks
}
if ($PassThru) { return $result }
if ($Strict -and -not $coreOk) { throw 'AMD HIP SDK prerequisites are incomplete.' }

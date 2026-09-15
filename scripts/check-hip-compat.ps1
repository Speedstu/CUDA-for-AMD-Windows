[CmdletBinding()]
param(
    [string]$HipRoot,
    [int]$GpuIndex = 0,
    [switch]$PassThru
)

$ErrorActionPreference = 'Stop'

function Convert-ToVersion {
    param([string]$Text)
    if (-not $Text) { return $null }
    $m = [regex]::Match($Text, '(\d+)\.(\d+)(?:\.(\d+))?')
    if (-not $m.Success) { return $null }
    $patch = if ($m.Groups[3].Success) { $m.Groups[3].Value } else { '0' }
    try { return [version]("{0}.{1}.{2}" -f $m.Groups[1].Value, $m.Groups[2].Value, $patch) } catch { return $null }
}

$scan = & (Join-Path $PSScriptRoot 'gpu-scan.ps1') -HipRoot $HipRoot -GpuIndex $GpuIndex -Quiet -PassThru
if (-not $scan.selected_gpu) { throw 'No AMD GPU was detected.' }
if (-not $scan.hip_root) { throw 'No HIP SDK was detected.' }

$currentText = [string]$scan.hip_version
$current = Convert-ToVersion $currentText
$minimumText = [string]$scan.selected_gpu.minimum_hip_sdk
$minimum = Convert-ToVersion $minimumText
$compatible = $true
$reason = $null

if ($minimum) {
    if (-not $current) {
        $compatible = $false
        $reason = "Could not parse HIP SDK version '$currentText'; $($scan.selected_gpu.gfx) requires HIP SDK $minimumText or newer."
    } elseif ($current -lt $minimum) {
        $compatible = $false
        $reason = "HIP SDK $currentText is too old for $($scan.selected_gpu.gfx); minimum recorded version is $minimumText."
    }
}

$result = [pscustomobject][ordered]@{
    compatible = [bool]$compatible
    gpu = $scan.selected_gpu.name
    gfx = $scan.selected_gpu.gfx
    hip_root = $scan.hip_root
    hip_version = $currentText
    minimum_hip_sdk = if ($minimumText) { $minimumText } else { $null }
    project_status = $scan.selected_gpu.project_status
    functional_validation_required = [bool]$scan.selected_gpu.functional_validation_required
    reason = $reason
}

if (-not $PassThru) {
    Write-Host "GPU       : $($result.gpu) / $($result.gfx) [$($result.project_status)]"
    Write-Host "HIP SDK   : $($result.hip_root)"
    if ($minimumText) { Write-Host "HIP floor : $minimumText" }
    if ($compatible) { Write-Host '[PASS] HIP SDK version is compatible with the recorded architecture floor.' }
    else { Write-Warning $reason }
}

if ($PassThru) { return $result }
if (-not $compatible) { throw $reason }

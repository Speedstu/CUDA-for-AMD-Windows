[CmdletBinding()]
param(
    [string]$HipRoot,
    [int]$GpuIndex = -1,
    [string]$OutputPath,
    [switch]$AsJson,
    [switch]$Quiet,
    [switch]$PassThru
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$profilesPath = Join-Path $repo 'manifests\windows-gpu-profiles.json'
$profiles = Get-Content $profilesPath -Raw | ConvertFrom-Json

function Find-HipRoot {
    param([string]$Requested)
    if ($Requested -and (Test-Path $Requested)) { return (Resolve-Path $Requested).Path }
    if ($env:HIP_PATH -and (Test-Path $env:HIP_PATH)) { return (Resolve-Path $env:HIP_PATH).Path }

    $base = Join-Path $env:ProgramFiles 'AMD\ROCm'
    if (Test-Path $base) {
        $dirs = Get-ChildItem $base -Directory -ErrorAction SilentlyContinue | Sort-Object {
            try { [version]$_.Name } catch { [version]'0.0' }
        } -Descending
        foreach ($dir in $dirs) {
            if (Test-Path (Join-Path $dir.FullName 'bin\hipInfo.exe')) { return $dir.FullName }
        }
    }
    return $null
}

function Get-FallbackArch {
    param([string]$Name)
    foreach ($entry in $profiles.fallback_model_patterns) {
        if ($Name -match $entry.pattern) { return [string]$entry.gfx }
    }
    return $null
}

function Get-ArchMetadata {
    param([string]$Arch)
    if (-not $Arch) { return $null }
    $prop = $profiles.architectures.PSObject.Properties[$Arch]
    if ($prop) { return $prop.Value }
    return $null
}

$resolvedHip = Find-HipRoot $HipRoot
$hipInfoPath = if ($resolvedHip) { Join-Path $resolvedHip 'bin\hipInfo.exe' } else { $null }
$hipDevices = @()

if ($hipInfoPath -and (Test-Path $hipInfoPath)) {
    $lines = @(& $hipInfoPath 2>&1 | ForEach-Object { [string]$_ })
    $current = $null
    foreach ($line in $lines) {
        if ($line -match '^device#\s+(\d+)\s*$') {
            if ($current) { $hipDevices += [pscustomobject]$current }
            $current = [ordered]@{ index = [int]$Matches[1]; name = $null; gfx = $null; memory = $null; major = $null; minor = $null }
            continue
        }
        if (-not $current) { continue }
        if ($line -match '^Name:\s+(.+?)\s*$') { $current.name = $Matches[1].Trim(); continue }
        if ($line -match '^gcnArchName:\s+(\S+)') { $current.gfx = $Matches[1].Trim(); continue }
        if ($line -match '^totalGlobalMem:\s+(.+?)\s*$') { $current.memory = $Matches[1].Trim(); continue }
        if ($line -match '^major:\s+(\d+)') { $current.major = [int]$Matches[1]; continue }
        if ($line -match '^minor:\s+(\d+)') { $current.minor = [int]$Matches[1]; continue }
    }
    if ($current) { $hipDevices += [pscustomobject]$current }
}

$wmiDevices = @()
try {
    $wmiDevices = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match 'AMD|Radeon' } | ForEach-Object {
        [pscustomobject]@{
            name = [string]$_.Name
            driver_version = [string]$_.DriverVersion
            pnp_device_id = [string]$_.PNPDeviceID
        }
    })
} catch {}

$devices = @()
if ($hipDevices.Count -gt 0) {
    foreach ($d in $hipDevices) {
        $meta = Get-ArchMetadata $d.gfx
        $wmi = $wmiDevices | Where-Object { $_.name -eq $d.name } | Select-Object -First 1
        $isReference = ($d.gfx -eq 'gfx1201')
        $devices += [pscustomobject][ordered]@{
            index = $d.index
            name = $d.name
            gfx = $d.gfx
            generation = if ($meta) { $meta.generation } else { $null }
            memory = $d.memory
            driver_version = if ($wmi) { $wmi.driver_version } else { $null }
            detection = 'hipInfo'
            current_windows_hip_sdk = if ($meta) { [bool]$meta.current_windows_hip_sdk } else { $null }
            project_tested = $isReference
            project_status = if ($isReference) { 'validated-reference' } elseif ($meta -and $meta.current_windows_hip_sdk) { 'unverified-candidate' } else { 'experimental' }
        }
    }
} else {
    $i = 0
    foreach ($wmi in $wmiDevices) {
        $arch = Get-FallbackArch $wmi.name
        $meta = Get-ArchMetadata $arch
        $isReference = ($arch -eq 'gfx1201')
        $devices += [pscustomobject][ordered]@{
            index = $i++
            name = $wmi.name
            gfx = $arch
            generation = if ($meta) { $meta.generation } else { $null }
            memory = $null
            driver_version = $wmi.driver_version
            detection = 'WMI-fallback'
            current_windows_hip_sdk = if ($meta) { [bool]$meta.current_windows_hip_sdk } else { $null }
            project_tested = $isReference
            project_status = if ($isReference) { 'validated-reference' } elseif ($meta -and $meta.current_windows_hip_sdk) { 'unverified-candidate' } else { 'experimental' }
        }
    }
}

if ($GpuIndex -ge 0) {
    $selected = $devices | Where-Object { $_.index -eq $GpuIndex } | Select-Object -First 1
    if (-not $selected -and $devices.Count -gt 0) {
        throw "GPU index $GpuIndex was not found. Run scripts/gpu-scan.ps1 to list available devices."
    }
} else {
    $selected = $devices | Where-Object { $_.gfx -eq 'gfx1201' } | Select-Object -First 1
    if (-not $selected) { $selected = $devices | Where-Object { $_.project_tested } | Select-Object -First 1 }
    if (-not $selected -and $devices.Count -gt 0) { $selected = $devices[0] }
}

$hipVersion = $null
if ($resolvedHip) { $hipVersion = Split-Path $resolvedHip -Leaf }

$report = [pscustomobject][ordered]@{
    schema = 1
    generated_utc = [DateTime]::UtcNow.ToString('o')
    windows = [Environment]::OSVersion.VersionString
    hip_root = $resolvedHip
    hip_version = $hipVersion
    hip_info = $hipInfoPath
    selected_gpu_index = if ($selected) { $selected.index } else { $null }
    selected_gpu = $selected
    devices = $devices
    notes = 'Without -GpuIndex, gfx1201 is preferred over integrated or other AMD GPUs. Use -GpuIndex to select a specific HIP device.'
}

if ($OutputPath) {
    $parent = Split-Path $OutputPath -Parent
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $report | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $OutputPath
}

if ($AsJson) {
    $report | ConvertTo-Json -Depth 6
} elseif (-not $Quiet) {
    Write-Host 'CUDA for AMD - GPU scanner'
    Write-Host '--------------------------'
    if ($resolvedHip) { Write-Host "HIP root : $resolvedHip" } else { Write-Warning 'HIP SDK not found. Architecture detection is limited to WMI fallback rules.' }
    if ($devices.Count -eq 0) {
        Write-Warning 'No AMD GPU was detected.'
    } else {
        $devices | Format-Table index,name,gfx,generation,detection,current_windows_hip_sdk,project_status -AutoSize
        Write-Host ''
        if ($selected.project_tested) {
            Write-Host '[validated] This is the reference GPU tested by the project.'
        } elseif ($selected.current_windows_hip_sdk) {
            Write-Warning 'This GPU is a current Windows HIP SDK candidate, but this project has not validated it yet.'
        } else {
            Write-Warning 'This GPU is experimental for this project and/or not in the current Windows HIP SDK support set.'
        }
        if ($OutputPath) { Write-Host "Report: $OutputPath" }
    }
}

if ($PassThru) { Write-Output $report }

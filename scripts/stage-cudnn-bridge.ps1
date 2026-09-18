[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TargetDir,
    [string]$BridgeDll,
    [string]$BuildOutputDir,
    [string]$Gcc,
    [string]$Objdump,
    [string]$MiopenDll,
    [switch]$Restore,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$TargetDir = [IO.Path]::GetFullPath($TargetDir)
if (-not (Test-Path $TargetDir)) { throw "Target directory not found: $TargetDir" }

$target = Join-Path $TargetDir 'cudnn64_8.dll'
$backup = Join-Path $TargetDir 'cudnn64_8_nvidia.dll'
$statePath = Join-Path $TargetDir '.cudaamd-cudnn-bridge.json'

if ($Restore) {
    if (-not (Test-Path $backup)) { throw "Original backup not found: $backup" }
    $state = if (Test-Path $statePath) { Get-Content $statePath -Raw | ConvertFrom-Json } else { $null }
    if ($state -and (Test-Path $target) -and -not $Force) {
        $currentHash = (Get-FileHash $target -Algorithm SHA256).Hash
        if ($state.bridge_sha256 -and $currentHash -ne [string]$state.bridge_sha256) {
            throw 'The staged cudnn64_8.dll no longer matches the recorded bridge. Use -Force only if overwriting it is intentional.'
        }
    }
    if (Test-Path $target) { Remove-Item $target -Force }
    Move-Item $backup $target -Force
    if ($state -and $state.original_sha256) {
        $restoredHash = (Get-FileHash $target -Algorithm SHA256).Hash
        if ($restoredHash -ne [string]$state.original_sha256) {
            throw "Restored original hash mismatch: $restoredHash"
        }
    }
    if (Test-Path $statePath) { Remove-Item $statePath -Force }
    Write-Host "Restored original cuDNN: $target"
    return
}

if (-not (Test-Path $target)) { throw "Top-level cuDNN DLL not found: $target" }
if ((Test-Path $backup) -or (Test-Path $statePath)) {
    if (-not $Force) { throw 'A cuDNN bridge staging state already exists. Restore it first.' }
    throw 'Force reinstall over an existing staged state is intentionally unsupported; restore first.'
}

$originalHash = (Get-FileHash $target -Algorithm SHA256).Hash
if (-not $BridgeDll) {
    if (-not $BuildOutputDir) { $BuildOutputDir = Join-Path $repo '.runtime\cudnn-bridge' }
    $buildArgs = @{
        OriginalCudnn = $target
        OutputDir = $BuildOutputDir
    }
    if ($Gcc) { $buildArgs.Gcc = $Gcc }
    if ($Objdump) { $buildArgs.Objdump = $Objdump }
    & (Join-Path $PSScriptRoot 'build-cudnn-bridge.ps1') @buildArgs
    $BridgeDll = Join-Path ([IO.Path]::GetFullPath($BuildOutputDir)) 'cudnn64_8.dll'
}
$BridgeDll = [IO.Path]::GetFullPath($BridgeDll)
if (-not (Test-Path $BridgeDll)) { throw "Bridge DLL not found: $BridgeDll" }
$bridgeHash = (Get-FileHash $BridgeDll -Algorithm SHA256).Hash
if ($bridgeHash -eq $originalHash) { throw 'Bridge DLL is identical to the original DLL; refusing to stage.' }

$resolvedMiopen = $null
if ($MiopenDll) {
    $resolvedMiopen = [IO.Path]::GetFullPath($MiopenDll)
    if (-not (Test-Path $resolvedMiopen)) { throw "MIOpen DLL not found: $resolvedMiopen" }
}

$installed = $false
try {
    Move-Item $target $backup
    Copy-Item $BridgeDll $target -Force
    $installed = $true

    $state = [ordered]@{
        schema = 1
        staged_utc = [DateTime]::UtcNow.ToString('o')
        target_dir = $TargetDir
        original_path = $backup
        original_sha256 = $originalHash
        bridge_source = $BridgeDll
        bridge_sha256 = $bridgeHash
        fallback_module = 'cudnn64_8_nvidia.dll'
        miopen_dll = $resolvedMiopen
        miopen_sha256 = if ($resolvedMiopen) { (Get-FileHash $resolvedMiopen -Algorithm SHA256).Hash } else { $null }
        backend_env = if ($resolvedMiopen) { 'CUDA_AMD_MIOPEN_DLL' } else { $null }
    }
    $state | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $statePath
} catch {
    if ($installed -and (Test-Path $target)) { Remove-Item $target -Force -ErrorAction SilentlyContinue }
    if ((Test-Path $backup) -and -not (Test-Path $target)) {
        Move-Item $backup $target -Force -ErrorAction SilentlyContinue
    }
    throw
}

Write-Host 'Experimental cuDNN -> MIOpen bridge staged'
Write-Host "Target   : $target"
Write-Host "Original : $backup"
Write-Host "Bridge   : $BridgeDll"
if ($resolvedMiopen) {
    Write-Host "MIOpen   : $resolvedMiopen"
    Write-Host "Set CUDA_AMD_MIOPEN_DLL to that path in the test process."
} else {
    Write-Warning 'No -MiopenDll was supplied. cudnnCreate will fail closed unless MIOpen.dll is already discoverable by the process.'
}
Write-Host "State    : $statePath"
Write-Host 'Use -Restore to put the original DLL back.'

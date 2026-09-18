[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TargetDir,
    [string]$ProxyDll,
    [string]$BuildOutputDir,
    [string]$Gcc,
    [string]$Objdump,
    [switch]$Restore,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$TargetDir = [IO.Path]::GetFullPath($TargetDir)
if (-not (Test-Path $TargetDir)) { throw "Target directory not found: $TargetDir" }

$target = Join-Path $TargetDir 'cusolver64_11.dll'
$backup = Join-Path $TargetDir 'cusolver64_11_nvidia.dll'
$statePath = Join-Path $TargetDir '.cudaamd-cusolver-proxy.json'

if ($Restore) {
    if (-not (Test-Path $backup)) { throw "Original backup not found: $backup" }
    $state = if (Test-Path $statePath) { Get-Content $statePath -Raw | ConvertFrom-Json } else { $null }
    if ($state -and (Test-Path $target) -and -not $Force) {
        $currentHash = (Get-FileHash $target -Algorithm SHA256).Hash
        if ($state.proxy_sha256 -and $currentHash -ne [string]$state.proxy_sha256) {
            throw 'The staged cusolver64_11.dll no longer matches the recorded proxy. Use -Force only if overwriting it is intentional.'
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
    Write-Host "Restored original cuSOLVER: $target"
    return
}

if (-not (Test-Path $target)) { throw "Target cuSOLVER DLL not found: $target" }
if ((Test-Path $backup) -or (Test-Path $statePath)) {
    if (-not $Force) { throw 'A cuSOLVER proxy staging state already exists. Restore it first or use -Force after inspecting the target.' }
    throw 'Force reinstall over an existing staged state is intentionally not supported yet; restore first.'
}

$originalHash = (Get-FileHash $target -Algorithm SHA256).Hash
if (-not $ProxyDll) {
    if (-not $BuildOutputDir) { $BuildOutputDir = Join-Path $repo '.runtime\cusolver-proxy' }
    $build = Join-Path $PSScriptRoot 'build-cusolver-proxy.ps1'
    $buildArgs = @{
        OriginalCusolver = $target
        OutputDir = $BuildOutputDir
    }
    if ($Gcc) { $buildArgs.Gcc = $Gcc }
    if ($Objdump) { $buildArgs.Objdump = $Objdump }
    & $build @buildArgs
    $ProxyDll = Join-Path ([IO.Path]::GetFullPath($BuildOutputDir)) 'cusolver64_11.dll'
}
$ProxyDll = [IO.Path]::GetFullPath($ProxyDll)
if (-not (Test-Path $ProxyDll)) { throw "Proxy DLL not found: $ProxyDll" }
$proxyHash = (Get-FileHash $ProxyDll -Algorithm SHA256).Hash
if ($proxyHash -eq $originalHash) { throw 'Proxy DLL is identical to the original DLL; refusing to stage.' }

$installed = $false
try {
    Move-Item $target $backup
    Copy-Item $ProxyDll $target -Force
    $installed = $true

    $state = [ordered]@{
        schema = 1
        staged_utc = [DateTime]::UtcNow.ToString('o')
        target_dir = $TargetDir
        original_path = $backup
        original_sha256 = $originalHash
        proxy_source = $ProxyDll
        proxy_sha256 = $proxyHash
        fallback_module = 'cusolver64_11_nvidia.dll'
    }
    $state | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $statePath
} catch {
    if ($installed -and (Test-Path $target)) { Remove-Item $target -Force -ErrorAction SilentlyContinue }
    if ((Test-Path $backup) -and -not (Test-Path $target)) { Move-Item $backup $target -Force -ErrorAction SilentlyContinue }
    throw
}

Write-Host 'cuSOLVER proxy staged'
Write-Host "Target   : $target"
Write-Host "Original : $backup"
Write-Host "Proxy    : $ProxyDll"
Write-Host "State    : $statePath"
Write-Host 'Use this script again with -Restore to put the original DLL back.'

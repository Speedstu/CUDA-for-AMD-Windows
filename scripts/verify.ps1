[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [switch]$StrictRecoveredHashes
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$manifest = Join-Path $repo 'manifests\recovered-artifacts.sha256'
$configPath = Join-Path $RuntimeRoot 'runtime-config.json'

Write-Host '=== ZLUDA AMD Windows Custom verification ==='
if (Test-Path $configPath) {
    $config = Get-Content $configPath -Raw | ConvertFrom-Json
    Write-Host "profile: $($config.recovered_profile)"
    Write-Host "ZLUDA : $($config.zluda_root)"
    Write-Host "HIP   : $($config.hip_root)"
    Write-Host "Torch : $($config.libtorch_root)"
    Write-Host "overlay: $($config.custom_overlay_root)"
    if ($config.zluda_root -and (Test-Path (Join-Path $config.zluda_root 'zluda.exe'))) { Write-Host '[PASS] zluda.exe' } else { Write-Warning '[MISS] zluda.exe' }
    if ($config.hip_root -and (Test-Path (Join-Path $config.hip_root 'bin'))) { Write-Host '[PASS] HIP bin' } else { Write-Warning '[MISS] HIP bin' }
    if ($config.libtorch_root -and (Test-Path (Join-Path $config.libtorch_root 'lib\torch_cuda.dll'))) { Write-Host '[PASS] LibTorch CUDA' } else { Write-Warning '[MISS] LibTorch CUDA' }
} else {
    Write-Warning 'runtime-config.json not generated yet; run setup.ps1.'
}

Write-Host "`n=== Recovered local artifact hashes ==="
$local = Join-Path $repo 'local-artifacts\custom'
$expected = @{}
Get-Content $manifest | Where-Object { $_ -and -not $_.StartsWith('#') } | ForEach-Object {
    $p = $_ -split '\s+',2
    if ($p.Count -eq 2) { $expected[$p[1].Trim()] = $p[0].Trim().ToUpperInvariant() }
}
foreach ($name in $expected.Keys | Sort-Object) {
    $path = Join-Path $local $name
    if (-not (Test-Path $path)) { Write-Warning "[MISS] $name"; continue }
    $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actual -eq $expected[$name]) { Write-Host "[PASS] $name $actual" }
    else {
        Write-Warning "[FAIL] $name expected=$($expected[$name]) actual=$actual"
        if ($StrictRecoveredHashes) { throw "Hash mismatch: $name" }
    }
}

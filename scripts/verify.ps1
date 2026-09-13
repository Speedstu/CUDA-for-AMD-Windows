[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [switch]$CheckRecoveredArtifacts,
    [switch]$StrictRecoveredHashes
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$manifest = Join-Path $repo 'manifests\recovered-artifacts.sha256'
$configPath = Join-Path $RuntimeRoot 'runtime-config.json'

Write-Host '=== CUDA for AMD verification ==='
if (Test-Path $configPath) {
    $config = Get-Content $configPath -Raw | ConvertFrom-Json
    $profile = if ($config.profile) { $config.profile } else { $config.recovered_profile }
    Write-Host "profile: $profile"
    if ($config.gpu -and $config.gpu.name) {
        Write-Host "GPU   : $($config.gpu.name) / $($config.gpu.arch) [$($config.gpu.project_status)]"
    }
    Write-Host "ZLUDA : $($config.zluda_root)"
    Write-Host "HIP   : $($config.hip_root)"
    if ($config.libtorch_root) { Write-Host "Torch : $($config.libtorch_root)" }
    if ($config.custom_overlay_root) { Write-Host "overlay: $($config.custom_overlay_root)" }

    if ($config.zluda_root -and (Test-Path (Join-Path $config.zluda_root 'zluda.exe'))) { Write-Host '[PASS] zluda.exe' } else { Write-Warning '[MISS] zluda.exe' }
    if ($config.hip_root -and (Test-Path (Join-Path $config.hip_root 'bin'))) { Write-Host '[PASS] HIP bin' } else { Write-Warning '[MISS] HIP bin' }
    if ($config.libtorch_root) {
        if (Test-Path (Join-Path $config.libtorch_root 'lib\torch_cuda.dll')) { Write-Host '[PASS] LibTorch CUDA' } else { Write-Warning '[MISS] LibTorch CUDA' }
    } else {
        Write-Host '[SKIP] LibTorch not configured'
    }

    if ($config.gpu -and -not $config.gpu.tested_reference) {
        Write-Warning 'This GPU has not been validated by the project yet. Please publish a GPU compatibility issue with the scanner report.'
    }
} else {
    Write-Warning 'runtime-config.json not generated yet; run setup.ps1.'
}

if ($CheckRecoveredArtifacts -or $StrictRecoveredHashes) {
    Write-Host "`n=== Optional recovered reference artifact hashes ==="
    $local = Join-Path $repo 'local-artifacts\custom'
    if (-not (Test-Path $local)) {
        if ($StrictRecoveredHashes) { throw 'Recovered local artifacts are not present.' }
        Write-Host '[SKIP] local-artifacts/custom is not present in this checkout.'
        return
    }

    $expected = @{}
    Get-Content $manifest | Where-Object { $_ -and -not $_.StartsWith('#') } | ForEach-Object {
        $p = $_ -split '\s+',2
        if ($p.Count -eq 2) { $expected[$p[1].Trim()] = $p[0].Trim().ToUpperInvariant() }
    }
    foreach ($name in $expected.Keys | Sort-Object) {
        $path = Join-Path $local $name
        if (-not (Test-Path $path)) {
            Write-Warning "[MISS] $name"
            if ($StrictRecoveredHashes) { throw "Missing recovered artifact: $name" }
            continue
        }
        $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($actual -eq $expected[$name]) { Write-Host "[PASS] $name $actual" }
        else {
            Write-Warning "[FAIL] $name expected=$($expected[$name]) actual=$actual"
            if ($StrictRecoveredHashes) { throw "Hash mismatch: $name" }
        }
    }
}

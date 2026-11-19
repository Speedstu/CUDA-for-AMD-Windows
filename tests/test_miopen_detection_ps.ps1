$ErrorActionPreference = 'Stop'
$source = Join-Path (Split-Path $PSScriptRoot -Parent) 'scripts\miopen-detection.ps1'
. $source
$root = Join-Path ([IO.Path]::GetTempPath()) ('cudaamd-miopen-regression-' + [Guid]::NewGuid().ToString('N'))
$hip = Join-Path $root 'hip'
$zluda = Join-Path $root 'zluda'
$venv = Join-Path $root 'venv'
$targetDir = Join-Path $venv 'Scripts'
$python = Join-Path $targetDir 'python.exe'
$backend = Join-Path $hip 'bin\MIOpen.dll'
$bridgeBackend = Join-Path $root 'outside\alternate-MIOpen.dll'
$bridgeState = Join-Path $venv 'Lib\site-packages\torch\lib\.cudaamd-cudnn-bridge.json'
New-Item -ItemType Directory -Force $targetDir,$zluda,(Split-Path $backend -Parent),(Split-Path $bridgeBackend -Parent),(Split-Path $bridgeState -Parent) | Out-Null
function Check([bool]$ok, [string]$reason) {
    if (-not $ok) { throw "MIOpen shared detection regression: $reason" }
}
function Resolve-Test([string]$explicit = '') {
    Get-CudaAmdMiopenState -HipRoot $hip -ZludaRoot $zluda -TargetDir $targetDir -ProgramPath $python -InheritedHipPath '' -ConfiguredMiopenDll $explicit
}
try {
    $m = Resolve-Test
    Check (-not $m.miopen_present -and -not $m.bridge_ready -and $m.source -eq 'missing') 'missing must fail closed'

    Set-Content -LiteralPath $backend -Value 'fixture'
    $m = Resolve-Test
    Check ($m.miopen_present -and -not $m.bridge_ready -and $m.source -eq 'existing_backend_file') 'HIP bin DLL must not impersonate staged bridge'
    Check ($m.miopen_path -eq [IO.Path]::GetFullPath($backend)) 'HIP backend path must be reported'

    Set-Content -LiteralPath $bridgeBackend -Value 'fixture'
    $m = Resolve-Test $bridgeBackend
    Check ($m.bridge_ready -and $m.source -eq 'explicit_env') 'explicit bridge path must take precedence'
    Check ($m.miopen_path -eq [IO.Path]::GetFullPath($bridgeBackend)) 'explicit bridge path must be retained'

    $m = Resolve-Test (Join-Path $root 'missing-file.dll')
    Check (-not $m.bridge_ready -and $m.miopen_present) 'stale environment variable must not enable bridge'

    $json = @{miopen_dll = $bridgeBackend} | ConvertTo-Json
    Set-Content -LiteralPath $bridgeState -Value $json
    $m = Resolve-Test
    Check ($m.bridge_ready -and $m.source -eq 'staged_bridge') 'venv torch/lib staged bridge must be recognized'
    Check ($m.miopen_path -eq [IO.Path]::GetFullPath($bridgeBackend)) 'staged bridge path must be retained'

    Set-Content -LiteralPath $bridgeState -Value '{invalid-json'
    $m = Resolve-Test
    Check ($m.miopen_present -and -not $m.bridge_ready) 'invalid JSON must not enable bridge'

    Remove-Item $backend -Force
    $m = Resolve-Test
    Check (-not $m.miopen_present -and -not $m.bridge_ready) 'no available backend must fail closed'

    $localState = Join-Path $targetDir '.cudaamd-cudnn-bridge.json'
    Set-Content -LiteralPath $localState -Value $json
    $m = Resolve-Test
    Check ($m.bridge_ready -and $m.source -eq 'staged_bridge') 'local staging state must be recognized'
    Write-Output 'PASS: common MIOpen resolution missing/HIP/env/state/invalid-path test matrix'
} finally {
    Remove-Item $root -Recurse -Force -ErrorAction SilentlyContinue
}

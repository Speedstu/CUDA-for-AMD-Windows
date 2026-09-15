[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [string]$VelocityRoot,
    [string]$PythonExe,
    [string]$Source,
    [string]$Backend,
    [int]$Agents = 512,
    [int]$Threads = 8,
    [int]$Rollout = 16,
    [int]$Minibatch = 4096,
    [int]$SmokeUpdates = 1,
    [int]$TimeoutSeconds = 180,
    [string]$ReportPath
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
if (-not $VelocityRoot) { $VelocityRoot = Join-Path (Split-Path $repo -Parent) 'VelocityRL' }
$VelocityRoot = [System.IO.Path]::GetFullPath($VelocityRoot)
if (-not $PythonExe) { $PythonExe = Join-Path $VelocityRoot '.venv-yukan-zluda\Scripts\python.exe' }
if (-not $Source) { $Source = Join-Path $VelocityRoot 'runs\yukan_velocity\source_cpp_latest.pt' }
if (-not $Backend) { $Backend = Join-Path $VelocityRoot 'build-rs-yukan-pressure\Release\velocityrl_rocketsim_backend.dll' }
if (-not $ReportPath) { $ReportPath = Join-Path $RuntimeRoot 'velocityrl-smoke.json' }

$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
$trainScript = Join-Path $VelocityRoot 'benchmarks\train_yukan_velocity.py'
foreach ($required in @($configPath, $PythonExe, $Source, $Backend, $trainScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file not found: $required" }
}
if ($Agents -lt 4 -or $Agents % 4) { throw '-Agents must be divisible by 4.' }
if ($SmokeUpdates -lt 1) { throw '-SmokeUpdates must be at least 1.' }

$config = Get-Content $configPath -Raw | ConvertFrom-Json
$zluda = [string]$config.zluda_root
$hip = [string]$config.hip_root
$launcher = Join-Path $zluda 'zluda.exe'
if (-not (Test-Path $launcher)) { throw "Missing zluda.exe: $launcher" }
if (-not $hip -or -not (Test-Path (Join-Path $hip 'bin'))) { throw "HIP SDK bin directory is missing: $hip" }

function Quote-ProcessArgument {
    param([string]$Value)
    if ($null -eq $Value) { return '""' }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Set-ProcessEnvironment {
    param([System.Diagnostics.ProcessStartInfo]$Info, [string]$Name, [string]$Value)
    if ($Info.PSObject.Properties.Name -contains 'Environment') {
        $Info.Environment[$Name] = $Value
    } else {
        $Info.EnvironmentVariables[$Name] = $Value
    }
}

function Stop-ProcessTree {
    param([System.Diagnostics.Process]$Process)
    try {
        $taskkill = Join-Path $env:SystemRoot 'System32\taskkill.exe'
        if (Test-Path $taskkill) { & $taskkill /PID $Process.Id /T /F 2>$null | Out-Null }
        else { $Process.Kill() }
    } catch { try { $Process.Kill() } catch {} }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$runDir = Join-Path $RuntimeRoot "velocityrl-smoke\$stamp"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$arguments = @(
    '--', $PythonExe, '-u', $trainScript,
    '--source', $Source,
    '--run', $runDir,
    '--backend', $Backend,
    '--agents', [string]$Agents,
    '--threads', [string]$Threads,
    '--rollout', [string]$Rollout,
    '--epochs', '1',
    '--minibatch', [string]$Minibatch,
    '--train-mode', 'head',
    '--historical-fraction', '0',
    '--league-fraction', '0',
    '--target-kl', '0.002',
    '--max-lr', '0.0001',
    '--save-every', '1',
    '--version-every', '1000000000',
    '--smoke-updates', [string]$SmokeUpdates
)

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $launcher
$psi.Arguments = ($arguments | ForEach-Object { Quote-ProcessArgument ([string]$_) }) -join ' '
$psi.WorkingDirectory = $VelocityRoot
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true
Set-ProcessEnvironment $psi 'HIP_PATH' $hip
Set-ProcessEnvironment $psi 'ZLUDA_CC' $(if ($config.zluda_cc) { [string]$config.zluda_cc } else { '8.6' })
Set-ProcessEnvironment $psi 'ROCBLAS_TENSILE_LIBPATH' (Join-Path $hip 'bin\rocblas\library')
Set-ProcessEnvironment $psi 'HIPBLASLT_TENSILE_LIBPATH' (Join-Path $hip 'bin\hipblaslt\library')
Set-ProcessEnvironment $psi 'PATH' "$hip\bin;$zluda;$env:PATH"
Set-ProcessEnvironment $psi 'PYTHONPATH' $VelocityRoot
if ($config.torch_allow_tf32_cublas_override) {
    Set-ProcessEnvironment $psi 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE' ([string]$config.torch_allow_tf32_cublas_override)
}

Write-Host '=== VelocityRL PPO integration smoke ==='
Write-Host "Runtime : $RuntimeRoot"
Write-Host "Velocity: $VelocityRoot"
Write-Host "Agents  : $Agents / rollout $Rollout / updates $SmokeUpdates"
Write-Host "Run dir : $runDir"

$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
[void]$p.Start()
$sw = [Diagnostics.Stopwatch]::StartNew()
while (-not $p.HasExited -and $sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) { Start-Sleep -Milliseconds 250 }
$timedOut = -not $p.HasExited
if ($timedOut) {
    Stop-ProcessTree $p
    try { $p.WaitForExit() } catch {}
}
$stdout = $p.StandardOutput.ReadToEnd()
$stderr = $p.StandardError.ReadToEnd()
if ($stdout) { Write-Host $stdout.TrimEnd() }
if ($stderr) { Write-Warning $stderr.TrimEnd() }

$statusPath = Join-Path $runDir 'status.json'
$latestPath = Join-Path $runDir 'latest.pt'
$status = $null
if (Test-Path $statusPath) {
    try { $status = Get-Content $statusPath -Raw | ConvertFrom-Json } catch {}
}
$exitCode = if ($timedOut) { $null } else { $p.ExitCode }
$metricsOk = $false
if ($status) {
    $sps = [double]$status.sps
    $entropy = [double]$status.entropy
    $kl = [double]$status.kl
    $metricsOk = ([double]::IsNaN($sps) -eq $false) -and ([double]::IsInfinity($sps) -eq $false) -and $sps -gt 0 -and
                 ([double]::IsNaN($entropy) -eq $false) -and ([double]::IsInfinity($entropy) -eq $false) -and
                 ([double]::IsNaN($kl) -eq $false) -and ([double]::IsInfinity($kl) -eq $false)
}
$passed = (-not $timedOut) -and ($exitCode -eq 0) -and $metricsOk -and (Test-Path $latestPath)

$result = [ordered]@{
    schema = 1
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    passed = [bool]$passed
    timed_out = [bool]$timedOut
    timeout_seconds = $TimeoutSeconds
    process_exit = $exitCode
    runtime_root = $RuntimeRoot
    velocity_root = $VelocityRoot
    run_dir = $runDir
    python = $PythonExe
    source = $Source
    backend = $Backend
    agents = $Agents
    rollout = $Rollout
    smoke_updates = $SmokeUpdates
    status = $status
    checkpoint_written = [bool](Test-Path $latestPath)
    stdout = $stdout.TrimEnd()
    stderr = $stderr.TrimEnd()
}
New-Item -ItemType Directory -Force -Path (Split-Path $ReportPath -Parent) | Out-Null
$result | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $ReportPath
Write-Host "Report: $ReportPath"
if ($passed) {
    Write-Host '[PASS] VelocityRL completed real PPO rollout + learning + optimizer work through this repository runtime.'
} else {
    throw 'VelocityRL PPO integration smoke failed.'
}

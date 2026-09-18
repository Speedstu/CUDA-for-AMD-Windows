[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Program,
    [string[]]$ProgramArgs = @(),
    [string]$RuntimeRoot,
    [int]$TimeoutSeconds = 120,
    [string]$OutputRoot,
    [switch]$NoStage,
    [switch]$AcknowledgeGpuResetRisk
)

$ErrorActionPreference = 'Stop'
if ($TimeoutSeconds -lt 1) { throw 'TimeoutSeconds must be at least 1.' }
if (-not $AcknowledgeGpuResetRisk) {
    throw 'Trace mode can execute the application GPU workload and a bad kernel can still reset/freeze the GPU after the process is killed. Re-run with -AcknowledgeGpuResetRisk only on a machine where a forced reboot is acceptable.'
}

$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$Program = (Resolve-Path $Program).Path
$targetDir = Split-Path $Program -Parent

$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
if (-not (Test-Path $configPath)) { throw "Missing runtime config: $configPath" }
$config = Get-Content $configPath -Raw | ConvertFrom-Json
$zluda = [string]$config.zluda_root
$hip = [string]$config.hip_root
$launcher = Join-Path $zluda 'zluda.exe'
if (-not (Test-Path $launcher)) { throw "Missing ZLUDA launcher: $launcher" }
if (-not $hip -or -not (Test-Path (Join-Path $hip 'bin'))) { throw "Missing HIP bin directory: $hip" }

if (-not $NoStage) {
    & (Join-Path $PSScriptRoot 'stage-runtime.ps1') -TargetDir $targetDir -RuntimeRoot $RuntimeRoot
}

if (-not $OutputRoot) { $OutputRoot = Join-Path $RuntimeRoot 'traces' }
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

function Get-Sha256OrNull([string]$Path) {
    if (-not $Path -or -not (Test-Path $Path -PathType Leaf)) { return $null }
    try { return (Get-FileHash $Path -Algorithm SHA256 -ErrorAction Stop).Hash }
    catch { return $null }
}

function Quote-Arg([string]$Value) {
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
        if (Test-Path $taskkill) {
            & $taskkill /PID $Process.Id /T /F 2>$null | Out-Null
        } else {
            $Process.Kill()
        }
    } catch {
        try { $Process.Kill() } catch {}
    }
}

$traceBase = Join-Path $env:TEMP 'zluda'
$before = @{}
if (Test-Path $traceBase) {
    Get-ChildItem $traceBase -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $before[$_.FullName] = $_.LastWriteTimeUtc
    }
}

$timestamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
$runDir = Join-Path $OutputRoot ("zluda-trace-" + $timestamp)
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $launcher
$parts = @('--zluda-trace', '--', $Program) + @($ProgramArgs)
$psi.Arguments = ($parts | ForEach-Object { Quote-Arg ([string]$_) }) -join ' '
$psi.WorkingDirectory = $targetDir
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true

Set-ProcessEnvironment $psi 'HIP_PATH' $hip
Set-ProcessEnvironment $psi 'ZLUDA_CC' $(if ($config.zluda_cc) { [string]$config.zluda_cc } else { '8.6' })
$rocblasLib = Join-Path $hip 'bin\rocblas\library'
if (Test-Path $rocblasLib) { Set-ProcessEnvironment $psi 'ROCBLAS_TENSILE_LIBPATH' $rocblasLib }
$hipblasltLib = Join-Path $hip 'bin\hipblaslt\library'
if ($config.gpu -and $config.gpu.arch) {
    $archLib = Join-Path $hipblasltLib ([string]$config.gpu.arch)
    if (Test-Path $archLib) { $hipblasltLib = $archLib }
}
if (Test-Path $hipblasltLib) { Set-ProcessEnvironment $psi 'HIPBLASLT_TENSILE_LIBPATH' $hipblasltLib }
Set-ProcessEnvironment $psi 'PATH' "$targetDir;$zluda;$hip\bin;$env:PATH"
if ($config.gpu -and $null -ne $config.gpu.index) {
    Set-ProcessEnvironment $psi 'HIP_VISIBLE_DEVICES' ([string]$config.gpu.index)
    Set-ProcessEnvironment $psi 'ROCR_VISIBLE_DEVICES' ([string]$config.gpu.index)
}

Write-Host '=== ZLUDA trace capture ==='
Write-Host "Program : $Program"
Write-Host ("Timeout : {0}s" -f $TimeoutSeconds)
Write-Host "Trace   : $traceBase"
Write-Warning 'A process timeout cannot guarantee recovery from an already-faulted GPU kernel.'

$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
[void]$p.Start()
$stdoutTask = $p.StandardOutput.ReadToEndAsync()
$stderrTask = $p.StandardError.ReadToEndAsync()
$sw = [Diagnostics.Stopwatch]::StartNew()
while (-not $p.HasExited -and $sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
    Start-Sleep -Milliseconds 100
}

$timedOut = -not $p.HasExited
if ($timedOut) {
    Stop-ProcessTree $p
    try { $p.WaitForExit() } catch {}
}
try { $stdout = $stdoutTask.GetAwaiter().GetResult() } catch { $stdout = '' }
try { $stderr = $stderrTask.GetAwaiter().GetResult() } catch { $stderr = '' }
$exitCode = if ($timedOut) { $null } else { $p.ExitCode }

$stdoutPath = Join-Path $runDir 'stdout.txt'
$stderrPath = Join-Path $runDir 'stderr.txt'
$stdout | Set-Content -Encoding UTF8 $stdoutPath
$stderr | Set-Content -Encoding UTF8 $stderrPath

$newTraceDirs = @()
if (Test-Path $traceBase) {
    $newTraceDirs = @(Get-ChildItem $traceBase -Directory -ErrorAction SilentlyContinue | Where-Object {
        (-not $before.ContainsKey($_.FullName)) -or $_.LastWriteTimeUtc -gt $before[$_.FullName]
    } | Sort-Object LastWriteTimeUtc)
}

$copied = @()
foreach ($dir in $newTraceDirs) {
    $dest = Join-Path $runDir $dir.Name
    Copy-Item $dir.FullName $dest -Recurse -Force
    $copied += $dest
}

$traceFiles = @()
foreach ($dir in $copied) {
    $traceFiles += @(Get-ChildItem $dir -File -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        [pscustomobject][ordered]@{
            path = $_.FullName.Substring($runDir.Length).TrimStart('\')
            size = $_.Length
            sha256 = Get-Sha256OrNull $_.FullName
        }
    })
}

$projectRevision = $null
try {
    $git = Get-Command git.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $git) { $git = Get-Command git -ErrorAction SilentlyContinue | Select-Object -First 1 }
    if ($git) {
        $rev = (& $git.Source -C $repo rev-parse HEAD 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $rev) { $projectRevision = ([string]$rev).Trim() }
    }
} catch {}

$report = [ordered]@{
    schema = 1
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    project_revision = $projectRevision
    mode = 'zluda_trace'
    gpu_reset_risk_acknowledged = $true
    program = $Program
    program_args = @($ProgramArgs)
    program_sha256 = Get-Sha256OrNull $Program
    runtime_root = $RuntimeRoot
    runtime_hashes = [ordered]@{
        launcher_sha256 = Get-Sha256OrNull $launcher
        nvcuda_sha256 = Get-Sha256OrNull (Join-Path $zluda 'nvcuda.dll')
        runtime_config_sha256 = Get-Sha256OrNull $configPath
    }
    gpu = if ($config.gpu) { $config.gpu } else { $null }
    zluda_cc = if ($config.zluda_cc) { [string]$config.zluda_cc } else { '8.6' }
    timed_out = [bool]$timedOut
    timeout_seconds = $TimeoutSeconds
    process_exit = $exitCode
    elapsed_ms = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
    trace_source_root = $traceBase
    captured_trace_directories = @($copied | ForEach-Object { Split-Path $_ -Leaf })
    trace_files = $traceFiles
    stdout_sha256 = Get-Sha256OrNull $stdoutPath
    stderr_sha256 = Get-Sha256OrNull $stderrPath
    warning = 'Timeout/kill only terminates the process tree; it cannot guarantee recovery from a GPU/driver hard lock.'
}

$reportPath = Join-Path $runDir 'trace-report.json'
$report | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $reportPath

$zipPath = $runDir + '.zip'
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path (Join-Path $runDir '*') -DestinationPath $zipPath -CompressionLevel Optimal

Write-Host "Exit    : $(if ($timedOut) { 'TIMEOUT' } else { $exitCode })"
Write-Host "Captured: $($copied.Count) trace directorie(s)"
Write-Host "Report  : $reportPath"
Write-Host "Archive : $zipPath"
if ($copied.Count -eq 0) {
    Write-Warning 'No new %TEMP%\zluda trace directory was detected. Check stderr and confirm that this ZLUDA build includes trace support.'
}

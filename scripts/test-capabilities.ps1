[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [string]$PythonExe,
    [int]$TimeoutSeconds = 25,
    [int]$RetryTimeoutSeconds = 180,
    [string[]]$Tests,
    [string]$ReportPath,
    [switch]$Strict
)

$ErrorActionPreference = 'Stop'
if ($TimeoutSeconds -lt 1) { throw 'TimeoutSeconds must be at least 1.' }
if ($RetryTimeoutSeconds -lt 0) { throw 'RetryTimeoutSeconds cannot be negative.' }
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
if (-not $ReportPath) { $ReportPath = Join-Path $RuntimeRoot 'capability-test.json' }
$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
$probePath = Join-Path $PSScriptRoot 'capability_probe.py'
if (-not (Test-Path $configPath)) { throw "Missing runtime config: $configPath" }
if (-not (Test-Path $probePath)) { throw "Missing capability probe: $probePath" }
$config = Get-Content $configPath -Raw | ConvertFrom-Json
$zluda = [string]$config.zluda_root
$hip = [string]$config.hip_root
$launcher = Join-Path $zluda 'zluda.exe'
if (-not (Test-Path $launcher)) { throw "Missing zluda.exe: $launcher" }
if (-not $hip -or -not (Test-Path (Join-Path $hip 'bin'))) { throw "HIP SDK bin directory is missing: $hip" }

if (-not $PythonExe) {
    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $cmd) { $cmd = Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1 }
    if ($cmd) { $PythonExe = $cmd.Source }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) { throw 'Python not found. Pass -PythonExe <cuda-pytorch-venv\Scripts\python.exe>.' }
$torchCheck = & $PythonExe -c "import torch; print(torch.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) { throw "PyTorch import failed in $PythonExe`n$($torchCheck | Out-String)" }

if (-not $Tests -or $Tests.Count -eq 0) {
    $Tests = @(& $PythonExe $probePath --list-tests)
    if ($LASTEXITCODE -ne 0 -or $Tests.Count -eq 0) { throw 'Could not enumerate capability probes.' }
}

function Quote-Arg([string]$Value) {
    if ($null -eq $Value) { return '""' }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Set-Env([System.Diagnostics.ProcessStartInfo]$Info, [string]$Name, [string]$Value) {
    if ($Info.PSObject.Properties.Name -contains 'Environment') { $Info.Environment[$Name] = $Value }
    else { $Info.EnvironmentVariables[$Name] = $Value }
}

function Stop-Tree([System.Diagnostics.Process]$Process) {
    try {
        $taskkill = Join-Path $env:SystemRoot 'System32\taskkill.exe'
        if (Test-Path $taskkill) { & $taskkill /PID $Process.Id /T /F 2>$null | Out-Null }
        else { $Process.Kill() }
    } catch { try { $Process.Kill() } catch {} }
}

function Invoke-Probe([string]$Name, [int]$ProbeTimeoutSeconds) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $launcher
    $psi.Arguments = '-- ' + (Quote-Arg $PythonExe) + ' ' + (Quote-Arg $probePath) + ' --test ' + $Name
    $psi.WorkingDirectory = $zluda
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    Set-Env $psi 'HIP_PATH' $hip
    Set-Env $psi 'ZLUDA_CC' $(if ($config.zluda_cc) { [string]$config.zluda_cc } else { '8.6' })
    $rocblasLib = Join-Path $hip 'bin\rocblas\library'
    if (Test-Path $rocblasLib) { Set-Env $psi 'ROCBLAS_TENSILE_LIBPATH' $rocblasLib }
    $hipblasltLib = Join-Path $hip 'bin\hipblaslt\library'
    if ($config.gpu -and $config.gpu.arch) {
        $archLib = Join-Path $hipblasltLib ([string]$config.gpu.arch)
        if (Test-Path $archLib) { $hipblasltLib = $archLib }
    }
    if (Test-Path $hipblasltLib) { Set-Env $psi 'HIPBLASLT_TENSILE_LIBPATH' $hipblasltLib }
    Set-Env $psi 'PATH' "$hip\bin;$zluda;$env:PATH"
    if ($config.gpu -and $null -ne $config.gpu.index) {
        Set-Env $psi 'HIP_VISIBLE_DEVICES' ([string]$config.gpu.index)
        Set-Env $psi 'ROCR_VISIBLE_DEVICES' ([string]$config.gpu.index)
    }
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    [void]$p.Start()
    # Drain redirected pipes while the probe runs. Waiting until process exit
    # can deadlock a noisy CUDA kernel once the Windows pipe buffer fills.
    $stdoutTask = $p.StandardOutput.ReadToEndAsync()
    $stderrTask = $p.StandardError.ReadToEndAsync()
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while (-not $p.HasExited -and $sw.Elapsed.TotalSeconds -lt $ProbeTimeoutSeconds) { Start-Sleep -Milliseconds 100 }
    $timedOut = -not $p.HasExited
    if ($timedOut) { Stop-Tree $p; try { $p.WaitForExit() } catch {} }
    try { $stdout = $stdoutTask.GetAwaiter().GetResult() } catch { $stdout = '' }
    try { $stderr = $stderrTask.GetAwaiter().GetResult() } catch { $stderr = '' }
    $parsed = $null
    # Prefer the marked stderr copy so device-side printf cannot interleave
    # with the probe's machine-readable JSON.
    $markerLines = @($stderr -split "`r?`n" | Where-Object { $_ -like 'CUDAAMD_RESULT:*' })
    for ($i = $markerLines.Count - 1; $i -ge 0; $i--) {
        try {
            $json = $markerLines[$i].Substring('CUDAAMD_RESULT:'.Length)
            $candidate = $json | ConvertFrom-Json -ErrorAction Stop
            if ($candidate.schema -eq 1 -and $candidate.test -eq $Name) { $parsed = $candidate; break }
        } catch {}
    }
    if (-not $parsed) {
        $lines = @($stdout -split "`r?`n" | Where-Object { $_.Trim() })
        for ($i = $lines.Count - 1; $i -ge 0; $i--) {
            try {
                $candidate = $lines[$i] | ConvertFrom-Json -ErrorAction Stop
                if ($candidate.schema -eq 1 -and $candidate.test -eq $Name) { $parsed = $candidate; break }
            } catch {}
        }
    }
    # Some failing CUDA wrappers emit a useful result and then hang or crash
    # during process teardown. Preserve the parsed capability result, but do
    # not count a numerically-correct result as a clean PASS unless the child
    # process also exits normally.
    $exitCode = if ($timedOut) { $null } else { $p.ExitCode }
    # Some legacy CUDA libraries abort the child process directly instead of
    # surfacing a Python exception when the selected cubin has no translatable
    # PTX implementation. Infer only this explicit CUDA unsupported condition;
    # arbitrary non-zero exits remain errors.
    $combinedOutput = ($stdout + "`n" + $stderr).ToLowerInvariant()
    $inferredUnsupported = [bool](-not $timedOut -and -not $parsed -and $combinedOutput.Contains('no kernel image is available'))
    $parsedStatus = if ($parsed) { [string]$parsed.status } elseif ($inferredUnsupported) { 'unsupported' } else { $null }
    $expectedExit = switch ($parsedStatus) {
        'pass'        { 0 }
        'incorrect'   { 2 }
        'error'       { 2 }
        'unsupported' { 3 }
        'unavailable' { 4 }
        default       { $null }
    }
    $unexpectedExit = [bool](
        -not $timedOut -and $parsed -and $null -ne $expectedExit -and $exitCode -ne $expectedExit
    )
    $status = if ($timedOut -and $parsed) {
        'hang_after_result'
    } elseif ($timedOut) {
        'timeout'
    } elseif ($unexpectedExit) {
        'crash_after_result'
    } elseif ($parsed -or $inferredUnsupported) {
        $parsedStatus
    } else {
        'error'
    }
    [pscustomobject][ordered]@{
        test = $Name
        status = $status
        capability_status = $parsedStatus
        elapsed_ms = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
        timed_out = [bool]$timedOut
        process_hung_after_result = [bool]($timedOut -and $parsed)
        process_crashed_after_result = [bool]$unexpectedExit
        process_exit = $exitCode
        result = $parsed
        stdout = $stdout.TrimEnd()
        stderr = $stderr.TrimEnd()
    }
}

Write-Host '=== CUDA-on-AMD capability matrix ==='
Write-Host "Runtime : $RuntimeRoot"
Write-Host "ZLUDA   : $zluda"
Write-Host "HIP     : $hip"
Write-Host "Python  : $PythonExe"
Write-Host "PyTorch : $(($torchCheck | Out-String).Trim())"
Write-Host "Timeout : ${TimeoutSeconds}s initial / ${RetryTimeoutSeconds}s retry-on-timeout"
Write-Host ''

$results = @()
foreach ($name in $Tests) {
    Write-Host ("{0,-24}" -f $name) -NoNewline
    $entry = Invoke-Probe $name $TimeoutSeconds
    if ($entry.status -eq 'timeout' -and $RetryTimeoutSeconds -gt $TimeoutSeconds) {
        $initialAttempt = [ordered]@{
            status = $entry.status
            elapsed_ms = $entry.elapsed_ms
            timed_out = $entry.timed_out
            process_exit = $entry.process_exit
        }
        Write-Host " retry(${RetryTimeoutSeconds}s)" -NoNewline -ForegroundColor DarkYellow
        $entry = Invoke-Probe $name $RetryTimeoutSeconds
        $entry | Add-Member -NotePropertyName retried_after_timeout -NotePropertyValue $true
        $entry | Add-Member -NotePropertyName initial_attempt -NotePropertyValue $initialAttempt
    } else {
        $entry | Add-Member -NotePropertyName retried_after_timeout -NotePropertyValue $false
        $entry | Add-Member -NotePropertyName initial_attempt -NotePropertyValue $null
    }
    $results += $entry
    switch ($entry.status) {
        'pass'        { Write-Host ' PASS' -ForegroundColor Green }
        'unsupported' { Write-Host ' UNSUPPORTED' -ForegroundColor Yellow }
        'incorrect'   { Write-Host ' INCORRECT' -ForegroundColor Red }
        'timeout'            { Write-Host ' TIMEOUT' -ForegroundColor Red }
        'hang_after_result'  { Write-Host ' HANG_AFTER_RESULT' -ForegroundColor Red }
        'crash_after_result' { Write-Host ' CRASH_AFTER_RESULT' -ForegroundColor Red }
        'unavailable'        { Write-Host ' UNAVAILABLE' -ForegroundColor Yellow }
        default              { Write-Host (" {0}" -f $entry.status.ToUpperInvariant()) -ForegroundColor Red }
    }
}

$passed = @($results | Where-Object status -eq 'pass')
$unsupported = @($results | Where-Object { $_.status -in @('unsupported','unavailable') })
$incorrect = @($results | Where-Object status -eq 'incorrect')
$timeouts = @($results | Where-Object status -eq 'timeout')
$processHangs = @($results | Where-Object { $_.timed_out })
$processCrashes = @($results | Where-Object { $_.process_crashed_after_result })
$errors = @($results | Where-Object { $_.status -in @('error','hang_after_result','crash_after_result') })
$retried = @($results | Where-Object { $_.retried_after_timeout })
$total = $results.Count
$score = if ($total) { [math]::Round(100.0 * $passed.Count / $total, 1) } else { 0.0 }

$report = [ordered]@{
    schema = 1
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    profile = if ($config.profile) { [string]$config.profile } elseif ($config.recovered_profile) { [string]$config.recovered_profile } else { $null }
    runtime_root = $RuntimeRoot
    zluda_root = $zluda
    hip_root = $hip
    python = $PythonExe
    pytorch = ($torchCheck | Out-String).Trim()
    gpu = if ($config.gpu) { $config.gpu } else { $null }
    initial_timeout_seconds = $TimeoutSeconds
    retry_timeout_seconds = $RetryTimeoutSeconds
    timeout_retries = $retried.Count
    tested_capabilities = $total
    passed = $passed.Count
    safe_refusals = $unsupported.Count
    incorrect = $incorrect.Count
    timeouts = $timeouts.Count
    process_hangs = $processHangs.Count
    process_crashes = $processCrashes.Count
    errors = $errors.Count
    tested_capability_score_percent = $score
    all_tested_capabilities_pass = [bool]($passed.Count -eq $total)
    no_silent_corruption = [bool]($incorrect.Count -eq 0)
    no_hangs = [bool]($processHangs.Count -eq 0)
    no_process_crashes = [bool]($processCrashes.Count -eq 0)
    results = $results
}
New-Item -ItemType Directory -Force -Path (Split-Path $ReportPath -Parent) | Out-Null
$report | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $ReportPath

Write-Host ''
Write-Host "Passed      : $($passed.Count)/$total"
Write-Host "Unsupported : $($unsupported.Count)"
Write-Host "Incorrect   : $($incorrect.Count)"
Write-Host "Timeouts    : $($timeouts.Count)"
Write-Host "Proc hangs  : $($processHangs.Count)"
Write-Host "Proc crashes: $($processCrashes.Count)"
Write-Host "Errors      : $($errors.Count)"
Write-Host "Retried     : $($retried.Count)"
Write-Host "Tested capability score: ${score}%"
Write-Host "Report: $ReportPath"
if ($incorrect.Count -gt 0) { Write-Warning 'At least one tested capability returned numerically incorrect output.' }
if ($processHangs.Count -gt 0) { Write-Warning 'At least one probe process had to be killed after reaching the timeout.' }
if ($processCrashes.Count -gt 0) { Write-Warning 'At least one probe returned a result but the process crashed during teardown.' }
if ($Strict -and ($passed.Count -ne $total)) { throw 'Not every tested CUDA-facing capability passed cleanly.' }

[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [string]$PythonExe,
    [int]$TimeoutSeconds = 30,
    [string]$ReportPath,
    [switch]$Strict
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
if (-not $ReportPath) { $ReportPath = Join-Path $RuntimeRoot 'functional-test.json' }
$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
$probePath = Join-Path $PSScriptRoot 'functional_probe.py'

if (-not (Test-Path $configPath)) { throw "Missing runtime config: $configPath. Run install.ps1 or setup.ps1 first." }
if (-not (Test-Path $probePath)) { throw "Missing functional probe: $probePath" }
$config = Get-Content $configPath -Raw | ConvertFrom-Json
$zluda = [string]$config.zluda_root
$hip = [string]$config.hip_root
$launcher = Join-Path $zluda 'zluda.exe'
if (-not (Test-Path $launcher)) { throw "Missing zluda.exe: $launcher" }

if (-not $PythonExe) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $pythonCommand) { $pythonCommand = Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1 }
    if ($pythonCommand) { $PythonExe = $pythonCommand.Source }
}

function Write-ReportAndMaybeFail {
    param([hashtable]$Data, [string]$Failure)
    New-Item -ItemType Directory -Force -Path (Split-Path $ReportPath -Parent) | Out-Null
    $Data | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $ReportPath
    Write-Host "Report: $ReportPath"
    if ($Failure -and $Strict) { throw $Failure }
}

if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    $result = [ordered]@{
        schema = 1
        generated_utc = (Get-Date).ToUniversalTime().ToString('o')
        available = $false
        correctness_ok = $false
        full_support = $false
        reason = 'python_not_found'
        tests = @()
    }
    Write-Warning 'Functional validation was not run because Python was not found. Runtime smoke success alone is not numerical validation.'
    Write-ReportAndMaybeFail -Data $result -Failure 'Functional validation unavailable: Python not found.'
    return
}

# Confirm that this Python has PyTorch before going through ZLUDA.
$torchCheck = & $PythonExe -c "import torch; print(torch.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    $result = [ordered]@{
        schema = 1
        generated_utc = (Get-Date).ToUniversalTime().ToString('o')
        available = $false
        correctness_ok = $false
        full_support = $false
        reason = 'pytorch_not_found'
        python = $PythonExe
        import_output = ($torchCheck | Out-String).Trim()
        tests = @()
    }
    Write-Warning 'Functional validation was not run because this Python environment cannot import PyTorch.'
    Write-Warning 'Install a CUDA-facing PyTorch build in an isolated environment, then rerun scripts/test-functional.ps1 -PythonExe <venv\Scripts\python.exe>.'
    Write-ReportAndMaybeFail -Data $result -Failure 'Functional validation unavailable: PyTorch import failed.'
    return
}

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
        if (Test-Path $taskkill) {
            & $taskkill /PID $Process.Id /T /F 2>$null | Out-Null
        } else {
            $Process.Kill()
        }
    } catch {
        try { $Process.Kill() } catch {}
    }
}

function Invoke-FunctionalProbe {
    param([string]$TestName)

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $launcher
    $psi.Arguments = '-- ' + (Quote-ProcessArgument $PythonExe) + ' ' + (Quote-ProcessArgument $probePath) + ' --test ' + $TestName
    $psi.WorkingDirectory = $zluda
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    Set-ProcessEnvironment $psi 'HIP_PATH' $hip
    Set-ProcessEnvironment $psi 'ZLUDA_CC' $(if ($config.zluda_cc) { [string]$config.zluda_cc } else { '8.6' })
    Set-ProcessEnvironment $psi 'ROCBLAS_TENSILE_LIBPATH' (Join-Path $hip 'bin\rocblas\library')
    Set-ProcessEnvironment $psi 'HIPBLASLT_TENSILE_LIBPATH' (Join-Path $hip 'bin\hipblaslt\library')
    Set-ProcessEnvironment $psi 'PATH' "$hip\bin;$zluda;$env:PATH"
    if ($config.gpu -and $null -ne $config.gpu.index) {
        $gpuIndex = [string]$config.gpu.index
        Set-ProcessEnvironment $psi 'HIP_VISIBLE_DEVICES' $gpuIndex
        Set-ProcessEnvironment $psi 'ROCR_VISIBLE_DEVICES' $gpuIndex
    }

    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    [void]$p.Start()
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while (-not $p.HasExited -and $sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        Start-Sleep -Milliseconds 100
    }

    $timedOut = -not $p.HasExited
    if ($timedOut) {
        Stop-ProcessTree $p
        try { $p.WaitForExit() } catch {}
    }

    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    $parsed = $null
    if (-not $timedOut) {
        $lines = @($stdout -split "`r?`n" | Where-Object { $_.Trim() })
        for ($i = $lines.Count - 1; $i -ge 0; $i--) {
            try {
                $candidate = $lines[$i] | ConvertFrom-Json -ErrorAction Stop
                if ($candidate.schema -eq 1 -and $candidate.test -eq $TestName) {
                    $parsed = $candidate
                    break
                }
            } catch {}
        }
    }

    $status = if ($timedOut) { 'timeout' } elseif ($parsed) { [string]$parsed.status } else { 'error' }
    return [pscustomobject][ordered]@{
        test = $TestName
        status = $status
        timed_out = [bool]$timedOut
        timeout_seconds = $TimeoutSeconds
        process_exit = if ($timedOut) { $null } else { $p.ExitCode }
        result = $parsed
        stdout = $stdout.TrimEnd()
        stderr = $stderr.TrimEnd()
    }
}

Write-Host '=== ZLUDA numerical functional validation ==='
Write-Host "Python : $PythonExe"
Write-Host "PyTorch: $(($torchCheck | Out-String).Trim())"
Write-Host "Timeout: ${TimeoutSeconds}s per operation"
Write-Host ''

$testNames = @('matmul', 'conv2d', 'sdpa_math', 'sdpa_mem_efficient')
$tests = @()
foreach ($name in $testNames) {
    Write-Host ("Testing {0}..." -f $name) -NoNewline
    $entry = Invoke-FunctionalProbe $name
    $tests += $entry
    switch ($entry.status) {
        'pass'        { Write-Host ' PASS' }
        'unsupported' { Write-Host ' UNSUPPORTED (safe refusal)' -ForegroundColor Yellow }
        'timeout'     { Write-Host ' TIMEOUT' -ForegroundColor Red }
        'incorrect'   { Write-Host ' INCORRECT RESULT' -ForegroundColor Red }
        default       { Write-Host (" {0}" -f $entry.status.ToUpperInvariant()) -ForegroundColor Red }
    }
    if ($entry.result -and $entry.result.numerics) {
        Write-Host ("  max_abs={0} rel_scale={1} finite={2}" -f $entry.result.numerics.max_abs, $entry.result.numerics.max_abs_over_ref_scale, $entry.result.numerics.finite)
    }
}

$hardFailureStatuses = @('incorrect', 'error', 'timeout', 'unavailable')
$hardFailures = @($tests | Where-Object { $hardFailureStatuses -contains $_.status })
$unsupported = @($tests | Where-Object { $_.status -eq 'unsupported' })
$passed = @($tests | Where-Object { $_.status -eq 'pass' })

# Dense PPO validation only requires the GEMM path. Extended probes remain
# visible and can fail independently without misrepresenting PPO support.
$coreTestNames = @('matmul')
$coreTests = @($tests | Where-Object { $coreTestNames -contains $_.test })
$coreCorrectnessOk = ($coreTests.Count -eq $coreTestNames.Count) -and -not ($coreTests | Where-Object { $_.status -ne 'pass' })
$correctnessOk = ($hardFailures.Count -eq 0)
$fullSupport = ($passed.Count -eq $tests.Count)

$result = [ordered]@{
    schema = 1
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    available = $true
    python = $PythonExe
    pytorch = ($torchCheck | Out-String).Trim()
    gpu = if ($config.gpu) { $config.gpu } else { $null }
    core_profile = 'dense-ppo-gemm'
    core_tests = $coreTestNames
    core_correctness_ok = [bool]$coreCorrectnessOk
    correctness_ok = [bool]$correctnessOk
    full_support = [bool]$fullSupport
    safe_refusals = $unsupported.Count
    hard_failures = $hardFailures.Count
    tests = $tests
}

Write-Host ''
if ($coreCorrectnessOk) {
    Write-Host '[CORE PASS] Dense/GEMM numerical path passed.'
} else {
    Write-Warning '[CORE FAIL] Dense/GEMM numerical path failed; do not trust PPO/GEMM workloads.'
}

if ($correctnessOk) {
    if ($fullSupport) {
        Write-Host '[PASS] All tested operations completed with numerically valid results.'
    } else {
        Write-Warning '[PARTIAL] Tested operations were numerically safe, but one or more backends refused unsupported work.'
    }
} else {
    Write-Warning '[CAPABILITY FAIL] One or more extended operations returned an incorrect result, error, or hang. Review the per-operation report; do not use failing capabilities even when the dense/GEMM core passes.'
}
Write-ReportAndMaybeFail -Data $result -Failure $(if ($correctnessOk) { $null } else { 'ZLUDA functional correctness validation failed.' })

[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [int]$TimeoutSeconds = 30,
    [string]$ReportPath
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
if (-not $ReportPath) { $ReportPath = Join-Path $RuntimeRoot 'runtime-test.json' }
$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
if (-not (Test-Path $configPath)) { throw "Missing runtime config: $configPath. Run install.ps1 or setup.ps1 first." }
$config = Get-Content $configPath -Raw | ConvertFrom-Json
$zluda = [string]$config.zluda_root
$hip = [string]$config.hip_root
$launcher = Join-Path $zluda 'zluda.exe'
$probe = Join-Path $zluda 'cuda_check.exe'
if (-not (Test-Path $launcher)) { throw "Missing zluda.exe: $launcher" }
if (-not (Test-Path $probe)) { throw "Missing cuda_check.exe: $probe" }
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
        if (Test-Path $taskkill) {
            & $taskkill /PID $Process.Id /T /F 2>$null | Out-Null
        } else {
            $Process.Kill()
        }
    } catch {
        try { $Process.Kill() } catch {}
    }
}

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $launcher
$psi.Arguments = '-- ' + (Quote-ProcessArgument $probe)
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

Write-Host '=== ZLUDA cuda_check (runtime smoke only) ==='
if ($stdout) { Write-Host $stdout.TrimEnd() }
if ($stderr) { Write-Warning $stderr.TrimEnd() }

$requiredGroups = [ordered]@{
    nvcuda = @('nvcuda')
    cublas = @('cublas11','cublas12','cublas13')
    cublaslt = @('cublaslt11','cublaslt12','cublaslt13')
    cusparse = @('cusparse10','cusparse11','cusparse12')
    cufft = @('cufft10','cufft11','cufft12')
}
$checks = [ordered]@{}
foreach ($group in $requiredGroups.GetEnumerator()) {
    $ok = $false
    foreach ($name in $group.Value) {
        if ($stdout -match "(?m)^$([regex]::Escape($name))\s*:\s*OK") { $ok = $true; break }
    }
    $checks[$group.Key] = $ok
}
$coreGroupsOk = -not ($checks.Values -contains $false)
$runtimeSmokeOk = $coreGroupsOk -and -not $timedOut
$cudnnOk = [bool]($stdout -match '(?m)^cudnn[89]\s*:\s*OK')

Write-Host ''
foreach ($entry in $checks.GetEnumerator()) {
    Write-Host ("[{0}] {1}" -f ($(if($entry.Value){'PASS'}else{'FAIL'})), $entry.Key)
}
if ($cudnnOk) { Write-Host '[PASS] cudnn' }
else { Write-Warning '[OPTIONAL] cuDNN unavailable. Convolution-heavy workloads may require a different/newer stack.' }
if ($timedOut) { Write-Warning "cuda_check timed out after ${TimeoutSeconds}s. A timeout is a runtime-smoke failure even if earlier library probes printed OK." }

Write-Host ''
Write-Warning 'cuda_check validates library/runtime loading only. It does NOT prove numerical correctness.'
Write-Host 'Run .\scripts\test-functional.ps1 with a CUDA-facing PyTorch environment to test matmul, conv2d and SDPA numerics.'

$result = [ordered]@{
    schema = 2
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    core_ok = [bool]$runtimeSmokeOk
    core_groups_ok = [bool]$coreGroupsOk
    runtime_smoke_ok = [bool]$runtimeSmokeOk
    numerical_correctness_tested = $false
    timed_out = [bool]$timedOut
    process_exit = if ($timedOut) { $null } else { $p.ExitCode }
    cudnn_ok = [bool]$cudnnOk
    checks = [pscustomobject]$checks
    stdout = $stdout.TrimEnd()
    stderr = $stderr.TrimEnd()
}
New-Item -ItemType Directory -Force -Path (Split-Path $ReportPath -Parent) | Out-Null
$result | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $ReportPath
Write-Host "Report: $ReportPath"
if (-not $runtimeSmokeOk) { throw 'Core ZLUDA/HIP runtime smoke validation failed.' }

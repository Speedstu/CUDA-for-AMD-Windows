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

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $launcher
$psi.Arguments = '-- "' + $probe.Replace('"', '\"') + '"'
$psi.WorkingDirectory = $zluda
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true
$environment = $psi.EnvironmentVariables
if ($null -eq $environment) { throw 'Windows PowerShell could not initialize ProcessStartInfo.EnvironmentVariables.' }
$environment['HIP_PATH'] = $hip
$environment['ZLUDA_CC'] = if ($config.zluda_cc) { [string]$config.zluda_cc } else { '8.6' }
$environment['ROCBLAS_TENSILE_LIBPATH'] = Join-Path $hip 'bin\rocblas\library'
$environment['HIPBLASLT_TENSILE_LIBPATH'] = Join-Path $hip 'bin\hipblaslt\library'
$environment['PATH'] = "$($hip)\bin;$zluda;" + $env:PATH
if ($config.gpu -and $null -ne $config.gpu.hip_visible_device -and [string]$config.gpu.hip_visible_device -ne '') {
    $environment['HIP_VISIBLE_DEVICES'] = [string]$config.gpu.hip_visible_device
    $environment['ROCR_VISIBLE_DEVICES'] = [string]$config.gpu.hip_visible_device
}

$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
if (-not $p.Start()) { throw "Could not start ZLUDA runtime probe: $launcher" }
$sw = [Diagnostics.Stopwatch]::StartNew()
while (-not $p.HasExited -and $sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) { Start-Sleep -Milliseconds 250 }
$timedOut = -not $p.HasExited
if ($timedOut) { try { $p.Kill() } catch {}; $p.WaitForExit() }
$stdout = $p.StandardOutput.ReadToEnd()
$stderr = $p.StandardError.ReadToEnd()

Write-Host '=== ZLUDA cuda_check ==='
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
$coreOk = -not ($checks.Values -contains $false)
$cudnnOk = [bool]($stdout -match '(?m)^cudnn[89]\s*:\s*OK')

Write-Host ''
foreach ($entry in $checks.GetEnumerator()) {
    Write-Host ("[{0}] {1}" -f ($(if($entry.Value){'PASS'}else{'FAIL'})), $entry.Key)
}
if ($cudnnOk) { Write-Host '[PASS] cudnn' }
else { Write-Warning '[OPTIONAL] cuDNN unavailable. The stable Windows HIP SDK does not ship MIOpen; convolution-heavy workloads can need a newer/nightly stack.' }
if ($timedOut) { Write-Warning "cuda_check timed out after ${TimeoutSeconds}s." }

$result = [ordered]@{
    schema = 1
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    core_ok = [bool]$coreOk
    timed_out = [bool]$timedOut
    process_exit = $p.ExitCode
    cudnn_ok = [bool]$cudnnOk
    checks = [pscustomobject]$checks
    stdout = $stdout.TrimEnd()
    stderr = $stderr.TrimEnd()
}
New-Item -ItemType Directory -Force -Path (Split-Path $ReportPath -Parent) | Out-Null
$result | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $ReportPath
Write-Host "Report: $ReportPath"
if (-not $coreOk) { throw 'Core ZLUDA/HIP runtime validation failed.' }

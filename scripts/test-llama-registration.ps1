[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$LlamaRoot,
    [string]$RuntimeRoot,
    [string]$LlamaExe,
    [string]$PythonExe,
    [int]$TimeoutSeconds = 45,
    [string]$ReportPath
)

$ErrorActionPreference = 'Stop'
if ($TimeoutSeconds -lt 1) { throw 'TimeoutSeconds must be at least 1.' }

$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$LlamaRoot = [System.IO.Path]::GetFullPath($LlamaRoot)
if (-not $ReportPath) { $ReportPath = Join-Path $RuntimeRoot 'llama-registration-test.json' }

$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
if (-not (Test-Path $configPath)) { throw "Missing runtime config: $configPath" }
if (-not (Test-Path $LlamaRoot)) { throw "Missing llama.cpp directory: $LlamaRoot" }

if (-not $LlamaExe) {
    $candidate = Join-Path $LlamaRoot 'llama-cli.exe'
    if (Test-Path $candidate) { $LlamaExe = $candidate }
}
if (-not $LlamaExe -or -not (Test-Path $LlamaExe)) {
    throw 'llama-cli.exe not found. Pass -LlamaExe <path>.'
}
$LlamaExe = (Resolve-Path $LlamaExe).Path

$config = Get-Content $configPath -Raw | ConvertFrom-Json
$zluda = [string]$config.zluda_root
$hip = [string]$config.hip_root
$launcher = Join-Path $zluda 'zluda.exe'
$nvcuda = Join-Path $zluda 'nvcuda.dll'
if (-not (Test-Path $launcher)) { throw "Missing ZLUDA launcher: $launcher" }
if (-not $hip -or -not (Test-Path (Join-Path $hip 'bin'))) { throw "Missing HIP bin directory: $hip" }

function Get-Sha256OrNull([string]$Path) {
    if (-not $Path -or -not (Test-Path $Path -PathType Leaf)) { return $null }
    try { return (Get-FileHash $Path -Algorithm SHA256 -ErrorAction Stop).Hash }
    catch { return $null }
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

$projectRevision = $null
try {
    $git = Get-Command git.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $git) { $git = Get-Command git -ErrorAction SilentlyContinue | Select-Object -First 1 }
    if ($git) {
        $rev = ((& $git.Source -C $repo rev-parse HEAD 2>$null) | Out-String).Trim()
        if ($rev -match '^[0-9a-fA-F]{40}$') { $projectRevision = $rev.ToLowerInvariant() }
    }
} catch {}

$ggmlCuda = Join-Path (Split-Path $LlamaExe -Parent) 'ggml-cuda.dll'
$hashes = [ordered]@{
    llama_cli_sha256 = Get-Sha256OrNull $LlamaExe
    ggml_cuda_sha256 = Get-Sha256OrNull $ggmlCuda
    zluda_launcher_sha256 = Get-Sha256OrNull $launcher
    nvcuda_sha256 = Get-Sha256OrNull $nvcuda
    runtime_config_sha256 = Get-Sha256OrNull $configPath
}

$driverPreflight = $null
$driverReportPath = Join-Path $RuntimeRoot 'llama-driver-preflight.json'
if ($PythonExe) {
    try {
        & (Join-Path $PSScriptRoot 'test-capabilities.ps1') -RuntimeRoot $RuntimeRoot -PythonExe $PythonExe -Tests @('driver_pci_bus_id','driver_launch_ex','driver_func_attributes','driver_function_metadata','driver_ptx_selection','driver_buffer_clear') -TimeoutSeconds $TimeoutSeconds -RetryTimeoutSeconds 0 -ReportPath $driverReportPath
        if (Test-Path $driverReportPath) {
            $driverPreflight = Get-Content $driverReportPath -Raw | ConvertFrom-Json
        }
    } catch {
        $driverPreflight = [pscustomobject]@{
            error = $_.Exception.Message
            report_path = $driverReportPath
        }
    }
}

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $launcher
$psi.Arguments = '-- ' + (Quote-Arg $LlamaExe) + ' --list-devices'
$psi.WorkingDirectory = Split-Path $LlamaExe -Parent
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
Set-Env $psi 'PATH' "$hip\bin;$zluda;$($psi.WorkingDirectory);$env:PATH"
if ($config.gpu -and $null -ne $config.gpu.index) {
    Set-Env $psi 'HIP_VISIBLE_DEVICES' ([string]$config.gpu.index)
    Set-Env $psi 'ROCR_VISIBLE_DEVICES' ([string]$config.gpu.index)
}

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
    Stop-Tree $p
    try { $p.WaitForExit() } catch {}
}
try { $stdout = $stdoutTask.GetAwaiter().GetResult() } catch { $stdout = '' }
try { $stderr = $stderrTask.GetAwaiter().GetResult() } catch { $stderr = '' }
$exitCode = if ($timedOut) { $null } else { $p.ExitCode }

$combined = ($stdout + "`n" + $stderr)
$zludaDevice = [bool]($combined -match '(?im)^\s*(CUDA\d+|Device\s+\d+).*AMD.*\[ZLUDA\]')
$explicitNone = [bool]($combined -match '(?im)Available devices:\s*(?:\r?\n)?\s*\(none\)')
$cpuFallbackOnly = [bool](-not $zludaDevice -and $combined -match '(?i)CPU')
$registrationOk = [bool](-not $timedOut -and $exitCode -eq 0 -and $zludaDevice -and -not $explicitNone)
$driverOk = $null
if ($driverPreflight) {
    if ($driverPreflight.PSObject.Properties.Name -contains 'correctness_ok') {
        $driverOk = [bool]$driverPreflight.correctness_ok
    } elseif ($driverPreflight.PSObject.Properties.Name -contains 'all_tested_capabilities_pass') {
        $driverOk = [bool]$driverPreflight.all_tested_capabilities_pass
    }
}

$ptxMetadata = $null
$ptxVersion = $null
$ptxVersionExpected = $null
$ptxMetadataOk = $null
$pdlGateRisk = $null
$driverResults = @()
if ($driverPreflight) {
    if ($driverPreflight.PSObject.Properties.Name -contains 'results') {
        $driverResults = @($driverPreflight.results)
    } elseif ($driverPreflight.PSObject.Properties.Name -contains 'tests') {
        $driverResults = @($driverPreflight.tests)
    }
}
if ($driverResults.Count -gt 0) {
    $metadataEntry = @($driverResults | Where-Object { $_.test -eq 'driver_function_metadata' } | Select-Object -First 1)
    if ($metadataEntry.Count -gt 0 -and $metadataEntry[0].result) {
        $ptxMetadata = $metadataEntry[0].result
        if ($ptxMetadata.ptx_version) {
            $ptxVersion = [int]$ptxMetadata.ptx_version.value
            $ptxVersionExpected = [int]$ptxMetadata.ptx_version.expected
            $ptxMetadataOk = [bool](
                $ptxMetadata.ptx_version.rc -eq 0 -and
                $ptxVersion -eq $ptxVersionExpected
            )
        }
        if ($ptxMetadata.PSObject.Properties.Name -contains 'llama_b10978_false_pdl_gate') {
            $pdlGateRisk = [bool]$ptxMetadata.llama_b10978_false_pdl_gate
            if ($ptxMetadata.cases -and $ptxMetadata.cases.llama_b10978_ptx84_sm90) {
                $llamaCase = $ptxMetadata.cases.llama_b10978_ptx84_sm90
                $ptxVersion = [int]$llamaCase.ptx_version.value
                $ptxVersionExpected = [int]$llamaCase.ptx_version.expected
                $ptxMetadataOk = [bool]$llamaCase.ok
            }
        } elseif ($null -ne $ptxVersion -and $null -ne $ptxVersionExpected) {
            # Backward compatibility with older single-case capability reports.
            $pdlGateRisk = [bool]($ptxVersion -ge 90 -and $ptxVersionExpected -lt 90)
        }
    }
}

$ok = [bool]($registrationOk -and ($null -eq $driverOk -or $driverOk))

$report = [ordered]@{
    schema = 1
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    project_revision = $projectRevision
    test = 'llama_registration'
    mode = 'registration_only'
    model_kernels_executed = $false
    status = if ($ok) { 'pass' } elseif ($timedOut) { 'timeout' } elseif ($explicitNone -or $cpuFallbackOnly) { 'fallback_or_no_cuda_device' } elseif ($false -eq $driverOk) { 'driver_preflight_failed' } else { 'error' }
    runtime_root = $RuntimeRoot
    llama_root = $LlamaRoot
    llama_exe = $LlamaExe
    timeout_seconds = $TimeoutSeconds
    timed_out = [bool]$timedOut
    process_exit = $exitCode
    elapsed_ms = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
    zluda_device_detected = $zludaDevice
    explicit_no_devices = $explicitNone
    cpu_fallback_only = $cpuFallbackOnly
    registration_ok = $registrationOk
    driver_preflight_ok = $driverOk
    driver_preflight = $driverPreflight
    ptx_metadata = [ordered]@{
        observed_ptx_version = $ptxVersion
        expected_ptx_version = $ptxVersionExpected
        metadata_ok = $ptxMetadataOk
        llama_b10978_pdl_gate_risk = $pdlGateRisk
    }
    environment = [ordered]@{
        zluda_cc = if ($config.zluda_cc) { [string]$config.zluda_cc } else { '8.6' }
        ggml_cuda_pdl = 'not_overridden'
        gpu = if ($config.gpu) { $config.gpu } else { $null }
    }
    hashes = $hashes
    stdout = $stdout.TrimEnd()
    stderr = $stderr.TrimEnd()
}

New-Item -ItemType Directory -Force -Path (Split-Path $ReportPath -Parent) | Out-Null
$report | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $ReportPath

Write-Host '=== llama.cpp registration smoke ==='
Write-Host "llama-cli : $LlamaExe"
Write-Host "ZLUDA     : $zluda"
Write-Host "HIP       : $hip"
Write-Host "PDL       : not overridden (no model kernels are launched)"
Write-Host "Status    : $($report.status)"
Write-Host "Report    : $ReportPath"
Write-Host 'Model     : not loaded; no inference kernels executed'
if ($null -ne $driverOk) {
    Write-Host "Driver    : $(if ($driverOk) { 'focused preflight passed' } else { 'focused preflight failed' })"
}
if ($null -ne $ptxMetadataOk) {
    Write-Host "PTX meta  : observed=$ptxVersion expected=$ptxVersionExpected $(if ($ptxMetadataOk) { 'OK' } else { 'MISMATCH' })"
}
if ($true -eq $pdlGateRisk) {
    Write-Warning 'PTX metadata can falsely satisfy llama.cpp b10978 ptxVersion>=90 and select PDL. Use the PTX-version candidate only after its rebuilt runtime is validated.'
}

if (-not $ok) {
    if ($timedOut) { Write-Warning 'llama.cpp device enumeration timed out.' }
    elseif ($explicitNone -or $cpuFallbackOnly) { Write-Warning 'No AMD/ZLUDA CUDA device was confirmed; do not treat CPU fallback as a successful CUDA run.' }
    else { Write-Warning 'llama.cpp device enumeration did not complete cleanly through ZLUDA.' }
    exit 1
}
exit 0

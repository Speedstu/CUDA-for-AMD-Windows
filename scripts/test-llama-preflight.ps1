[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$LlamaExe,
    [string]$RuntimeRoot,
    [string]$PythonExe,
    [int]$TimeoutSeconds = 30,
    [string]$ReportPath,
    [switch]$NoStage,
    [switch]$Strict
)

$ErrorActionPreference = 'Stop'
if ($TimeoutSeconds -lt 1) { throw 'TimeoutSeconds must be at least 1.' }

$repo = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $repo '.runtime' }
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$LlamaExe = (Resolve-Path $LlamaExe).Path
if (-not $ReportPath) { $ReportPath = Join-Path $RuntimeRoot 'llama-preflight.json' }

$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
if (-not (Test-Path $configPath)) { throw "Missing runtime config: $configPath" }
$config = Get-Content $configPath -Raw | ConvertFrom-Json
$zluda = [string]$config.zluda_root
$hip = [string]$config.hip_root
$launcher = Join-Path $zluda 'zluda.exe'
if (-not (Test-Path $launcher)) { throw "Missing ZLUDA launcher: $launcher" }
if (-not $hip -or -not (Test-Path (Join-Path $hip 'bin'))) { throw "Missing HIP bin directory: $hip" }

$targetDir = Split-Path $LlamaExe -Parent
if (-not $NoStage) {
    & (Join-Path $PSScriptRoot 'stage-runtime.ps1') -TargetDir $targetDir -RuntimeRoot $RuntimeRoot
}

function Get-Sha256OrNull([string]$Path) {
    if (-not $Path -or -not (Test-Path $Path -PathType Leaf)) { return $null }
    try { return (Get-FileHash $Path -Algorithm SHA256 -ErrorAction Stop).Hash }
    catch { return $null }
}

function Quote-ProcessArgument([string]$Value) {
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

function Invoke-ZludaProgram {
    param([string]$Program, [string[]]$Arguments)

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $launcher
    $all = @('--', $Program) + @($Arguments)
    $psi.Arguments = ($all | ForEach-Object { Quote-ProcessArgument ([string]$_) }) -join ' '
    $psi.WorkingDirectory = (Split-Path $Program -Parent)
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $zludaCc = if ($config.zluda_cc) { [string]$config.zluda_cc } else { '8.6' }
    Set-ProcessEnvironment $psi 'ZLUDA_CC' $zludaCc
    Set-ProcessEnvironment $psi 'HIP_PATH' $hip
    Set-ProcessEnvironment $psi 'ROCBLAS_TENSILE_LIBPATH' (Join-Path $hip 'bin\rocblas\library')

    $hipblasltLib = Join-Path $hip 'bin\hipblaslt\library'
    if ($config.gpu -and $config.gpu.arch) {
        $archLib = Join-Path $hipblasltLib ([string]$config.gpu.arch)
        if (Test-Path $archLib) { $hipblasltLib = $archLib }
    }
    if (Test-Path $hipblasltLib) {
        Set-ProcessEnvironment $psi 'HIPBLASLT_TENSILE_LIBPATH' $hipblasltLib
    }

    $parts = @($targetDir, $zluda, (Join-Path $hip 'bin'))
    Set-ProcessEnvironment $psi 'PATH' ((($parts | Where-Object { $_ -and (Test-Path $_) }) -join ';') + ';' + $env:PATH)

    if ($config.gpu -and $null -ne $config.gpu.index) {
        $gpuIndex = [string]$config.gpu.index
        Set-ProcessEnvironment $psi 'HIP_VISIBLE_DEVICES' $gpuIndex
        Set-ProcessEnvironment $psi 'ROCR_VISIBLE_DEVICES' $gpuIndex
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
        Stop-ProcessTree $p
        try { $p.WaitForExit() } catch {}
    }

    try { $stdout = $stdoutTask.GetAwaiter().GetResult() } catch { $stdout = '' }
    try { $stderr = $stderrTask.GetAwaiter().GetResult() } catch { $stderr = '' }
    return [pscustomobject][ordered]@{
        timed_out = [bool]$timedOut
        exit_code = if ($timedOut) { $null } else { $p.ExitCode }
        elapsed_ms = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
        stdout = $stdout.TrimEnd()
        stderr = $stderr.TrimEnd()
    }
}

$driverReportPath = Join-Path $RuntimeRoot 'llama-driver-preflight.json'
$driverPreflight = $null
if ($PythonExe) {
    try {
        & (Join-Path $PSScriptRoot 'test-capabilities.ps1') -RuntimeRoot $RuntimeRoot -PythonExe $PythonExe -Tests @('driver_pci_bus_id','driver_launch_ex','driver_func_attributes') -TimeoutSeconds $TimeoutSeconds -RetryTimeoutSeconds 0 -ReportPath $driverReportPath
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

Write-Host '=== llama.cpp ZLUDA preflight ==='
Write-Host "llama  : $LlamaExe"
Write-Host "runtime: $RuntimeRoot"
Write-Host 'mode   : device registration only; no model kernels are launched'
Write-Host ''

$list = Invoke-ZludaProgram -Program $LlamaExe -Arguments @('--list-devices')
$combined = ($list.stdout + [Environment]::NewLine + $list.stderr)
$hasZluda = [bool]($combined -match '\[ZLUDA\]')
$hasCudaDevice = [bool]($combined -match '(?im)^\s*(CUDA\d+|Device\s+\d+:).*')
$cpuFallbackOnly = [bool](-not $hasZluda -and $combined -match '(?im)\bCPU\b')
$listOk = [bool](-not $list.timed_out -and $list.exit_code -eq 0 -and $hasZluda -and $hasCudaDevice)

$driverOk = $null
if ($driverPreflight -and $driverPreflight.PSObject.Properties.Name -contains 'correctness_ok') {
    $driverOk = [bool]$driverPreflight.correctness_ok
}
$overallOk = [bool]($listOk -and ($null -eq $driverOk -or $driverOk))

$report = [ordered]@{
    schema = 1
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    mode = 'registration-preflight-only'
    safe_by_default = $true
    model_kernels_executed = $false
    llama_exe = $LlamaExe
    llama_sha256 = Get-Sha256OrNull $LlamaExe
    runtime_root = $RuntimeRoot
    zluda_cc = if ($config.zluda_cc) { [string]$config.zluda_cc } else { '8.6' }
    gpu = if ($config.gpu) { $config.gpu } else { $null }
    runtime_hashes = [ordered]@{
        launcher_sha256 = Get-Sha256OrNull $launcher
        nvcuda_sha256 = Get-Sha256OrNull (Join-Path $zluda 'nvcuda.dll')
        runtime_config_sha256 = Get-Sha256OrNull $configPath
    }
    driver_preflight = $driverPreflight
    list_devices = $list
    detected_zluda_device = $hasZluda
    detected_cuda_device = $hasCudaDevice
    cpu_fallback_only = $cpuFallbackOnly
    ok = $overallOk
    notes = @(
        'This preflight does not claim inference compatibility.',
        'The stock CUDA runtime shipped with llama.cpp should remain present; stage-runtime does not rename nvcudart_hybrid64.dll over cudart64_*.dll.',
        'GGML_CUDA_PDL=0 is intentionally not set here because bypassing the safe 801 refusal can expose a deeper incompatible kernel path on unvalidated GPUs.'
    )
}

New-Item -ItemType Directory -Force -Path (Split-Path $ReportPath -Parent) | Out-Null
$report | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $ReportPath

if ($listOk) {
    Write-Host '[PASS] llama.cpp sees an AMD/ZLUDA CUDA device.'
} else {
    Write-Warning '[FAIL] llama.cpp did not cleanly enumerate an AMD/ZLUDA CUDA device.'
}
if ($null -ne $driverOk) {
    if ($driverOk) { Write-Host '[PASS] Focused CUDA driver preflight passed.' }
    else { Write-Warning '[FAIL] Focused CUDA driver preflight did not pass.' }
}
Write-Host "Report: $ReportPath"
Write-Host 'No model inference was executed.'

if ($Strict -and -not $overallOk) {
    throw 'llama.cpp ZLUDA preflight failed.'
}

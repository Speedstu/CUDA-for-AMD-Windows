param(
    [string]$RuntimeRoot = ".runtime",
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [switch]$SkipStaticPrecompile,
    [switch]$SkipTrainingWarmup,
    [int]$TimeoutSeconds = 600,
    [string]$ReportPath
)

$ErrorActionPreference = 'Stop'
$RuntimeRoot = (Resolve-Path $RuntimeRoot).Path
$configPath = Join-Path $RuntimeRoot 'runtime-config.json'
if (-not (Test-Path $configPath)) { throw "Missing runtime config: $configPath" }
if (-not (Test-Path $PythonExe)) { throw "Python not found: $PythonExe" }
$config = Get-Content $configPath -Raw | ConvertFrom-Json
$zluda = [string]$config.zluda_root
$hip = [string]$config.hip_root
if (-not (Test-Path $zluda)) { throw "ZLUDA root not found: $zluda" }
if (-not (Test-Path $hip)) { throw "HIP root not found: $hip" }
if (-not $ReportPath) { $ReportPath = Join-Path $RuntimeRoot 'pytorch-warmup.json' }

$repoRoot = Split-Path $PSScriptRoot -Parent
$probe = Join-Path $PSScriptRoot 'training_warmup.py'
$launcher = Join-Path $zluda 'zluda.exe'
$precompiler = Join-Path $zluda 'zluda_precompile.exe'
$gpuIndex = if ($config.gpu -and $null -ne $config.gpu.index) { [string]$config.gpu.index } else { '0' }
$arch = if ($config.gpu -and $config.gpu.arch) { [string]$config.gpu.arch } else { $null }
$hipblasltLib = Join-Path $hip 'bin\hipblaslt\library'
if ($arch) {
    $candidate = Join-Path $hipblasltLib $arch
    if (Test-Path $candidate) { $hipblasltLib = $candidate }
}

function Quote-Arg([string]$value) { return '"' + ($value -replace '"','\"') + '"' }
function Add-Env([Diagnostics.ProcessStartInfo]$psi, [string]$name, [string]$value) {
    if ($null -ne $value -and $value -ne '') { $psi.Environment[$name] = $value }
}
function Invoke-Captured([string]$file, [string]$arguments, [string]$workingDirectory) {
    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $file
    $psi.Arguments = $arguments
    $psi.WorkingDirectory = $workingDirectory
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    Add-Env $psi 'HIP_PATH' $hip
    Add-Env $psi 'ZLUDA_CC' $(if ($config.zluda_cc) { [string]$config.zluda_cc } else { '8.6' })
    Add-Env $psi 'ROCBLAS_TENSILE_LIBPATH' (Join-Path $hip 'bin\rocblas\library')
    if (Test-Path $hipblasltLib) { Add-Env $psi 'HIPBLASLT_TENSILE_LIBPATH' $hipblasltLib }
    Add-Env $psi 'PATH' "$hip\bin;$zluda;$env:PATH"
    Add-Env $psi 'HIP_VISIBLE_DEVICES' $gpuIndex
    Add-Env $psi 'ROCR_VISIBLE_DEVICES' $gpuIndex
    $p = [Diagnostics.Process]::new(); $p.StartInfo = $psi; [void]$p.Start()
    $stdoutTask = $p.StandardOutput.ReadToEndAsync(); $stderrTask = $p.StandardError.ReadToEndAsync()
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while (-not $p.HasExited -and $sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) { Start-Sleep -Milliseconds 200 }
    $timedOut = -not $p.HasExited
    if ($timedOut) {
        try { & "$env:SystemRoot\System32\taskkill.exe" /PID $p.Id /T /F 2>$null | Out-Null } catch { try { $p.Kill() } catch {} }
        try { $p.WaitForExit() } catch {}
    }
    try { $stdout = $stdoutTask.GetAwaiter().GetResult() } catch { $stdout = '' }
    try { $stderr = $stderrTask.GetAwaiter().GetResult() } catch { $stderr = '' }
    [pscustomobject]@{ timed_out=$timedOut; exit=if($timedOut){$null}else{$p.ExitCode}; seconds=$sw.Elapsed.TotalSeconds; stdout=$stdout; stderr=$stderr }
}

$torchCuda = (& $PythonExe -c "import pathlib,torch; print(pathlib.Path(torch.__file__).resolve().parent/'lib'/'torch_cuda.dll')").Trim()
$static = $null
if (-not $SkipStaticPrecompile -and (Test-Path $precompiler) -and (Test-Path $torchCuda)) {
    Write-Host "Static precompile: $torchCuda"
    $static = Invoke-Captured $precompiler ("-d={0} {1}" -f $gpuIndex,(Quote-Arg $torchCuda)) $zluda
    if ($static.stdout.Trim()) { Write-Host $static.stdout.TrimEnd() }
    if ($static.stderr.Trim()) { Write-Warning $static.stderr.TrimEnd() }
    if ($static.timed_out -or $static.exit -ne 0) { throw 'ZLUDA static precompile failed or timed out.' }
}

$dynamic = $null
$parsed = $null
if (-not $SkipTrainingWarmup) {
    Write-Host 'Dynamic training-kernel warmup...'
    $dynamic = Invoke-Captured $launcher ("-- {0} {1}" -f (Quote-Arg $PythonExe),(Quote-Arg $probe)) $zluda
    if ($dynamic.stdout.Trim()) { Write-Host $dynamic.stdout.TrimEnd() }
    if ($dynamic.stderr.Trim()) {
        $visible = @($dynamic.stderr -split "`r?`n" | Where-Object { $_ -and $_ -notlike 'CUDAAMD_WARMUP:*' })
        if ($visible.Count) { Write-Warning ($visible -join "`n") }
        $markers = @($dynamic.stderr -split "`r?`n" | Where-Object { $_ -like 'CUDAAMD_WARMUP:*' })
        if ($markers.Count) { try { $parsed = $markers[-1].Substring('CUDAAMD_WARMUP:'.Length) | ConvertFrom-Json } catch {} }
    }
    if ($dynamic.timed_out -or $dynamic.exit -ne 0 -or -not $parsed -or -not $parsed.ok) { throw 'PyTorch training warmup failed or timed out.' }
}

$cachePath = Join-Path $env:LOCALAPPDATA 'zluda\ComputeCache\zluda2.db'
$result = [ordered]@{
    schema = 1
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    runtime_root = $RuntimeRoot
    python = $PythonExe
    torch_cuda = $torchCuda
    gpu = $config.gpu
    static_precompile = if($static){[ordered]@{exit=$static.exit;seconds=$static.seconds}}else{$null}
    dynamic_warmup = $parsed
    cache = [ordered]@{path=$cachePath;exists=(Test-Path $cachePath);bytes=if(Test-Path $cachePath){(Get-Item $cachePath).Length}else{$null}}
}
New-Item -ItemType Directory -Force (Split-Path $ReportPath -Parent) | Out-Null
$result | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $ReportPath
Write-Host "Report: $ReportPath"

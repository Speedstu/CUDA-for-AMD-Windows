[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$LlamaDir,
    [Parameter(Mandatory=$true)][string]$RuntimeRoot,
    [Parameter(Mandatory=$true)][string]$Model,
    [int]$GpuLayers = 1,
    [int]$Predict = 1,
    [string]$Prompt = 'Hello',
    [string]$TraceFile,
    [switch]$AllowMoreThanFourGpuLayers
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$LlamaDir = (Resolve-Path $LlamaDir).Path
$RuntimeRoot = (Resolve-Path $RuntimeRoot).Path
$Model = (Resolve-Path $Model).Path

if ($GpuLayers -lt 1) { throw 'GpuLayers must be at least 1 for a CUDA smoke.' }
if ($GpuLayers -gt 4 -and -not $AllowMoreThanFourGpuLayers) {
    throw 'Refusing >4 GPU layers in the gfx1150 safe launcher. Re-run with -AllowMoreThanFourGpuLayers only after the 1-layer smoke passes.'
}

$cli = Join-Path $LlamaDir 'llama-cli.exe'
if (-not (Test-Path $cli)) { throw "Missing llama-cli.exe: $cli" }

$manifest = Join-Path $LlamaDir 'build-manifest.json'
if (-not (Test-Path $manifest)) {
    throw 'Missing build-manifest.json. This launcher only accepts the repository safe llama artifact; do not point it at a generic b10978 CUDA build.'
}
$m = Get-Content $manifest -Raw | ConvertFrom-Json
if ([string]$m.llama_revision -ne '1e7bcf3da4b2741868d152fa47976fb2501c85e3') {
    throw "Unexpected llama.cpp build revision: $($m.llama_revision)"
}
if ([string]$m.cuda_toolkit -ne '12.4') {
    throw "Unexpected CUDA toolkit in safe artifact: $($m.cuda_toolkit)"
}
$forceCublas = $m.PSObject.Properties['force_cublas']
if (-not $forceCublas -or -not [bool]$forceCublas.Value) {
    throw 'Safe artifact must be built with force_cublas=true.'
}
foreach ($name in @('force_mmq','cuda_graphs','cuda_vmm','peer_copy','flash_attention_compiled','nccl','openssl')) {
    $prop = $m.PSObject.Properties[$name]
    if (-not $prop) { throw "Safe artifact manifest is missing required field: $name" }
    if ([bool]$prop.Value) { throw "Safe artifact requires $name=false." }
}
$arches = @($m.cuda_architectures | ForEach-Object { [string]$_ })
foreach ($requiredArch in @('75-virtual','80-virtual')) {
    if ($arches -notcontains $requiredArch) {
        throw "Safe artifact is missing required PTX target: $requiredArch"
    }
}

# Stage only project compatibility DLLs. The stock NVIDIA CUDA runtime itself
# is deliberately not redistributed by this project.
& (Join-Path $PSScriptRoot 'stage-runtime.ps1') -TargetDir $LlamaDir -RuntimeRoot $RuntimeRoot

$cudart = Join-Path $LlamaDir 'cudart64_12.dll'
if (-not (Test-Path $cudart)) {
    throw 'Missing stock CUDA 12.x cudart64_12.dll beside llama-cli.exe. Use the same CUDA 12.4 runtime package already validated for b10978.'
}

$config = Get-Content (Join-Path $RuntimeRoot 'runtime-config.json') -Raw | ConvertFrom-Json
$zluda = [string]$config.zluda_root
$hip = [string]$config.hip_root
$launcher = Join-Path $zluda 'zluda.exe'
if (-not (Test-Path $launcher)) { throw "Missing ZLUDA launcher: $launcher" }

if (-not $TraceFile) {
    $TraceFile = Join-Path $LlamaDir 'zluda-kernels.log'
}
$TraceFile = [IO.Path]::GetFullPath($TraceFile)
Remove-Item $TraceFile -Force -ErrorAction SilentlyContinue

$env:ZLUDA_CC = if ($config.zluda_cc) { [string]$config.zluda_cc } else { '8.6' }
if ($config.gpu -and $null -ne $config.gpu.index) {
    $env:HIP_VISIBLE_DEVICES = [string]$config.gpu.index
    Remove-Item Env:ROCR_VISIBLE_DEVICES -ErrorAction SilentlyContinue
}
$env:CUDA_LAUNCH_BLOCKING = '1'
$env:ZLUDA_LAUNCH_TRACE = '1'
$env:ZLUDA_LAUNCH_TRACE_FILE = $TraceFile
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
$env:HIP_PATH = $hip
if ($hip) {
    $env:ROCBLAS_TENSILE_LIBPATH = Join-Path $hip 'bin\rocblas\library'
    $env:PATH = (Join-Path $hip 'bin') + ';' + $zluda + ';' + $LlamaDir + ';' + $env:PATH
} else {
    $env:PATH = $zluda + ';' + $LlamaDir + ';' + $env:PATH
}

Write-Host "[safe-llama] ZLUDA_CC=$env:ZLUDA_CC"
if ($env:HIP_VISIBLE_DEVICES) { Write-Host "[safe-llama] HIP_VISIBLE_DEVICES=$env:HIP_VISIBLE_DEVICES" }
Write-Host "[safe-llama] CUDA_LAUNCH_BLOCKING=1"
Write-Host "[safe-llama] GGML_CUDA_DISABLE_GRAPHS=1"
Write-Host "[safe-llama] gpu_layers=$GpuLayers predict=$Predict"
Write-Host "[safe-llama] kernel trace: $TraceFile"

$args = @(
    '--',
    $cli,
    '-m', $Model,
    '-ngl', [string]$GpuLayers,
    '-dev', 'CUDA0',
    '-sm', 'none',
    '-nkvo',
    '--no-op-offload',
    '-fa', 'off',
    '-st',
    '-n', [string]$Predict,
    '-p', $Prompt,
    '--no-perf'
)

# ProcessStartInfo.ArgumentList avoids command-line re-tokenization of model
# paths and prompts. pwsh 7+ is recommended for this diagnostic launcher.
$psi = [Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $launcher
$psi.WorkingDirectory = $LlamaDir
$psi.UseShellExecute = $false
foreach ($arg in $args) { [void]$psi.ArgumentList.Add([string]$arg) }

$p = [Diagnostics.Process]::new()
$p.StartInfo = $psi
[void]$p.Start()
$p.WaitForExit()

$exitCode = $p.ExitCode
Write-Host "[safe-llama] exit=$exitCode"

$traceBegin = 0
$traceTerminal = 0
$traceErrors = 0
$openLaunches = @{}

if (Test-Path $TraceFile) {
    $traceLines = @(Get-Content $TraceFile)
    foreach ($line in $traceLines) {
        if ($line -match '^\[zluda-launch\]\s+#(\d+)\s+(begin|done|launch-error|sync-error)\s+kernel=(\S+)') {
            $id = [int64]$Matches[1]
            $event = [string]$Matches[2]
            $kernel = [string]$Matches[3]
            if ($event -eq 'begin') {
                $traceBegin++
                $openLaunches[$id] = $kernel
            } else {
                $traceTerminal++
                [void]$openLaunches.Remove($id)
                if ($event -eq 'launch-error' -or $event -eq 'sync-error') { $traceErrors++ }
            }
        }
    }

    $tail = @($traceLines | Select-Object -Last 12)
    if ($tail.Count) {
        Write-Host '[safe-llama] trace tail:'
        $tail | ForEach-Object { Write-Host $_ }
    }
}

Write-Host "[safe-llama] trace summary: begins=$traceBegin terminals=$traceTerminal errors=$traceErrors unmatched=$($openLaunches.Count)"

if ($exitCode -eq 0) {
    if ($traceBegin -lt 1) {
        throw 'Safe llama run exited 0 but no ZLUDA launch-blocking trace was recorded. Refusing to accept an old/non-instrumented nvcuda.dll.'
    }
    if ($traceErrors -ne 0) {
        throw "Safe llama run exited 0 but the launch trace contains $traceErrors launch/sync error(s)."
    }
    if ($openLaunches.Count -ne 0) {
        $last = $openLaunches.GetEnumerator() | Sort-Object Name | Select-Object -Last 1
        throw "Safe llama run exited 0 with $($openLaunches.Count) unmatched launch(es); last open #$($last.Name) kernel=$($last.Value)"
    }
    if ($traceBegin -ne $traceTerminal) {
        throw "Safe llama trace accounting mismatch: begins=$traceBegin terminals=$traceTerminal."
    }
}

exit $exitCode

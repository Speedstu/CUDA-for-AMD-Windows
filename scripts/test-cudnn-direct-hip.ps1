[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$TheRockRoot,
    [Parameter(Mandatory)][string]$BridgeDll,
    [string]$PythonExe = 'python.exe',
    [string]$OutputDir,
    [ValidateSet('all','activation','softmax','pooling','tensor','half')][string]$Only = 'all',
    [ValidateRange(10,600)][int]$TimeoutSeconds = 120,
    [switch]$AllowGpuTest
)

$ErrorActionPreference='Stop'
if (-not $AllowGpuTest) {
    throw 'Explicit -AllowGpuTest required: this script runs real AMD GPU kernels.'
}
$repo = Split-Path $PSScriptRoot -Parent
$TheRockRoot = [IO.Path]::GetFullPath($TheRockRoot)
$BridgeDll = [IO.Path]::GetFullPath($BridgeDll)
$hip = Join-Path $TheRockRoot 'bin\amdhip64_7.dll'
$miopen = Join-Path $TheRockRoot 'bin\MIOpen.dll'
$probe = Join-Path $PSScriptRoot 'test-cudnn-direct-hip-gpu.py'
foreach ($path in @($hip,$miopen,$BridgeDll,$probe)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "GPU test prerequisite missing: $path"
    }
}
if ((Split-Path $BridgeDll -Leaf) -ine 'cudnn64_8.dll') {
    throw 'Bridge DLL must be the intended cudnn64_8.dll, not a system CUDA DLL.'
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $repo '.runtime\cudnn-direct-hip-gpu-evidence'
}
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$reportPath = Join-Path $OutputDir 'gpu-evidence.json'
$stdout = Join-Path $OutputDir 'stdout.txt'
$stderr = Join-Path $OutputDir 'stderr.txt'
Remove-Item $reportPath,$stdout,$stderr -ErrorAction SilentlyContinue

# Explicit argument vector: avoid lossy quoting of Windows paths with spaces.
$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $PythonExe
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
foreach ($arg in @($probe,'--bridge-dll',$BridgeDll,'--hip-dll',$hip,
                   '--miopen-dll',$miopen,'--report',$reportPath,'--only',$Only)) {
    [void]$psi.ArgumentList.Add([string]$arg)
}
$psi.Environment['PATH'] = (Join-Path $TheRockRoot 'bin') + ';' + [Environment]::GetEnvironmentVariable('PATH')
$psi.Environment['CUDA_AMD_MIOPEN_DLL'] = $miopen
$proc = [System.Diagnostics.Process]::Start($psi)
if (-not $proc) { throw 'Failed to create isolated GPU test process.' }
$stdoutTask = $proc.StandardOutput.ReadToEndAsync()
$stderrTask = $proc.StandardError.ReadToEndAsync()
try {
    if (-not $proc.WaitForExit($TimeoutSeconds*1000)) {
        $proc.Kill($true)
        [void]$proc.WaitForExit(10000)
        throw "Real GPU probe exceeded $TimeoutSeconds seconds. No automatic retry."
    }
    $outText = $stdoutTask.GetAwaiter().GetResult()
    $errText = $stderrTask.GetAwaiter().GetResult()
    [IO.File]::WriteAllText($stdout,$outText)
    [IO.File]::WriteAllText($stderr,$errText)
    if ($proc.ExitCode -ne 0) {
        throw "GPU probe failed (exit $($proc.ExitCode)); see $stderr"
    }
} finally {
    $proc.Dispose()
}
if (-not (Test-Path -LiteralPath $reportPath)) {
    throw "GPU process exited without a numerical evidence report: $reportPath"
}
$report = Get-Content $reportPath -Raw | ConvertFrom-Json
if ($report.scope -ne 'direct_HIP_device_allocations_real_MIOpen_no_PyTorch') {
    throw 'Unexpected or unsupported GPU evidence scope.'
}
if ($report.hip_device_count -lt 1 -or -not $report.results) {
    throw 'Missing real HIP device or numerical outcomes.'
}
$expected = switch ($Only) {
    'activation' { 3 }
    'softmax' { 1 }
    'pooling' { 1 }
    'tensor' { 3 }
    'half' { 8 }
    default { 16 }
}
if (@($report.results).Count -ne $expected) {
    throw "Expected $expected numerical results; got $(@($report.results).Count)"
}
if (@($report.results | Where-Object status -ne 'pass').Count) {
    throw 'Some reported GPU operations failed correctness checks.'
}
$modules = @(
    @{ data=$report.loaded_cudnn_bridge; path=$BridgeDll },
    @{ data=$report.loaded_hip; path=$hip },
    @{ data=$report.loaded_miopen; path=$miopen }
)
foreach ($module in $modules) {
    $loaded = $module.data
    if ($loaded.status -ne 'loaded') { throw 'Loaded GPU module identity missing.' }
    $actual = (Get-FileHash -LiteralPath $module.path -Algorithm SHA256).Hash
    if ($loaded.sha256 -ine $actual) {
        throw "Loaded module identity mismatch: $($module.path)"
    }
    if ([IO.Path]::GetFullPath($loaded.path) -ine [IO.Path]::GetFullPath($module.path)) {
        throw "Loaded GPU module path differs from staged path: $($loaded.path)"
    }
}
Write-Output "PASS: $expected bounded GPU numerical checks ($Only), loaded DLL identity verified."
Write-Output "Evidence: $reportPath"

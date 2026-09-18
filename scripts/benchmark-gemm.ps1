param(
    [string]$RuntimeRoot = ".runtime",
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [int[]]$Sizes = @(1024, 2048, 4096),
    [int]$Repetitions = 4,
    [int]$Warmup = 10,
    [int]$Iterations = 30,
    [double]$MaxMedianOverheadPercent = 20.0,
    [switch]$Strict,
    [string]$ReportPath
)
$ErrorActionPreference='Stop'
if ($Repetitions -lt 2 -or ($Repetitions % 2) -ne 0) { throw '-Repetitions must be an even integer >= 2 so execution order is balanced.' }
$RuntimeRoot=(Resolve-Path $RuntimeRoot).Path
$config=Get-Content (Join-Path $RuntimeRoot 'runtime-config.json') -Raw | ConvertFrom-Json
$zluda=[string]$config.zluda_root; $hip=[string]$config.hip_root
if(-not(Test-Path $PythonExe)){throw "Python not found: $PythonExe"}
if(-not(Test-Path $zluda)){throw "ZLUDA root not found: $zluda"}
if(-not(Test-Path $hip)){throw "HIP root not found: $hip"}
if(-not $ReportPath){$ReportPath=Join-Path $RuntimeRoot 'gemm-performance.json'}
$arch=if($config.gpu -and $config.gpu.arch){[string]$config.gpu.arch}else{throw 'GPU architecture missing from runtime config.'}
$repoRoot=Split-Path $PSScriptRoot -Parent
$source=Join-Path $repoRoot 'native\perf\rocblas_gemm_bench.cpp'
$pyBench=Join-Path $PSScriptRoot 'gemm_benchmark.py'
$perfDir=Join-Path $RuntimeRoot 'perf'; New-Item -ItemType Directory -Force $perfDir|Out-Null
$nativeExe=Join-Path $perfDir 'rocblas_gemm_bench.exe'
$hipcc=Join-Path $hip 'bin\hipcc.exe'
if(-not(Test-Path $hipcc)){throw "hipcc not found: $hipcc"}

Write-Host "Building native rocBLAS baseline for $arch..."
& $hipcc $source "--offload-arch=$arch" '-std=c++17' '-Wno-ignored-attributes' "-I$hip\include" "-L$hip\lib" -lrocblas -O3 -o $nativeExe
if($LASTEXITCODE -ne 0 -or -not(Test-Path $nativeExe)){throw 'Native rocBLAS benchmark build failed.'}

$oldPath=$env:PATH; $oldHip=$env:HIP_PATH; $oldRoc=$env:ROCBLAS_TENSILE_LIBPATH; $oldLt=$env:HIPBLASLT_TENSILE_LIBPATH; $oldCc=$env:ZLUDA_CC
try {
    $env:HIP_PATH=$hip; $env:ZLUDA_CC=if($config.zluda_cc){[string]$config.zluda_cc}else{'8.6'}
    $env:ROCBLAS_TENSILE_LIBPATH=Join-Path $hip 'bin\rocblas\library'
    $lt=Join-Path $hip 'bin\hipblaslt\library'; $ltArch=Join-Path $lt $arch; if(Test-Path $ltArch){$lt=$ltArch}; $env:HIPBLASLT_TENSILE_LIBPATH=$lt
    $env:PATH="$hip\bin;$zluda;$oldPath"
    function Invoke-NativeGemm([int]$n) {
        $text = & $nativeExe $n $Warmup $Iterations
        if($LASTEXITCODE -ne 0){throw "Native rocBLAS benchmark failed for n=$n"}
        return ($text | Select-Object -Last 1 | ConvertFrom-Json)
    }
    function Invoke-ZludaGemm([int]$n) {
        $text = & (Join-Path $zluda 'zluda.exe') -- $PythonExe $pyBench --n $n --warmup $Warmup --iterations $Iterations
        if($LASTEXITCODE -ne 0){throw "ZLUDA benchmark failed for n=$n"}
        return ($text | Select-Object -Last 1 | ConvertFrom-Json)
    }

    $rows=@()
    foreach($n in $Sizes){
        foreach($rep in 1..$Repetitions){
            # Alternate execution order to reduce boost/thermal/order bias.
            if($rep % 2 -eq 1){
                $order='direct-first'; $native=Invoke-NativeGemm $n; $compat=Invoke-ZludaGemm $n
            } else {
                $order='zluda-first'; $compat=Invoke-ZludaGemm $n; $native=Invoke-NativeGemm $n
            }
            $rows += [pscustomobject][ordered]@{n=$n;rep=$rep;order=$order;direct_ms=[double]$native.wall_ms_per_gemm;zluda_ms=[double]$compat.wall_ms_per_gemm;direct_event_ms=[double]$native.event_ms_per_gemm;zluda_event_ms=[double]$compat.event_ms_per_gemm;direct_wall_tflops=[double]$native.wall_tflops;zluda_wall_tflops=[double]$compat.wall_tflops;overhead_percent=(([double]$compat.wall_ms_per_gemm/[double]$native.wall_ms_per_gemm)-1)*100.0}
        }
    }
} finally {
    $env:PATH=$oldPath; $env:HIP_PATH=$oldHip; $env:ROCBLAS_TENSILE_LIBPATH=$oldRoc; $env:HIPBLASLT_TENSILE_LIBPATH=$oldLt; $env:ZLUDA_CC=$oldCc
}

function Median([double[]]$values){$s=@($values|Sort-Object);$count=$s.Count;if($count%2){return [double]$s[[int]($count/2)]};return ([double]$s[$count/2-1]+[double]$s[$count/2])/2.0}
$summary=@()
foreach($n in $Sizes){$group=@($rows|Where-Object {$_.n -eq $n});$direct=Median @($group.direct_ms);$compat=Median @($group.zluda_ms);$pairedOverhead=Median @($group.overhead_percent);$rawMedianRatio=(($compat/$direct)-1)*100.0;$summary += [pscustomobject][ordered]@{n=$n;direct_median_ms=$direct;zluda_median_ms=$compat;paired_median_overhead_percent=$pairedOverhead;raw_median_ratio_percent=$rawMedianRatio;paired_throughput_ratio=1.0/(1.0+$pairedOverhead/100.0);direct_median_tflops=(2.0*$n*$n*$n)/($direct*1e9);zluda_median_tflops=(2.0*$n*$n*$n)/($compat*1e9)}}
$summary|Format-Table @{L='N';E={$_.n}},@{L='rocBLAS median ms';E={'{0:N4}'-f $_.direct_median_ms}},@{L='ZLUDA median ms';E={'{0:N4}'-f $_.zluda_median_ms}},@{L='paired delta %';E={'{0:N2}'-f $_.paired_median_overhead_percent}},@{L='paired ratio';E={'{0:N3}'-f $_.paired_throughput_ratio}} -AutoSize
$result=[ordered]@{schema=2;method='wall-clock-synchronized-alternating-paired-median';generated_utc=(Get-Date).ToUniversalTime().ToString('o');runtime_root=$RuntimeRoot;gpu=$config.gpu;python=$PythonExe;warmup=$Warmup;iterations=$Iterations;repetitions=$Repetitions;rows=$rows;summary=$summary;max_allowed_median_overhead_percent=$MaxMedianOverheadPercent}
New-Item -ItemType Directory -Force (Split-Path $ReportPath -Parent)|Out-Null;$result|ConvertTo-Json -Depth 10|Set-Content -Encoding UTF8 $ReportPath;Write-Host "Report: $ReportPath"
$bad=@($summary|Where-Object {$_.paired_median_overhead_percent -gt $MaxMedianOverheadPercent});if($Strict -and $bad.Count){throw "Performance regression: paired median overhead exceeded $MaxMedianOverheadPercent% for one or more sizes."}

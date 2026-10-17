# Hardware-free regression for PowerShell case-insensitive $Tests/$tests shadowing.
# Executes the REAL report aggregation block with mocked per-probe PASS objects.
$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'scripts/test-functional.ps1'
$source = [IO.File]::ReadAllText($scriptPath)
$start = $source.IndexOf('$allowedTests = @(')
$end = $source.IndexOf("Write-Host ''", $start)
if ($start -lt 0 -or $end -le $start) { throw 'Cannot locate functional aggregation block.' }
$reportBlock = [scriptblock]::Create($source.Substring($start, $end - $start))

function Invoke-SyntheticReport {
    param([string[]]$Tests)
    $projectRevision = 'b' * 40
    $runtimeHashes = [ordered]@{ nvcuda_sha256 = 'a' * 64 }
    $PythonExe = 'python.exe'
    $torchCheck = 'torch 2.3.0+cu118'
    $config = [pscustomobject]@{ zluda_cc = '8.6'; gpu = [pscustomobject]@{arch='gfx1150'} }
    function Invoke-FunctionalProbe([string]$Name) {
        return [pscustomobject][ordered]@{
            test = $Name
            status = 'pass'
            timed_out = $false
            process_exit = 0
            process_crashed_after_result = $false
            result = [pscustomobject]@{
                ok = $true
                loaded_nvcuda = [pscustomobject]@{
                    status='loaded'
                    path='C:\ZLUDA\nvcuda.dll'
                    sha256=('a' * 64)
                }
            }
        }
    }
    . $reportBlock
    return ($result | ConvertTo-Json -Depth 9 | ConvertFrom-Json)
}

$two = Invoke-SyntheticReport -Tests @('matmul','sdpa_math')
if (-not $two.core_correctness_ok -or -not $two.correctness_ok) {
    throw 'Two passing probes incorrectly marked core or overall failure.'
}
if ($two.tests.Count -ne 2 -or $two.tests[0] -is [string]) {
    throw 'Selected functional tests were coerced to strings.'
}
if (@($two.tests | Where-Object {$_.test -eq 'matmul' -and $_.status -eq 'pass'}).Count -ne 1) {
    throw 'GEMM core test missing or no longer a structured object.'
}
if (@($two.tests | Where-Object {$_.test -eq 'sdpa_math' -and $_.status -eq 'pass'}).Count -ne 1) {
    throw 'SDPA math test missing.'
}
if ($two.tests[1].result.loaded_nvcuda.sha256 -ne ('a' * 64)) {
    throw 'In-process loaded driver hash missing from serialized functional report.'
}

$default = Invoke-SyntheticReport
if ($default.tests.Count -ne 5 -or $default.selected_tests.Count -ne 5 -or -not $default.core_correctness_ok) {
    throw 'Default functional test set or report structure regressed.'
}
if (@($default.tests | Where-Object {$_ -is [string]}).Count) {
    throw 'Default reports were coerced to System.String.'
}
Write-Output 'PASS: typed -Tests parameter and structured report entries coexist in two-test and default modes'

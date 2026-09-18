[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ZludaSourceRoot,
    [switch]$IncludeCandidates,
    [switch]$CheckOnly,
    [string]$SeriesPath
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $SeriesPath) {
    $SeriesPath = Join-Path $repo 'patches\zluda-v7-preview10\series.json'
}
$SeriesPath = [System.IO.Path]::GetFullPath($SeriesPath)
$ZludaSourceRoot = [System.IO.Path]::GetFullPath($ZludaSourceRoot)

if (-not (Test-Path $SeriesPath)) { throw "Missing patch series: $SeriesPath" }
if (-not (Test-Path (Join-Path $ZludaSourceRoot '.git'))) { throw "Not a Git checkout: $ZludaSourceRoot" }

$series = Get-Content $SeriesPath -Raw | ConvertFrom-Json
if ($series.schema -ne 1) { throw "Unsupported patch series schema: $($series.schema)" }

$git = Get-Command git.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $git) { $git = Get-Command git -ErrorAction Stop | Select-Object -First 1 }

$head = (& $git.Source -C $ZludaSourceRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'git rev-parse HEAD failed.' }
$expected = [string]$series.upstream.commit
if ($head -ne $expected) {
    throw "ZLUDA source commit mismatch. Expected $expected, got $head"
}

$patchRoot = Split-Path $SeriesPath -Parent
$selected = @($series.patches | Sort-Object {[int]$_.order} | Where-Object {
    $_.status -eq 'validated' -or $IncludeCandidates
})
if ($selected.Count -eq 0) { throw 'Patch series selected zero patches.' }

$applied = @()
foreach ($entry in $selected) {
    $file = Join-Path $patchRoot ([string]$entry.file)
    if (-not (Test-Path $file)) { throw "Missing patch file: $file" }

    Write-Host ("[{0}] {1} ({2})" -f $entry.order, $entry.file, $entry.status)
    & $git.Source -C $ZludaSourceRoot apply --check --whitespace=error-all $file
    if ($LASTEXITCODE -ne 0) { throw "Patch check failed: $($entry.file)" }

    if (-not $CheckOnly) {
        & $git.Source -C $ZludaSourceRoot apply $file
        if ($LASTEXITCODE -ne 0) { throw "Patch apply failed: $($entry.file)" }
    }

    $applied += [pscustomobject][ordered]@{
        order = [int]$entry.order
        file = [string]$entry.file
        status = [string]$entry.status
        purpose = [string]$entry.purpose
        required_probes = @($entry.required_probes)
    }
}

if (-not $CheckOnly) {
    & $git.Source -C $ZludaSourceRoot diff --check
    if ($LASTEXITCODE -ne 0) { throw 'git diff --check failed after patch application.' }
}

$result = [pscustomobject][ordered]@{
    schema = 1
    upstream_commit = $head
    include_candidates = [bool]$IncludeCandidates
    check_only = [bool]$CheckOnly
    applied = $applied
}
$result | ConvertTo-Json -Depth 8

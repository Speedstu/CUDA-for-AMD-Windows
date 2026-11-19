# Shared source of truth for detecting the MIOpen backend without loading a DLL.
# Dot-source from scripts/run-zluda.ps1 and scripts/test-capabilities.ps1.
# A matching file is a path-level precondition, NOT proof of GPU correctness.
function Get-CudaAmdMiopenState {
    [CmdletBinding()]
    param(
        [string]$HipRoot,
        [string]$ZludaRoot,
        [string]$TargetDir,
        [string]$ProgramPath,
        [string]$InheritedHipPath,
        [string]$ConfiguredMiopenDll
    )

    if ($ConfiguredMiopenDll -and (Test-Path -LiteralPath $ConfiguredMiopenDll -PathType Leaf)) {
        return [pscustomobject]@{
            bridge_ready = $true
            miopen_present = $true
            miopen_path = [IO.Path]::GetFullPath($ConfiguredMiopenDll)
            source = 'explicit_env'
        }
    }

    $states = @()
    if ($TargetDir) { $states += (Join-Path $TargetDir '.cudaamd-cudnn-bridge.json') }
    if ($TargetDir -and $ProgramPath -and
        [IO.Path]::GetFileName($TargetDir) -ieq 'Scripts' -and
        [IO.Path]::GetFileName($ProgramPath) -match '^python(?:w|[0-9.]*)?\.exe$') {
        $venv = Split-Path $TargetDir -Parent
        $states += (Join-Path $venv 'Lib\site-packages\torch\lib\.cudaamd-cudnn-bridge.json')
    }
    foreach ($statePath in $states) {
        if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) { continue }
        try {
            $state = Get-Content -LiteralPath $statePath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
            if ($state.miopen_dll -and
                (Test-Path -LiteralPath ([string]$state.miopen_dll) -PathType Leaf)) {
                return [pscustomobject]@{
                    bridge_ready = $true
                    miopen_present = $true
                    miopen_path = [IO.Path]::GetFullPath([string]$state.miopen_dll)
                    source = 'staged_bridge'
                }
            }
        } catch {
            # A stale, invalid or incomplete state must never authorize cuDNN.
        }
    }

    foreach ($root in @($HipRoot, $InheritedHipPath, $ZludaRoot, $TargetDir)) {
        if (-not $root) { continue }
        foreach ($relative in @('MIOpen.dll','bin\MIOpen.dll')) {
            $candidate = Join-Path $root $relative
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return [pscustomobject]@{
                    bridge_ready = $false
                    miopen_present = $true
                    miopen_path = [IO.Path]::GetFullPath($candidate)
                    source = 'existing_backend_file'
                }
            }
        }
    }
    return [pscustomobject]@{
        bridge_ready = $false
        miopen_present = $false
        miopen_path = $null
        source = 'missing'
    }
}

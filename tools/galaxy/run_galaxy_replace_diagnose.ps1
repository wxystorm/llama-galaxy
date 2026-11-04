<#
  Reproduce and localize first-layer causal SP Attention REPLACE divergence.

  This intentionally does NOT run 2/4-layer replacements: REPLACE-1
  already failed independent logits parity, so promoting it is unsafe.
  Runs same real GGUF prompt twice with identical probe/dump boundaries:
  pure TP baseline and native SP REPLACE layer0. The GLG1 snapshots
  show the earliest boundary at which the numbers change.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][System.Collections.IDictionary]$Common,
    [string]$PythonExe = "python",
    [string]$OutputDir = ".\galaxy-sp-results\causal-replace-diagnosis",
    [int]$Layer = 0
)

$ErrorActionPreference = "Stop"
if ($Layer -ne 0) { throw "First diagnostic is limited to the failing Layer 0 REPLACE" }
foreach ($k in @("Model","PromptFile","Rpc")) {
    if (@($Common.Keys) -notcontains $k) { throw "Missing Common.$k" }
}
$smoke = Join-Path $PSScriptRoot "run_galaxy_sp_smoke.ps1"
$diff = Join-Path $PSScriptRoot "galaxy-activation-diff.py"
if (-not (Test-Path -LiteralPath $diff)) {
    throw "Missing diagnostic comparator: $diff"
}
$root = Join-Path $OutputDir ("diagnosis-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
New-Item -ItemType Directory -Force -Path $root | Out-Null
$root = (Resolve-Path -LiteralPath $root).Path

# No mutation of caller's session environment or $common. The smoke
# runner temporarily sets and restores all model env variables.
$shared = @{}
foreach ($key in $Common.Keys) { $shared[$key] = $Common[$key] }
foreach ($k in @("PureTP","LiveCausalLayer","LiveCausalPrefixLayers",
                "LiveCausalReplace","OutputDir","DumpLogits","ProbeQkvLayers",
                "DumpStages","DumpLayers","DumpFfnOps","Profile","Trace")) {
    [void]$shared.Remove($k)
}
$shared["Galaxy2"] = $false
$shared["Galaxy2FfnSpLayerSet"] = ""
$shared["FinalLayerSP"] = $false
$shared["DumpLogits"] = $true
$shared["ProbeQkvLayers"] = "0"
$shared["DumpStages"] = "0"
$shared["DumpLayers"] = "0"
$shared["DumpFfnOps"] = ""
$shared["Profile"] = $false
$shared["Trace"] = $false

function One-Run([string]$label,[bool]$replace) {
    $params = @{}
    foreach ($key in $shared.Keys) { $params[$key] = $shared[$key] }
    $params["LiveCausalLayer"] = -1
    $params["LiveCausalPrefixLayers"] = if ($replace) { 1 } else { 0 }
    $params["LiveCausalReplace"] = $replace
    $params["OutputDir"] = Join-Path $root $label
    Write-Host "[GALAXY_REPLACE_DIAG] stage=$label event=START"
    & $smoke @params -PureTP
    $runs = @(Get-ChildItem -LiteralPath $params["OutputDir"] -Directory -Filter "tp-*")
    if ($runs.Count -ne 1) {
        throw "Expected one completed $label run, got $($runs.Count)"
    }
    $run = $runs[0].FullName
    foreach ($p in @(
        (Join-Path $run "real-qkv\tokens-98\kqv_out-0.f32"),
        (Join-Path $run "stages\ffn_inp-0.f32"),
        (Join-Path $run "layers\l_out-0.f32"),
        (Join-Path $run "first_logits.f32"))) {
        if (-not (Test-Path -LiteralPath $p)) { throw "Missing stage probe: $p" }
    }
    Write-Host "[GALAXY_REPLACE_DIAG] stage=$label event=PASS dir=$run"
    return $run
}
$tp = One-Run "TP" $false
$sp = One-Run "REPLACE-1" $true
$json = Join-Path $root "stage-diff.json"
& $PythonExe $diff --tp-run $tp --sp-run $sp --tokens 98 --layer 0 --json $json
if ($LASTEXITCODE -ne 0) { throw "Stage comparison failed. Inspect $json" }
Write-Host "[GALAXY_REPLACE_DIAG_SUM] tp_run=$tp"
Write-Host "[GALAXY_REPLACE_DIAG_SUM] sp_run=$sp"
Write-Host "[GALAXY_REPLACE_DIAG_SUM] result=$json"
Write-Host "[GALAXY_REPLACE_DIAG_SUM] status=MEASURED (not REPLACE parity certified)"

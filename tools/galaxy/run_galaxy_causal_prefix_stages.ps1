<#
  Causal SP Attention incremental rollout for an unchanged Llama3.2-3B
  PC CUDA + Phone RPC configuration.

  Stage order (default): reference TP, SHADOW 1, SHADOW 2.
  With -IncludeReplace: TP, SHADOW 1, REPLACE 1, SHADOW 2, REPLACE 2...
  -MaxPrefix 4 also includes SHADOW/REPLACE 4 after the 2-layer pass.

  Every smoke MUST emit its own complete per-layer witness. Every stage
  MUST also pass full first-token vocabulary logits comparison against
  the SAME independent TP reference, or execution stops before the
  following stage. No background work or implicit activation of REPLACE.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][System.Collections.IDictionary]$Common,
    [ValidateSet(1,2,4)][int]$MaxPrefix = 2,
    [switch]$IncludeReplace,
    [string]$OutputDir = ".\galaxy-sp-results\causal-prefix-staged",
    [double]$MaxAbs = 0.10,
    [double]$MaxRmse = 0.02
)

$ErrorActionPreference = "Stop"
if ([double]::IsNaN($MaxAbs) -or [double]::IsInfinity($MaxAbs) -or $MaxAbs -lt 0 -or
        [double]::IsNaN($MaxRmse) -or [double]::IsInfinity($MaxRmse) -or $MaxRmse -lt 0) {
    throw "MaxAbs and MaxRmse must be finite nonnegative numbers"
}
foreach ($required in @("Model","PromptFile","Rpc")) {
    if (@($Common.Keys) -notcontains $required) {
        throw "Common must include $required"
    }
}

$smoke = Join-Path $PSScriptRoot "run_galaxy_sp_smoke.ps1"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$compare = Join-Path $repoRoot "build-galaxy\bin\Release\galaxy-logits-compare.exe"
if (-not (Test-Path -LiteralPath $compare)) {
    throw "Missing logits comparator: $compare"
}
if (-not (Test-Path -LiteralPath $smoke)) {
    throw "Missing smoke runner: $smoke"
}

$root = Join-Path $OutputDir ("prefix-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
New-Item -ItemType Directory -Force -Path $root | Out-Null
$root = (Resolve-Path -LiteralPath $root).Path

# Make a fresh splat without mutating the caller's $common hashtable.
$shared = @{}
foreach ($key in $Common.Keys) { $shared[$key] = $Common[$key] }
foreach ($name in @("PureTP","LiveCausalLayer","LiveCausalPrefixLayers",
                    "LiveCausalReplace","OutputDir","DumpLogits",
                    "ProbeQkvLayers","Profile","Trace")) {
    [void]$shared.Remove($name)
}
$shared["DumpLogits"] = $true
$shared["ProbeQkvLayers"] = ""
$shared["Galaxy2"] = $false
$shared["Galaxy2FfnSpLayerSet"] = ""
$shared["FinalLayerSP"] = $false
$shared["Profile"] = $false
$shared["Trace"] = $false
$shared["DumpLayers"] = ""
$shared["DumpStages"] = ""
$shared["DumpFfnOps"] = ""

$results = @()
function Save-StageManifest {
    $manifest = [ordered]@{
        status = "IN_PROGRESS"
        max_prefix = $MaxPrefix
        include_replace = [bool]$IncludeReplace
        limits = @{ max_abs = $MaxAbs; max_rmse = $MaxRmse }
        model = [string]$shared["Model"]
        prompt = [string]$shared["PromptFile"]
        rpc = [string]$shared["Rpc"]
        head_split = [string]$shared["HeadSplit"]
        ffn_split = [string]$shared["FfnSplit"]
        stages = @($results)
    }
    $manifest | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (Join-Path $root "manifest.json") -Encoding UTF8
}

function Latest-Run([string]$stageDir) {
    $runs = @(Get-ChildItem -LiteralPath $stageDir -Directory -Filter "tp-*" |
        Sort-Object CreationTime -Descending)
    if ($runs.Count -ne 1) {
        throw "Expected exactly one TP run in $stageDir; got $($runs.Count)"
    }
    return $runs[0]
}

# Baseline: no Shadow and no REPLACE, same model, same prompt and splits.
$base = @{}; foreach ($key in $shared.Keys) { $base[$key] = $shared[$key] }
$base["LiveCausalLayer"] = -1
$base["LiveCausalPrefixLayers"] = 0
$base["LiveCausalReplace"] = $false
$base["OutputDir"] = Join-Path $root "baseline"
Write-Host "[GALAXY_PREFIX_STAGE] stage=TP_BASELINE event=START"
& $smoke @base -PureTP
$baseline = Latest-Run $base["OutputDir"]
$tpLogits = Join-Path $baseline.FullName "first_logits.f32"
if (-not (Test-Path -LiteralPath $tpLogits)) {
    throw "Pure TP baseline logits missing: $tpLogits"
}
$results += [ordered]@{ name="TP_BASELINE"; prefix=0; mode="TP"; dir=$baseline.FullName; status="PASS" }
Save-StageManifest
Write-Host "[GALAXY_PREFIX_STAGE] stage=TP_BASELINE event=PASS logits=$tpLogits"

function Invoke-Prefix([int]$count, [bool]$replace) {
    $mode = if ($replace) { "REPLACE" } else { "SHADOW" }
    $name = "$mode-$count"
    $stageParams = @{}; foreach ($key in $shared.Keys) { $stageParams[$key] = $shared[$key] }
    $stageParams["LiveCausalLayer"] = -1
    $stageParams["LiveCausalPrefixLayers"] = $count
    $stageParams["LiveCausalReplace"] = $replace
    $stageParams["OutputDir"] = Join-Path $root $name
    Write-Host "[GALAXY_PREFIX_STAGE] stage=$name event=START layers=0..$($count-1)"
    & $smoke @stageParams -PureTP
    $run = Latest-Run $stageParams["OutputDir"]
    $candidate = Join-Path $run.FullName "first_logits.f32"
    $json = Join-Path $run.FullName "logits-to-tp.json"
    & $compare --tp $tpLogits --sp $candidate --max-abs $MaxAbs --max-rmse $MaxRmse --json $json
    if ($LASTEXITCODE -ne 0) {
        throw "Numerical parity FAILED for $name. Inspect $json and $($run.FullName)"
    }
    $report = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
    if ($report.status -ne "PASS") {
        throw "Unverified logits parity for $name. Inspect $json"
    }
    $script:results += [ordered]@{
        name=$name; prefix=$count; mode=$mode;
        dir=$run.FullName; comparison=$json; status="PASS"
    }
    Save-StageManifest
    Write-Host "[GALAXY_PREFIX_STAGE] stage=$name event=PASS comparison=$json"
}

foreach ($count in @(1,2,4)) {
    if ($count -gt $MaxPrefix) { continue }
    Invoke-Prefix $count $false
    if ($IncludeReplace) { Invoke-Prefix $count $true }
}

$manifestPath = Join-Path $root "manifest.json"
$data = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$data.status = "PASS"
$data | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Host "[GALAXY_PREFIX_STAGE_SUM] stages=$($results.Count) max_prefix=$MaxPrefix replace=$([bool]$IncludeReplace) status=PASS"
Write-Host "[GALAXY_PREFIX_STAGE_SUM] manifest=$manifestPath"

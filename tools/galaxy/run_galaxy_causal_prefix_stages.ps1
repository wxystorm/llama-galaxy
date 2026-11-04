<#
  Causal SP Attention incremental rollout for an unchanged Llama3.2-3B
  PC CUDA + Phone RPC configuration.

  Stage order (default): reference TP, SHADOW 1, SHADOW 2.
  With -IncludeReplace: TP, SHADOW 1, REPLACE 1, SHADOW 2, REPLACE 2...
  -MaxPrefix 8/16/28 also covers the larger model prefixes.

  Every smoke MUST emit its own complete per-layer execution witness.
  Strict logits parity is the default; explicit -ContinueOnLogitsMismatch
  records numerical failures and continues observing larger prefixes.
  Never call a failing logits comparison a PASS.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][System.Collections.IDictionary]$Common,
    [ValidateSet(1,2,4,8,16,28)][int]$MaxPrefix = 2,
    [switch]$IncludeReplace,
    [switch]$ContinueOnLogitsMismatch,
    [switch]$Trace,
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
$shared["Trace"] = [bool]$Trace
$shared["DumpLayers"] = ""
$shared["DumpStages"] = ""
$shared["DumpFfnOps"] = ""

$results = @()
$timings = @()
$driftCount = 0
function Save-StageManifest {
    $manifest = [ordered]@{
        status = "IN_PROGRESS"
        max_prefix = $MaxPrefix
        include_replace = [bool]$IncludeReplace
        continue_on_logits_mismatch = [bool]$ContinueOnLogitsMismatch
        trace = [bool]$Trace
        numerical_failures = $script:driftCount
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
    # PS 7 can be configured to throw on nonzero native exit codes.
    # Comparator exit=2 is an EXPECTED observed numerical mismatch,
    # so inspect its exit code/JSON explicitly instead of treating it
    # as a command-launch failure. Other errors still fail below.
    $PSNativeCommandUseErrorActionPreference = $false
    & $compare --tp $tpLogits --sp $candidate --max-abs $MaxAbs --max-rmse $MaxRmse --json $json
    $compareExit = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $json)) {
        throw "Logits comparator did not produce a report for $name. Inspect $($run.FullName)"
    }
    $report = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
    $numericalPass = $compareExit -eq 0 -and $report.status -eq "PASS"
    if (-not $numericalPass) {
        # Only an explicit numeric comparison failure (exit 2) is
        # eligible to continue. IO/parse/launch errors are always fatal.
        if ($compareExit -ne 2 -or $report.status -ne "FAIL") {
            throw "Logits comparator error for $name (exit=$compareExit). Inspect $json"
        }
        if (-not $ContinueOnLogitsMismatch) {
            throw "Numerical parity FAILED for $name. Inspect $json and $($run.FullName)"
        }
        $script:driftCount++
        Write-Warning "[GALAXY_PREFIX_STAGE] stage=$name numerical_parity=FAIL rmse=$($report.rmse) max_abs=$($report.max_abs) top1=$($report.sp_top1) continuing=EXPLICIT_OBSERVATION_ONLY"
    }

    # Extract one timing row for EACH selected Attention layer from the
    # real Meta execution; the smoke already checks both rank witnesses.
    $liveLog = Join-Path $run.FullName "stderr.log"
    $timedLayers = @{}
    foreach ($line in (Get-Content -LiteralPath $liveLog)) {
        if ($line -notmatch '^\[GALAXY_CAUSAL_LIVE\] layer=[0-9]+ tokens=[0-9]+ mode=(SHADOW|REPLACE)\b' -or
                $line -notmatch '\btp_sp_ms=') { continue }
        $fields = @{}
        foreach ($matched in [regex]::Matches($line,'([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)')) {
            $fields[$matched.Groups[1].Value] = $matched.Groups[2].Value
        }
        if ($fields["mode"] -ne $mode) { continue }
        $layer = [int]$fields["layer"]
        if ($layer -lt 0 -or $layer -ge $count -or $timedLayers.ContainsKey($layer)) {
            throw "Duplicate/unexpected layer timing in ${liveLog}: $line"
        }
        $timedLayers[$layer] = $true
        $script:timings += [pscustomobject][ordered]@{
            stage = $name
            prefix_layers = $count
            mode = $mode
            layer = $layer
            tokens = [int]$fields["tokens"]
            tp_sp_ms = [double]::Parse($fields["tp_sp_ms"],[System.Globalization.CultureInfo]::InvariantCulture)
            kv_gather_ms = [double]::Parse($fields["kv_gather_ms"],[System.Globalization.CultureInfo]::InvariantCulture)
            native_ready_wall_ms = [double]::Parse($fields["native_ready_wall_ms"],[System.Globalization.CultureInfo]::InvariantCulture)
            sp_tp_ms = [double]::Parse($fields["sp_tp_ms"],[System.Globalization.CultureInfo]::InvariantCulture)
            logits_parity = [string]$report.status
        }
    }
    if ($timedLayers.Count -ne $count) {
        throw "Expected $count timed layers in $name, got $($timedLayers.Count): $liveLog"
    }
    $script:timings | Sort-Object prefix_layers, mode, layer |
        Export-Csv -LiteralPath (Join-Path $root "per-layer-timings.csv") -NoTypeInformation -Encoding UTF8

    $status = if ($numericalPass) { "PASS" } else { "EXECUTED_NUMERICAL_FAIL" }
    $script:results += [ordered]@{
        name=$name; prefix=$count; mode=$mode
        dir=$run.FullName; comparison=$json
        numerical_parity=[string]$report.status
        logits_rmse=[double]$report.rmse
        logits_max_abs=[double]$report.max_abs
        reference_top1=[int]$report.tp_top1
        candidate_top1=[int]$report.sp_top1
        status=$status
    }
    Save-StageManifest
    Write-Host "[GALAXY_PREFIX_STAGE] stage=$name event=$status numerical_parity=$($report.status) comparison=$json"
}

foreach ($count in @(1,2,4,8,16,28)) {
    if ($count -gt $MaxPrefix) { continue }
    Invoke-Prefix $count $false
    if ($IncludeReplace) { Invoke-Prefix $count $true }
}

$manifestPath = Join-Path $root "manifest.json"
$data = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$finalStatus = if ($driftCount -gt 0) { "EXPERIMENTAL_NUMERICAL_DRIFT" } else { "PASS" }
$data.status = $finalStatus
$data | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Host "[GALAXY_PREFIX_STAGE_SUM] stages=$($results.Count) max_prefix=$MaxPrefix replace=$([bool]$IncludeReplace) numerical_failures=$driftCount status=$finalStatus"
Write-Host "[GALAXY_PREFIX_STAGE_SUM] manifest=$manifestPath"
Write-Host "[GALAXY_PREFIX_STAGE_SUM] per_layer_timings=$(Join-Path $root 'per-layer-timings.csv')"

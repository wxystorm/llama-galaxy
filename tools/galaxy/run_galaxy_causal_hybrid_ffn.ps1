<#
 Real 2-device Galaxy experimental hybrid:
  prefill Attention native SP REPLACE + Galaxy2 SP FFN on selected layers.
 Execute-only gate. No logits comparison and no changes to Decode.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][System.Collections.IDictionary]$Common,
    [ValidateSet(1,2,4,8,16,28)][int]$AttentionPrefix = 8,
    [string]$FfnSpLayerSet = "0",
    [switch]$FinalLayerSP,
    [switch]$Trace,
    [string]$OutputDir = ".\\galaxy-sp-results\\causal-hybrid-ffn"
)
$ErrorActionPreference = "Stop"
# Llama 3.2 3B has 28 layers, indexed 0..27. The final-layer
# option must be used with the full Attention REPLACE prefix.
if ($FinalLayerSP -and $AttentionPrefix -ne 28) {
    throw "FinalLayerSP requires AttentionPrefix 28, including Attention L27"
}
if ([string]::IsNullOrWhiteSpace($FfnSpLayerSet) -or $FfnSpLayerSet -eq "none") {
    throw "Choose at least one FFN-SP layer"
}
$lastEligible = if ($FinalLayerSP) { 27 } else { 26 }
$maxSelected = [Math]::Min($AttentionPrefix - 1, $lastEligible)
# 'all' selects the eligible layers within this Attention prefix, NOT
# the unselected rest of the 28-layer model.
$resolvedSet = if ($FfnSpLayerSet -eq "all") { "0-$maxSelected" } else { $FfnSpLayerSet }
$chosen = @{}
foreach ($part in $resolvedSet.Split(",")) {
    if ($part -notmatch '^(\d+)(?:-(\d+))?$') {
        throw "Bad FFN SP layer/range: $part"
    }
    $first = [int]$Matches[1]
    $last = if ($Matches[2]) { [int]$Matches[2] } else { $first }
    if ($last -lt $first -or $last -gt $maxSelected) {
        throw "FFN SP layer $part must be within 0..$maxSelected (AttentionPrefix=$AttentionPrefix, FinalLayerSP=$([bool]$FinalLayerSP))"
    }
    for ($il=$first; $il -le $last; ++$il) {
        if ($chosen.ContainsKey($il)) { throw "Duplicate FFN SP layer $il" }
        $chosen[$il] = $true
    }
}
if ($chosen.Count -eq 0) { throw "Select at least one FFN SP layer" }
if ($FinalLayerSP -and -not $chosen.ContainsKey(27)) {
    throw "FinalLayerSP requires including FFN layer 27; use FfnSpLayerSet '0-27' or 'all'"
}
foreach ($key in @("Model","PromptFile","Rpc")) {
    if (@($Common.Keys) -notcontains $key) {
        throw "Common must include $key"
    }
}
$runRoot = Join-Path $OutputDir ("hybrid-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
$runRoot = (New-Item -ItemType Directory -Force -Path $runRoot).FullName
$smoke = Join-Path $PSScriptRoot "run_galaxy_sp_smoke.ps1"
$opts = @{}
foreach ($key in $Common.Keys) { $opts[$key] = $Common[$key] }
foreach ($name in @("PureTP","Galaxy2","Galaxy2FfnSpLayers","Galaxy2FfnSpLayerSet",
                    "LiveCausalLayer","LiveCausalPrefixLayers","LiveCausalReplace",
                    "FinalLayerSP","DumpLogits","OutputDir","CausalHybridFfnSp","Trace")) {
    [void]$opts.Remove($name)
}
$opts["Galaxy2"] = $true
$opts["Galaxy2FfnSpLayerSet"] = $resolvedSet
$opts["LiveCausalLayer"] = -1
$opts["LiveCausalPrefixLayers"] = $AttentionPrefix
$opts["LiveCausalReplace"] = $true
$opts["CausalHybridFfnSp"] = $true
$opts["FinalLayerSP"] = [bool]$FinalLayerSP
$opts["DumpLogits"] = $false
$opts["Trace"] = [bool]$Trace
$opts["OutputDir"] = $runRoot
Write-Host "[GALAXY_CAUSAL_HYBRID] event=START attention_prefix=$AttentionPrefix ffn_sp_layer_set=$resolvedSet final_layer_sp=$([bool]$FinalLayerSP)"
& $smoke @opts
$run = @(Get-ChildItem -LiteralPath $runRoot -Directory -Filter "hybrid-*" | Sort-Object LastWriteTime -Descending)
if ($run.Count -ne 1) {
    throw "Expected one hybrid smoke output in $runRoot, got $($run.Count)"
}
$stderr = Join-Path $run[0].FullName "stderr.log"
$witness = @(Select-String -LiteralPath $stderr -Pattern '^\[GALAXY_CAUSAL_HYBRID_FFN_SUM\] mode=REPLACE .*status=PASS$')
if ($witness.Count -ne 1) {
    throw "Missing or duplicate hybrid summary: $stderr"
}
$line = $witness[0].Line
$expectedFinal = if ($FinalLayerSP) { 1 } else { 0 }
if ($line -notmatch "attention_replaced=$AttentionPrefix\b" -or
        $line -notmatch "ffn_sp_layers=$($chosen.Count)\b" -or
        $line -notmatch "final_layer_sp=$expectedFinal\b" -or
        ($FinalLayerSP -and ($line -notmatch "final_ffn_sp=1\b" -or
                             $line -notmatch "deferred_output_select=1\b"))) {
    throw "Hybrid execution counts/final-output contract mismatch (expected Attention=$AttentionPrefix FFN-SP=$($chosen.Count) Final=$expectedFinal): $line"
}
$manifest = [ordered]@{
    status = "EXECUTION_PASS_NUMERICAL_UNCHECKED"
    attention_prefix = $AttentionPrefix
    ffn_sp_layer_set = $resolvedSet
    ffn_sp_count = $chosen.Count
    final_layer_sp = [bool]$FinalLayerSP
    final_layer_ffn_sp = [bool]$FinalLayerSP
    deferred_output_select_verified = [bool]$FinalLayerSP
    execution_summary = $line
    trace = [bool]$Trace
    log_path = $stderr
    numerical_parity = "NOT_EVALUATED"
}
$manifest | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $runRoot "manifest.json") -Encoding UTF8
Write-Host "[GALAXY_CAUSAL_HYBRID] attention_replaced=$AttentionPrefix ffn_sp_layers=$($chosen.Count) final_layer_sp=$expectedFinal status=EXECUTION_PASS_NUMERICAL_UNCHECKED logs=$($run[0].FullName)"

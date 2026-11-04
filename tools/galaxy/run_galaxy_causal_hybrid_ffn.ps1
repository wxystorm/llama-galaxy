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
    [switch]$Trace,
    [string]$OutputDir = ".\\galaxy-sp-results\\causal-hybrid-ffn"
)
$ErrorActionPreference = "Stop"
if ($AttentionPrefix -le 0 -or $FfnSpLayerSet -eq "" -or $FfnSpLayerSet -eq "none" -or
        $FfnSpLayerSet -eq "all") {
    throw "Choose an explicit nonempty FFN SP layer set (e.g. 0 or 0-7). Final layer SP is not enabled."
}
$chosen = @{}
foreach ($part in $FfnSpLayerSet.Split(",")) {
    if ($part -notmatch '^(\d+)(?:-(\d+))?$') {
        throw "Bad FFN SP layer/range: $part"
    }
    $first = [int]$Matches[1]
    $last = if ($Matches[2]) { [int]$Matches[2] } else { $first }
    if ($last -lt $first -or $last -ge $AttentionPrefix -or $last -ge 27) {
        throw "FFN SP layer $part must be within the selected Attention prefix and below final layer 27"
    }
    for ($il=$first; $il -le $last; ++$il) {
        if ($chosen.ContainsKey($il)) { throw "Duplicate FFN SP layer $il" }
        $chosen[$il] = $true
    }
}
if ($chosen.Count -eq 0) { throw "Select at least one FFN SP layer" }
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
$opts["Galaxy2FfnSpLayerSet"] = $FfnSpLayerSet
$opts["LiveCausalLayer"] = -1
$opts["LiveCausalPrefixLayers"] = $AttentionPrefix
$opts["LiveCausalReplace"] = $true
$opts["CausalHybridFfnSp"] = $true
$opts["FinalLayerSP"] = $false
$opts["DumpLogits"] = $false
$opts["Trace"] = [bool]$Trace
$opts["OutputDir"] = $runRoot
Write-Host "[GALAXY_CAUSAL_HYBRID] event=START attention_prefix=$AttentionPrefix ffn_sp_layer_set=$FfnSpLayerSet"
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
if ($line -notmatch "attention_replaced=$AttentionPrefix\b" -or
        $line -notmatch "ffn_sp_layers=$($chosen.Count)\b") {
    throw "Hybrid execution counts mismatch (expected attention=$AttentionPrefix and FFN SP=$($chosen.Count)): $line"
}
$manifest = [ordered]@{
    status = "EXECUTION_PASS_NUMERICAL_UNCHECKED"
    attention_prefix = $AttentionPrefix
    ffn_sp_layer_set = $FfnSpLayerSet
    ffn_sp_count = $chosen.Count
    trace = [bool]$Trace
    log_path = $stderr
    numerical_parity = "NOT_EVALUATED"
}
$manifest | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $runRoot "manifest.json") -Encoding UTF8
Write-Host "[GALAXY_CAUSAL_HYBRID] attention_replaced=$AttentionPrefix ffn_sp_layers=$($chosen.Count) status=EXECUTION_PASS_NUMERICAL_UNCHECKED logs=$($run[0].FullName)"

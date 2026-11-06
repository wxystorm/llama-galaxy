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
    [switch]$NativeSpQkvL0,
    [ValidateSet(0,1,2,4,8,16,28)][int]$NativeSpQkvPrefixLayers = 0,
    [switch]$NativeSpQkvResidentWeights,
    [switch]$NativeSpQkvSkipTpPrefill,
    [switch]$Trace,
    [string]$OutputDir = ".\\galaxy-sp-results\\causal-hybrid-ffn"
)
$ErrorActionPreference = "Stop"
# Llama 3.2 3B has 28 layers, indexed 0..27. The final-layer
# option must be used with the full Attention REPLACE prefix.
if ($FinalLayerSP -and $AttentionPrefix -ne 28) {
    throw "FinalLayerSP requires AttentionPrefix 28, including Attention L27"
}
if ($NativeSpQkvL0 -and $NativeSpQkvPrefixLayers -gt 0) {
    throw "Choose NativeSpQkvL0 or NativeSpQkvPrefixLayers, not both"
}
$nativeCount = if ($NativeSpQkvL0) { 1 } else { $NativeSpQkvPrefixLayers }
if ($NativeSpQkvResidentWeights -and $nativeCount -eq 0) {
    throw "NativeSpQkvResidentWeights requires selecting native SP QKV layers"
}
if ($NativeSpQkvSkipTpPrefill -and (-not $NativeSpQkvResidentWeights -or $nativeCount -eq 0)) {
    throw "NativeSpQkvSkipTpPrefill requires resident SP QKV weights and a selected native prefix"
}
$residentExpected = if ($NativeSpQkvResidentWeights) { 1 } else { 0 }
if ($nativeCount -gt $AttentionPrefix) {
    throw "Native QKV-SP prefix must not exceed AttentionPrefix"
}
$nativeLayers = if ($nativeCount -gt 0) { @(0..($nativeCount - 1)) } else { @() }
$nativeLayerList = $nativeLayers -join ","
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
                    "FinalLayerSP","DumpLogits","OutputDir","CausalHybridFfnSp","NativeSpQkvL0","NativeSpQkvPrefixLayers","NativeSpQkvResidentWeights","NativeSpQkvSkipTpPrefill","RankLocalCollectives","Trace")) {
    [void]$opts.Remove($name)
}
$opts["Galaxy2"] = $true
$opts["Galaxy2FfnSpLayerSet"] = $resolvedSet
$opts["LiveCausalLayer"] = -1
$opts["LiveCausalPrefixLayers"] = $AttentionPrefix
$opts["LiveCausalReplace"] = $true
$opts["CausalHybridFfnSp"] = $true
$opts["NativeSpQkvL0"] = [bool]$NativeSpQkvL0
$opts["NativeSpQkvPrefixLayers"] = $NativeSpQkvPrefixLayers
$opts["NativeSpQkvResidentWeights"] = [bool]$NativeSpQkvResidentWeights
$opts["NativeSpQkvSkipTpPrefill"] = [bool]$NativeSpQkvSkipTpPrefill
$opts["FinalLayerSP"] = [bool]$FinalLayerSP
$opts["DumpLogits"] = $false
$opts["RankLocalCollectives"] = $false # keep independently verified HOST Hybrid path
$opts["Trace"] = [bool]$Trace
$opts["OutputDir"] = $runRoot
Write-Host "[GALAXY_CAUSAL_HYBRID] event=START attention_prefix=$AttentionPrefix ffn_sp_layer_set=$resolvedSet final_layer_sp=$([bool]$FinalLayerSP) native_sp_qkv_prefix=$nativeCount resident_weights=$residentExpected skip_tp_prefill=$([bool]$NativeSpQkvSkipTpPrefill)"
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
if ($nativeCount -gt 0) {
    $escapedNative = [regex]::Escape($nativeLayerList)
    $nativeSummary = @(Select-String -LiteralPath $stderr -Pattern "^\[GALAXY_SP_QKV_PREFIX_SUM\] mode=SP_NATIVE layers=$escapedNative completed=$nativeCount expected=$nativeCount all_layers_verified=1 tp_cache_preserved=1 resident_weights=$residentExpected status=PASS$")
    if ($nativeSummary.Count -ne 1) {
        throw "Missing Native SP QKV prefix verification in $stderr"
    }
}
if ($NativeSpQkvSkipTpPrefill) {
    $prefix = [regex]::Escape($nativeLayerList)
    $skipped = 8 * $nativeCount
    $stores = 2 * $nativeCount
    $pattern = "^\[GALAXY_SP_QKV_TP_BYPASS_SUM\] layers=$prefix completed=$nativeCount expected=$nativeCount tp_qkv_nodes_skipped=$skipped cache_kv_store_ops=$stores kv_source=SP_TOKEN_TO_TP_HEAD decode=TP_ORIGINAL status=PASS$"
    $bypass = @(Select-String -LiteralPath $stderr -Pattern $pattern)
    if ($bypass.Count -ne 1) {
        throw "Expected native SP K/V to original TP cache store completion; inspect $stderr"
    }
}
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
    native_sp_qkv_l0 = [bool]$NativeSpQkvL0
    native_sp_qkv_prefix = $nativeCount
    native_sp_qkv_layers = $nativeLayerList
    native_sp_qkv_verified = ($nativeCount -gt 0)
    native_sp_qkv_resident_weights = [bool]$NativeSpQkvResidentWeights
    native_sp_qkv_skip_tp_prefill = [bool]$NativeSpQkvSkipTpPrefill
    kv_cache_writes = "ORIGINAL_LLAMA_TP_SET_ROWS"
    native_kv_cache_layout = if ($NativeSpQkvSkipTpPrefill) { "SP_TO_TP_HEAD" } else { "TP_PROJECTION" }
    qkv_prefill_weight_read_bytes = if ($NativeSpQkvResidentWeights) { 0 } else { $null }
    qkv_prefill_weight_upload_bytes = if ($NativeSpQkvResidentWeights) { 0 } else { $null }
    tp_qkv_kept_for_kv_cache = ($nativeCount -gt 0 -and -not $NativeSpQkvSkipTpPrefill)
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
Write-Host "[GALAXY_CAUSAL_HYBRID] attention_replaced=$AttentionPrefix ffn_sp_layers=$($chosen.Count) final_layer_sp=$expectedFinal native_sp_qkv_prefix=$nativeCount resident_weights=$residentExpected skip_tp_prefill=$([bool]$NativeSpQkvSkipTpPrefill) status=EXECUTION_PASS_NUMERICAL_UNCHECKED logs=$($run[0].FullName)"

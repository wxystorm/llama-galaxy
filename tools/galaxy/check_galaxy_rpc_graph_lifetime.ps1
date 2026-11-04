<# 
  Analyze Galaxy RPC client trace for transient Attention graph UID reuse.
  Pass SHADOW-8's stderr.log after running with -Trace.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$LogPath,
    [switch]$RequireNormalReuse
)

$ErrorActionPreference = "Stop"
$lines = @(Get-Content -LiteralPath $LogPath)
$pattern = '\[RPC_GRAPH\].*uid=(?<uid>[0-9]+).*nodes=(?<nodes>[0-9]+).*tensors=(?<tensors>[0-9]+|reused).*cmd=(?<cmd>GRAPH_COMPUTE|GRAPH_RECOMPUTE)'
$flashUids = @{}
$flashNew = 0
$normalReuses = 0
foreach ($line in $lines) {
    if ($line -notmatch $pattern) { continue }
    $uid = $Matches["uid"]
    $nodes = [int]$Matches["nodes"]
    $tensors = $Matches["tensors"]
    $cmd = $Matches["cmd"]

    if ($nodes -eq 7 -and $tensors -eq "11" -and $cmd -eq "GRAPH_COMPUTE") {
        if ($flashUids.ContainsKey($uid)) {
            throw "Transient native Attention UID repeated after buffer release: $uid"
        }
        $flashUids[$uid] = $true
        ++$flashNew
    }
    if ($cmd -eq "GRAPH_RECOMPUTE") {
        if ($flashUids.ContainsKey($uid)) {
            throw "Stale transient Attention GRAPH_RECOMPUTE detected: $uid"
        }
        ++$normalReuses
    }
}
if ($flashNew -lt 2) {
    throw "Expected multiple transient Attention GRAPH_COMPUTE entries, found $flashNew. Enable -Trace and select SHADOW-8."
}
if ($RequireNormalReuse -and $normalReuses -lt 1) {
    throw "Expected normal TP GRAPH_RECOMPUTE but saw none"
}
Write-Host "[GALAXY_RPC_GRAPH_LIFETIME] transient_full_compute=$flashNew normal_recompute=$normalReuses status=PASS"

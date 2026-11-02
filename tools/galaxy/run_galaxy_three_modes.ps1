<#
Run the same synthetic first-token test as pure TP, Galaxy and Galaxy2.
Profile is invasive (synchronized per rank) and not a speed benchmark.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [string]$Exe = ".\build-galaxy\bin\Release\llama-cli.exe",
    [string]$Comparator = ".\build-galaxy\bin\Release\galaxy-logits-compare.exe",
    [string]$OutputDir = ".\galaxy-sp-results",
    [string]$Devices = "CUDA0,RPC0",
    [string]$HeadSplit = "20,20",
    [string]$FfnSplit = "8704,8704",
    [string]$Galaxy2FfnSpLayers = "1",
    [double]$MaxAbs = 0.1,
    [double]$MaxRmse = 0.02,
    [bool]$Profile = $true,
    [switch]$Trace,
    [int]$TimeoutSec = 300,
    [int]$NPredict = 1,
    [int]$Ctx = 256,
    [int]$Batch = 256,
    [int]$UBatch = 256,
    [int]$Threads = 8
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "galaxy-process.ps1")
if (-not (Test-Path -LiteralPath $Comparator)) {
    throw "Build galaxy-logits-compare first: $Comparator"
}
if ($Galaxy2FfnSpLayers -ne "all" -and $Galaxy2FfnSpLayers -notmatch '^[0-9]+$') {
    throw "Galaxy2FfnSpLayers must be all or a nonnegative number"
}
& $Comparator --selftest
if ($LASTEXITCODE -ne 0) { throw "Comparator selftest failed" }
& (Join-Path $PSScriptRoot "test_galaxy_process.ps1")

$root = Join-Path $OutputDir ("three-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $root -Force | Out-Null
$root = (Resolve-Path -LiteralPath $root).Path
$smoke = Join-Path $PSScriptRoot "run_galaxy_sp_smoke.ps1"
$opts = @{
    Model = $Model
    PromptFile = $PromptFile
    Rpc = $Rpc
    Exe = $Exe
    OutputDir = $root
    Devices = $Devices
    HeadSplit = $HeadSplit
    FfnSplit = $FfnSplit
    DumpLogits = $true
    Profile = $Profile
    Trace = $Trace
    TimeoutSec = $TimeoutSec
    NPredict = $NPredict
    Ctx = $Ctx
    Batch = $Batch
    UBatch = $UBatch
    Threads = $Threads
}
$results = @{}
foreach ($mode in @("tp", "sp", "galaxy2")) {
    Write-Host "[GALAXY_THREE_MODE] phase=$mode"
    if ($mode -eq "tp") {
        & $smoke @opts -PureTP
    } elseif ($mode -eq "sp") {
        & $smoke @opts
    } else {
        & $smoke @opts -Galaxy2 -Galaxy2FfnSpLayers $Galaxy2FfnSpLayers
    }
    $folder = Get-ChildItem -LiteralPath $root -Directory -Filter "$mode-*" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $folder) { throw "Missing $mode result directory" }
    $logits = Join-Path $folder.FullName "first_logits.f32"
    if (-not (Test-Path -LiteralPath $logits)) { throw "Missing $mode full logits" }
    $results[$mode] = @{ dir = $folder.FullName; logits = $logits }
}
$bad = @()
foreach ($mode in @("sp", "galaxy2")) {
    $logPath = Join-Path $root ("compare-" + $mode + ".log")
    $errorPath = Join-Path $root ("compare-" + $mode + ".stderr.log")
    $report = Join-Path $root ("compare-" + $mode + ".json")
    $ci = [System.Globalization.CultureInfo]::InvariantCulture
    $args = @(
        "--tp", ('"' + $results["tp"].logits + '"'),
        "--sp", ('"' + $results[$mode].logits + '"'),
        "--max-abs", $MaxAbs.ToString("G17", $ci),
        "--max-rmse", $MaxRmse.ToString("G17", $ci),
        "--json", ('"' + $report + '"')
    )
    $outcome = Invoke-GalaxyProcess -Executable (Resolve-Path -LiteralPath $Comparator).Path -ArgumentList $args -StdoutPath $logPath -StderrPath $errorPath
    Get-Content -LiteralPath $logPath | ForEach-Object { Write-Host $_ }
    if ($outcome.ExitCode -ne 0) { $bad += $mode }
}
$witness = @{}
foreach ($mode in @("sp", "galaxy2")) {
    $f = Join-Path $results[$mode].dir "stderr.log"
    $lines = @(Select-String -LiteralPath $f -Pattern '\[GALAXY_SP_SUM\]\s+reduced_scatter=(\d+)\s+residual_scatter=(\d+)\s+all_gather=(\d+)' -ErrorAction SilentlyContinue)
    $witness[$mode] = @($lines | ForEach-Object { $_.Line.Trim() })
    foreach ($line in $witness[$mode]) { Write-Host "[GALAXY_THREE_MODE] $mode $line" }
}
if (-not $witness["sp"] -or -not $witness["galaxy2"]) {
    Write-Warning "Missing actual SP collective counts; numeric parity alone does not prove Galaxy2 execution"
}
$manifest = @{
    model = (Resolve-Path -LiteralPath $Model).Path
    prompt = (Resolve-Path -LiteralPath $PromptFile).Path
    prompt_sha256 = (Get-FileHash -LiteralPath $PromptFile -Algorithm SHA256).Hash
    rpc = $Rpc
    heads = $HeadSplit
    ffn = $FfnSplit
    ctx = $Ctx
    batch = $Batch
    ubatch = $UBatch
    threads = $Threads
    n_predict = $NPredict
    collective_witness = $witness
    galaxy2_sp_ffn_layers = $Galaxy2FfnSpLayers
    galaxy2_scope = if ($Galaxy2FfnSpLayers -eq "all") { "full" } else { "partial_hybrid" }
    profile_forced_sync = $Profile
    modes = $results
    numerical_failures = $bad
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $root "manifest.json") -Encoding UTF8
Write-Host "[GALAXY_THREE_MODE] results=$root"
if ($bad.Count -gt 0) {
    throw "Numerical comparison failed for: $($bad -join ', ') (see compare JSONs in $root)"
}
Write-Host "[GALAXY_THREE_MODE] PASS numeric_thresholds=yes (check SP collective counts separately)"

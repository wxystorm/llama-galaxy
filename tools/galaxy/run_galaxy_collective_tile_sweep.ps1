<#
A/B: original 32-token vs optimized 256-token Host Staging tiles.
Runs Galaxy and layer-capped Galaxy2, saves per-mode Prefill and full
first-token logits. No invasive -Profile by default.
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
    [string]$SeqWeights = "1,1",
    [switch]$FinalLayerSP,
    [int]$BaselineTile = 32,
    [int]$OptimizedTile = 256,
    [switch]$Profile,
    [int]$TimeoutSec = 300,
    [int]$NPredict = 1,
    [int]$Ctx = 256,
    [int]$Batch = 256,
    [int]$UBatch = 256,
    [int]$Threads = 8
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "galaxy-process.ps1")
if ($BaselineTile -lt 1 -or $BaselineTile -gt 1024 -or
        $OptimizedTile -lt 1 -or $OptimizedTile -gt 1024) {
    throw "Both collective tile sizes must be 1..1024"
}
if ($BaselineTile -eq $OptimizedTile) { throw "Tile choices must differ" }
if (-not (Test-Path -LiteralPath $Comparator)) { throw "Build galaxy-logits-compare first" }
if (-not (Test-Path -LiteralPath $Exe)) { throw "Missing llama-cli: $Exe" }

$root = Join-Path $OutputDir ("tile-sweep-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Path $root -Force | Out-Null
$root = (Resolve-Path -LiteralPath $root).Path
$smoke = Join-Path $PSScriptRoot "run_galaxy_sp_smoke.ps1"
$opts = @{
    Model = $Model
    PromptFile = $PromptFile
    Rpc = $Rpc
    Exe = $Exe
    Devices = $Devices
    HeadSplit = $HeadSplit
    FfnSplit = $FfnSplit
    Galaxy2FfnSpLayers = $Galaxy2FfnSpLayers
    SeqWeights = $SeqWeights
    FinalLayerSP = $FinalLayerSP
    OutputDir = $root
    Profile = $Profile
    DumpLogits = $true
    TimeoutSec = $TimeoutSec
    NPredict = $NPredict
    Ctx = $Ctx
    Batch = $Batch
    UBatch = $UBatch
    Threads = $Threads
}
$records = @()
$folders = @{}
$ci = [System.Globalization.CultureInfo]::InvariantCulture

foreach ($mode in @("sp", "galaxy2")) {
    foreach ($tile in @($BaselineTile, $OptimizedTile)) {
        Write-Host "[GALAXY_TILE_SWEEP] mode=$mode tile_tokens=$tile profiling=$([bool]$Profile)"
        if ($mode -eq "galaxy2") {
            & $smoke @opts -Galaxy2 -CollectiveTileTokens $tile
        } else {
            & $smoke @opts -CollectiveTileTokens $tile
        }
        if (-not $?) { throw "Smoke failed: $mode/$tile" }
        $folder = Get-ChildItem -LiteralPath $root -Directory -Filter "$mode-*" |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $folder) { throw "Missing $mode result directory" }
        $folders["$mode/$tile"] = $folder.FullName
        $logPaths = @((Join-Path $folder.FullName "stdout.log"),(Join-Path $folder.FullName "stderr.log"))
        $lines = @(Select-String -LiteralPath $logPaths -Pattern 'prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens' -ErrorAction SilentlyContinue)
        if ($lines.Count -eq 0) { throw "Missing timing in $($folder.FullName)" }
        $match = [regex]::Match($lines[-1].Line,
            'prompt eval time\s*=\s*(?<ms>[0-9.]+)\s*ms\s*/\s*(?<tokens>[0-9]+)\s*tokens')
        if (-not $match.Success) { throw "Unparseable timing: $($lines[-1].Line)" }
        $ms = [double]::Parse($match.Groups["ms"].Value, $ci)
        $tokens = [int]::Parse($match.Groups["tokens"].Value, $ci)
        $records += [pscustomobject]@{
            mode = $mode
            tile_tokens = $tile
            prefill_tokens = $tokens
            prefill_ms = $ms
            prefill_tps = if ($ms -gt 0) { [math]::Round($tokens*1000/$ms, 4) } else { 0 }
            directory = $folder.FullName
        }
    }
}
$csv = Join-Path $root "timing.csv"
$records | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
Write-Host "[GALAXY_TILE_SWEEP] timing=$csv"

$parity = @{}
foreach ($mode in @("sp", "galaxy2")) {
    $oldDir = $folders["$mode/$BaselineTile"]
    $newDir = $folders["$mode/$OptimizedTile"]
    $baseline = Join-Path $oldDir "first_logits.f32"
    $optimized = Join-Path $newDir "first_logits.f32"
    $json = Join-Path $root ("tile-parity-" + $mode + ".json")
    $args = @(
        "--tp", ('"' + $baseline + '"'),
        "--sp", ('"' + $optimized + '"'),
        "--max-abs", "0.1", "--max-rmse", "0.02",
        "--json", ('"' + $json + '"')
    )
    $log = Join-Path $root ("tile-parity-" + $mode + ".log")
    $err = Join-Path $root ("tile-parity-" + $mode + ".stderr.log")
    $result = Invoke-GalaxyProcess -Executable (Resolve-Path -LiteralPath $Comparator).Path -ArgumentList $args -StdoutPath $log -StderrPath $err
    Get-Content -LiteralPath $log | ForEach-Object { Write-Host $_ }
    $parity[$mode] = @{ exit_code = $result.ExitCode; json = $json }
    $old = @($records | Where-Object { $_.mode -eq $mode -and $_.tile_tokens -eq $BaselineTile })[0]
    $new = @($records | Where-Object { $_.mode -eq $mode -and $_.tile_tokens -eq $OptimizedTile })[0]
    if ($old.prefill_tokens -ne $new.prefill_tokens) { throw "Mismatched token counts in $mode" }
    $speedup = if ($new.prefill_ms -gt 0) { $old.prefill_ms/$new.prefill_ms } else { 0 }
    Write-Host ("[GALAXY_TILE_SWEEP_SUM] mode={0} old_ms={1:N2} new_ms={2:N2} speedup={3:N3}x" -f
        $mode, $old.prefill_ms, $new.prefill_ms, $speedup)
}
$manifest = @{
    model = (Resolve-Path -LiteralPath $Model).Path
    prompt = (Resolve-Path -LiteralPath $PromptFile).Path
    prompt_sha256 = (Get-FileHash -LiteralPath $PromptFile -Algorithm SHA256).Hash
    rpc = $Rpc
    baseline_tile = $BaselineTile
    optimized_tile = $OptimizedTile
    galaxy2_sp_ffn_layers = $Galaxy2FfnSpLayers
    seq_weights = $SeqWeights
    final_layer_sp = [bool]$FinalLayerSP
    profiling_forced_sync = [bool]$Profile
    results = $records
    tile_parity = $parity
    note = "Single-run A/B observations; known TP/SP baseline numerical differences are separate"
}
$manifest | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $root "manifest.json") -Encoding UTF8
Write-Host "[GALAXY_TILE_SWEEP] results=$root"
$failed = @($parity.Values | Where-Object { $_.exit_code -ne 0 })
if ($failed.Count -gt 0) { throw "Optimized tiles changed the numerical result; inspect $root" }
Write-Host "[GALAXY_TILE_SWEEP] PASS tile_numeric_consistency=yes"

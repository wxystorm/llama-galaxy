<#
Galaxy + Galaxy2 A/B experiment for optional fused ReduceScatter source reads.
Same model, prompt, devices and 256-token staging tile for both strategies.
Without -Profile, use Prefill timing for performance. With -Profile, inspect
communication_timeline.csv for read_calls and rs_fused_read per RS operation.
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
    [int]$CollectiveTileTokens = 256,
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
if ($CollectiveTileTokens -lt 1 -or $CollectiveTileTokens -gt 1024) {
    throw "CollectiveTileTokens must be 1..1024"
}
if (-not (Test-Path -LiteralPath $Exe)) { throw "Missing llama-cli: $Exe" }
if (-not (Test-Path -LiteralPath $Comparator)) { throw "Missing comparator: $Comparator" }

$root = Join-Path $OutputDir ("rs-fused-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
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
    OutputDir = $root
    CollectiveTileTokens = $CollectiveTileTokens
    Profile = $Profile
    DumpLogits = $true
    TimeoutSec = $TimeoutSec
    NPredict = $NPredict
    Ctx = $Ctx
    Batch = $Batch
    UBatch = $UBatch
    Threads = $Threads
}
$ci = [System.Globalization.CultureInfo]::InvariantCulture
$records = @()
$folders = @{}
foreach ($mode in @("sp", "galaxy2")) {
    foreach ($enabled in @($false, $true)) {
        $state = if ($enabled) { "on" } else { "off" }
        Write-Host "[GALAXY_RS_FUSED_SWEEP] mode=$mode fused=$state tile=$CollectiveTileTokens profile=$([bool]$Profile)"
        if ($mode -eq "galaxy2") {
            & $smoke @opts -Galaxy2 -Galaxy2FfnSpLayers $Galaxy2FfnSpLayers -RsFusedRead:$enabled
        } else {
            & $smoke @opts -RsFusedRead:$enabled
        }
        if (-not $?) { throw "Smoke failed for $mode/$state" }
        $folder = Get-ChildItem -LiteralPath $root -Directory -Filter "$mode-*" |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $folder) { throw "No results folder for $mode/$state" }
        $key = "$mode/$state"
        $folders[$key] = $folder.FullName

        $logs = @((Join-Path $folder.FullName "stderr.log"),(Join-Path $folder.FullName "stdout.log"))
        $matches = @(Select-String -LiteralPath $logs -Pattern 'prompt eval time\s*=\s*[0-9.]+\s*ms\s*/\s*[0-9]+\s*tokens' -ErrorAction SilentlyContinue)
        if ($matches.Count -eq 0) { throw "No Prefill timing for $key" }
        $match = [regex]::Match($matches[-1].Line, 'prompt eval time\s*=\s*(?<ms>[0-9.]+)\s*ms\s*/\s*(?<tokens>[0-9]+)\s*tokens')
        if (-not $match.Success) { throw "Unparseable Prefill timing for $key" }
        $ms = [double]::Parse($match.Groups["ms"].Value, $ci)
        $tokens = [int]::Parse($match.Groups["tokens"].Value, $ci)
        $records += [pscustomobject]@{
            mode = $mode
            fused_read = $state
            tile_tokens = $CollectiveTileTokens
            prefill_tokens = $tokens
            prefill_ms = $ms
            prefill_tps = if ($ms -gt 0) { [math]::Round($tokens * 1000.0 / $ms, 5) } else { 0 }
            directory = $folder.FullName
        }
    }
}
$timing = Join-Path $root "timing.csv"
$records | Export-Csv -LiteralPath $timing -NoTypeInformation -Encoding UTF8
Write-Host "[GALAXY_RS_FUSED_SWEEP] timing=$timing"

$comparison = @{}
foreach ($mode in @("sp", "galaxy2")) {
    $old = Join-Path $folders["$mode/off"] "first_logits.f32"
    $new = Join-Path $folders["$mode/on"] "first_logits.f32"
    $json = Join-Path $root ("parity-" + $mode + ".json")
    $log = Join-Path $root ("parity-" + $mode + ".stdout.log")
    $err = Join-Path $root ("parity-" + $mode + ".stderr.log")
    $args = @(
        "--tp", ('"' + $old + '"'), "--sp", ('"' + $new + '"'),
        "--max-abs", "0", "--max-rmse", "0",
        "--json", ('"' + $json + '"')
    )
    $result = Invoke-GalaxyProcess -Executable (Resolve-Path -LiteralPath $Comparator).Path -ArgumentList $args -StdoutPath $log -StderrPath $err
    Get-Content -LiteralPath $log | ForEach-Object { Write-Host $_ }
    $comparison[$mode] = @{ exit_code = $result.ExitCode; report = $json }

    $oldTiming = @($records | Where-Object { $_.mode -eq $mode -and $_.fused_read -eq "off" })[0]
    $newTiming = @($records | Where-Object { $_.mode -eq $mode -and $_.fused_read -eq "on" })[0]
    if ($oldTiming.prefill_tokens -ne $newTiming.prefill_tokens) {
        throw "Token counts changed for mode=$mode"
    }
    $ratio = $oldTiming.prefill_ms / $newTiming.prefill_ms
    Write-Host ("[GALAXY_RS_FUSED_SUM] mode={0} off_ms={1:N2} on_ms={2:N2} speedup={3:N3}x parity_exit={4}" -f
        $mode, $oldTiming.prefill_ms, $newTiming.prefill_ms, $ratio, $result.ExitCode)
}
$profileSummary = @()
if ($Profile) {
    foreach ($mode in @("sp", "galaxy2")) {
        foreach ($state in @("off", "on")) {
            $profileCsv = Join-Path $folders["$mode/$state"] "profile\communication_timeline.csv"
            if (-not (Test-Path -LiteralPath $profileCsv)) {
                throw "Profile was requested but communication CSV is missing: $profileCsv"
            }
            $rsRows = @(Import-Csv -LiteralPath $profileCsv | Where-Object { $_.kind -eq "RS" })
            if ($rsRows.Count -eq 0) { throw "No RS records in $profileCsv" }
            $reads = 0
            $writes = 0
            $usedFused = 0
            foreach ($item in $rsRows) {
                $reads += [int]$item.read_calls
                $writes += [int]$item.write_calls
                if ($item.rs_fused_read -eq "1") { $usedFused++ }
            }
            $profileSummary += [pscustomobject]@{
                mode = $mode
                fused_requested = $state
                rs_operations = $rsRows.Count
                rs_fused_used = $usedFused
                rs_backend_read_calls = $reads
                rs_backend_write_calls = $writes
            }
            Write-Host "[GALAXY_RS_FUSED_CALLS] mode=$mode state=$state rs=$($rsRows.Count) fused_used=$usedFused reads=$reads writes=$writes"
        }
    }
    $profileSummary | Export-Csv -LiteralPath (Join-Path $root "rpc_call_counts.csv") -NoTypeInformation -Encoding UTF8
}
$manifest = @{
    model = (Resolve-Path -LiteralPath $Model).Path
    prompt = (Resolve-Path -LiteralPath $PromptFile).Path
    prompt_sha256 = (Get-FileHash -LiteralPath $PromptFile -Algorithm SHA256).Hash
    rpc = $Rpc
    devices = $Devices
    head_split = $HeadSplit
    ffn_split = $FfnSplit
    galaxy2_sp_ffn_layers = $Galaxy2FfnSpLayers
    collective_tile_tokens = $CollectiveTileTokens
    profile_forced_sync = [bool]$Profile
    rpc_call_count_summary = $profileSummary
    profiles = $records
    fused_vs_tiled_parity = $comparison
    note = "Compare fused and unfused within the SAME execution mode; TP/SP differences are separate"
}
$manifest | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $root "manifest.json") -Encoding UTF8
Write-Host "[GALAXY_RS_FUSED_SWEEP] results=$root"
$failures = @($comparison.Values | Where-Object { $_.exit_code -ne 0 })
if ($failures.Count -gt 0) {
    throw "Fused ReduceScatter changed full logits; inspect the reports in $root"
}
Write-Host "[GALAXY_RS_FUSED_SWEEP] PASS fused_vs_tiled_bitwise=yes"

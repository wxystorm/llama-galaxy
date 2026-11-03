<#
Galaxy PC + Phone TP/SP smoke on Windows PowerShell.
Uses System.Diagnostics.Process with separate streaming log files.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [string]$Exe = ".\build-galaxy\bin\Release\llama-cli.exe",
    [string]$Devices = "CUDA0,RPC0",
    [string]$HeadSplit = "20,20",
    [string]$FfnSplit = "8704,8704",
    [string]$OutputDir = ".\galaxy-sp-results",
    [switch]$PureTP,
    [switch]$Galaxy2,
    [string]$Galaxy2FfnSpLayers = "1",
    [string]$Galaxy2FfnSpLayerSet = "",
    [int]$CollectiveTileTokens = 32,
    [string]$SeqWeights = "1,1",
    [switch]$FinalLayerSP,
    [switch]$RsFusedRead,
    [switch]$Profile,
    [string]$DumpLayers = "",
    [string]$DumpStages = "",
    [string]$DumpFfnOps = "",
    [string]$ProbeQkvLayers = "",
    [switch]$Trace,
    [switch]$DumpLogits,
    [int]$TimeoutSec = 0,
    [int]$NPredict = 1,
    [int]$Ctx = 256,
    [int]$Batch = 256,
    [int]$UBatch = 256,
    [int]$Threads = 8
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "galaxy-process.ps1")
if (-not (Test-Path -LiteralPath $Exe)) { throw "Missing executable: $Exe" }
if (-not (Test-Path -LiteralPath $Model)) { throw "Missing model: $Model" }
if (-not (Test-Path -LiteralPath $PromptFile)) { throw "Missing prompt: $PromptFile" }
if ($Devices.Split(",").Count -ne 2) { throw "Devices must contain exactly PC,RPC" }
if ($PureTP -and $Galaxy2) { throw "PureTP and Galaxy2 cannot both be enabled" }
if ($Galaxy2FfnSpLayerSet -and -not $Galaxy2) {
    throw "Galaxy2FfnSpLayerSet requires -Galaxy2"
}
if ($Galaxy2FfnSpLayerSet -and $PSBoundParameters.ContainsKey("Galaxy2FfnSpLayers")) {
    throw "Specify Galaxy2FfnSpLayerSet OR Galaxy2FfnSpLayers, not both"
}
if ($Galaxy2FfnSpLayerSet -and $Galaxy2FfnSpLayerSet -ne "all" -and
        $Galaxy2FfnSpLayerSet -ne "none" -and
        $Galaxy2FfnSpLayerSet -notmatch '^[0-9,-]+$') {
    throw "Galaxy2FfnSpLayerSet must use zero-based indices/ranges, all, or none"
}
if ($PureTP -and $FinalLayerSP) { throw "FinalLayerSP requires Galaxy or Galaxy2" }
if ($SeqWeights -notmatch '^[1-9][0-9]*,[1-9][0-9]*$') {
    throw "SeqWeights must be two positive integers such as 20,12"
}
if ($CollectiveTileTokens -lt 1 -or $CollectiveTileTokens -gt 1024) {
    throw "CollectiveTileTokens must be 1..1024"
}
if ($Galaxy2FfnSpLayers -ne "all" -and $Galaxy2FfnSpLayers -notmatch '^[0-9]+$') {
    throw "Galaxy2FfnSpLayers must be 'all' or a nonnegative integer"
}
if ($Ctx -le 0 -or $Batch -le 0 -or $UBatch -le 0 -or $Threads -le 0 -or $NPredict -le 0) {
    throw "Context, batch, ubatch, threads and predict must be positive"
}

$keys = @("LLAMA_GALAXY_TP", "LLAMA_GALAXY_TP_HEADS", "LLAMA_GALAXY_TP_FFN", "LLAMA_GALAXY_SP", "LLAMA_GALAXY_SP_TRACE", "LLAMA_GALAXY_FIRST_LOGITS_PATH",
    "LLAMA_GALAXY2", "LLAMA_GALAXY2_FFN_SP_LAYERS", "LLAMA_GALAXY2_FFN_SP_LAYER_SET", "LLAMA_GALAXY_PROFILE_DIR",
    "LLAMA_GALAXY_LAYER_DUMP", "LLAMA_GALAXY_LAYER_DUMP_DIR",
    "LLAMA_GALAXY_STAGE_DUMP", "LLAMA_GALAXY_STAGE_DUMP_DIR",
    "LLAMA_GALAXY_FFN_OP_DUMP", "LLAMA_GALAXY_FFN_OP_DUMP_DIR",
    "LLAMA_GALAXY_QKV_PROBE_LAYERS", "LLAMA_GALAXY_QKV_PROBE_DIR",
    "LLAMA_GALAXY_COLLECTIVE_TILE_TOKENS", "LLAMA_GALAXY_RS_FUSED_READ", "LLAMA_GALAXY_SP_SEQ_WEIGHTS",
    "LLAMA_GALAXY_SP_FINAL_LAYER")
$old = @{}
foreach ($key in $keys) { $old[$key] = [Environment]::GetEnvironmentVariable($key, "Process") }

try {
    $env:LLAMA_GALAXY_TP       = "1"
    $env:LLAMA_GALAXY_TP_HEADS = $HeadSplit
    $env:LLAMA_GALAXY_TP_FFN   = $FfnSplit
    $env:LLAMA_GALAXY_SP       = if ($PureTP) { "0" } else { "1" }
    $env:LLAMA_GALAXY_SP_TRACE = if ($Trace) { "1" } else { "0" }
    $env:LLAMA_GALAXY_SP_SEQ_WEIGHTS = $SeqWeights
    $env:LLAMA_GALAXY_SP_FINAL_LAYER = if ($FinalLayerSP) { "1" } else { "0" }
    $env:LLAMA_GALAXY_COLLECTIVE_TILE_TOKENS = $CollectiveTileTokens.ToString()
    $env:LLAMA_GALAXY_RS_FUSED_READ = if ($RsFusedRead) { "1" } else { "0" }
    $env:LLAMA_GALAXY2 = if ($Galaxy2) { "1" } else { "0" }
    $env:LLAMA_GALAXY2_FFN_SP_LAYERS = if ($Galaxy2 -and -not $Galaxy2FfnSpLayerSet) { $Galaxy2FfnSpLayers } else { $null }
    $env:LLAMA_GALAXY2_FFN_SP_LAYER_SET = if ($Galaxy2 -and $Galaxy2FfnSpLayerSet) { $Galaxy2FfnSpLayerSet } else { $null }

    $mode = if ($PureTP) { "tp" } elseif ($Galaxy2) { "galaxy2" } else { "sp" }
    $dir = Join-Path $OutputDir "$mode-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    # Every artifact path should be absolute. Powershell's current location
    # and the .NET process working directory are not necessarily the same.
    $dir = (Resolve-Path -LiteralPath $dir).Path
    $env:LLAMA_GALAXY_PROFILE_DIR = if ($Profile) {
        $profileDir = Join-Path (Resolve-Path -LiteralPath $dir).Path "profile"
        New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
        $profileDir
    } else { $null }
    $env:LLAMA_GALAXY_LAYER_DUMP = if ($DumpLayers) { $DumpLayers } else { $null }
    $env:LLAMA_GALAXY_LAYER_DUMP_DIR = if ($DumpLayers) {
        $layerDir = Join-Path (Resolve-Path -LiteralPath $dir).Path "layers"
        New-Item -ItemType Directory -Force -Path $layerDir | Out-Null
        $layerDir
    } else { $null }
    $stdout = Join-Path $dir "stdout.log"
    $stderr = Join-Path $dir "stderr.log"
    $dumpPath = Join-Path (Resolve-Path -LiteralPath $dir).Path "first_logits.f32"
    $env:LLAMA_GALAXY_STAGE_DUMP = if ($DumpStages) { $DumpStages } else { $null }
    $env:LLAMA_GALAXY_STAGE_DUMP_DIR = if ($DumpStages) {
        $stageDir = Join-Path (Resolve-Path -LiteralPath $dir).Path "stages"
        New-Item -ItemType Directory -Force -Path $stageDir | Out-Null
        $stageDir
    } else { $null }
    $env:LLAMA_GALAXY_FFN_OP_DUMP = if ($DumpFfnOps) { $DumpFfnOps } else { $null }
    $env:LLAMA_GALAXY_FFN_OP_DUMP_DIR = if ($DumpFfnOps) {
        $ffnOpDir = Join-Path (Resolve-Path -LiteralPath $dir).Path "ffn_ops"
        New-Item -ItemType Directory -Force -Path $ffnOpDir | Out-Null
        $ffnOpDir
    } else { $null }
    $env:LLAMA_GALAXY_QKV_PROBE_LAYERS = if ($ProbeQkvLayers) { $ProbeQkvLayers } else { $null }
    $env:LLAMA_GALAXY_QKV_PROBE_DIR = if ($ProbeQkvLayers) {
        $probeDir = Join-Path $dir "real-qkv"
        New-Item -ItemType Directory -Force -Path $probeDir | Out-Null
        (Resolve-Path -LiteralPath $probeDir).Path
    } else { $null }
    $env:LLAMA_GALAXY_FIRST_LOGITS_PATH = if ($DumpLogits) { $dumpPath } else { $null }

    $modelArg = '"' + (Resolve-Path -LiteralPath $Model).Path + '"'
    $promptArg = '"' + (Resolve-Path -LiteralPath $PromptFile).Path + '"'
    $logVerbosity = if ($Trace) { "4" } else { "3" }
    $arguments = @(
        "--single-turn", "-lv", $logVerbosity,
        "-m", $modelArg, "--rpc", $Rpc, "-dev", $Devices, "-sm", "tensor",
        "-ts", "1,1", "-ngl", "all", "-fa", "on", "-ctk", "f16", "-ctv", "f16",
        "-fit", "off", "-c", $Ctx.ToString(), "-b", $Batch.ToString(),
        "-ub", $UBatch.ToString(), "-t", $Threads.ToString(),
        "-n", $NPredict.ToString(), "--temp", "0", "-f", $promptArg
    )
    Write-Host "[GALAXY_SP_SMOKE] mode=$mode devices=$Devices rpc=$Rpc profiling=$([bool]$Profile) tile_tokens=$CollectiveTileTokens rs_fused=$([bool]$RsFusedRead) seq_weights=$SeqWeights final_layer_sp=$([bool]$FinalLayerSP)"
    if ($Galaxy2FfnSpLayerSet) {
        Write-Host "[GALAXY2] mode=explicit_layer_set schedule=$Galaxy2FfnSpLayerSet final_layer_sp=$([bool]$FinalLayerSP)"
    } elseif ($Galaxy2 -and $Galaxy2FfnSpLayers -ne "all") {
        Write-Host "[GALAXY2] hybrid_layer_cap=$Galaxy2FfnSpLayers (not full Galaxy2)"
    }
    $result = Invoke-GalaxyProcess `
        -Executable (Resolve-Path -LiteralPath $Exe).Path `
        -ArgumentList $arguments -StdoutPath $stdout -StderrPath $stderr `
        -TimeoutSec $TimeoutSec
    if ($result.TimedOut) {
        Write-Host "[GALAXY_SP_SMOKE] TIMEOUT mode=$mode logs=$dir"
        @(Select-String -Path $stderr, $stdout `
            -Pattern "GALAXY_META_TRACE","GALAXY_SP_SUM","GALAXY_SP_LAYOUT_FAIL","GALAXY_SP]","RPC_GRAPH" `
            -ErrorAction SilentlyContinue | Select-Object -Last 30) |
            ForEach-Object { Write-Host $_.Line.Trim() }
        throw "Galaxy $mode run timed out after $TimeoutSec seconds; logs=$dir"
    }
    $exitCode = $result.ExitCode
    Write-Host "[GALAXY_SP_SMOKE] exit=$exitCode logs=$dir"

    $hits = @(Select-String -Path $stderr, $stdout -Pattern "GALAXY_META_TRACE","GALAXY_SP_LAYOUT_FAIL","GALAXY_SP_LAYOUT_SRC","GALAXY_SP_SUM","GALAXY_SP]","GALAXY_TP]","prompt eval time","eval time","ERROR","failed" -ErrorAction SilentlyContinue)
    # Keep the full trace in stderr.log, but print only the last 80
    # matching lines on the terminal to avoid flooding a long SP run.
    $displayHits = if ($Trace) { @($hits | Select-Object -Last 80) } else { $hits }
    foreach ($hit in $displayHits) { Write-Host $hit.Line.Trim() }

    if ($exitCode -ne 0) { throw "Smoke run failed (exit=$exitCode); examine $stderr and $stdout" }
    if ($DumpLogits -and ((-not (Test-Path -LiteralPath $dumpPath)) -or
            (Get-Item -LiteralPath $dumpPath).Length -le 8)) {
        throw "Missing or incomplete first-token full logits: $dumpPath"
    }
    $spWitness = [bool](@($hits | Where-Object {
        $_.Line -match '\[GALAXY_SP_SUM\]\s+reduced_scatter=[1-9][0-9]*'
    }).Count -gt 0)
    if (-not $PureTP -and -not $spWitness) {
        Write-Warning "SP run exited normally but no positive ReduceScatter count was logged. Continue numerical comparison; actual SP collective execution remains UNVERIFIED."
    }
    if ($Profile) { Write-Host "[GALAXY_PROFILE] directory=$($env:LLAMA_GALAXY_PROFILE_DIR)" }
    if ($DumpLayers) { Write-Host "[GALAXY_LAYER_DUMP] layers=$DumpLayers dir=$($env:LLAMA_GALAXY_LAYER_DUMP_DIR)" }
    if ($DumpStages) {
        $stageNames = @("ffn_inp", "ffn_norm")
        foreach ($layer in $DumpStages.Split(",")) {
            foreach ($stage in $stageNames) {
                $snapshot = Join-Path $env:LLAMA_GALAXY_STAGE_DUMP_DIR "$stage-$layer.f32"
                if ($DumpStages -ne "all" -and (-not (Test-Path -LiteralPath $snapshot) -or
                        (Get-Item -LiteralPath $snapshot).Length -le 8)) {
                    throw "Missing Galaxy diagnostic stage snapshot: $snapshot"
                }
            }
        }
        Write-Host "[GALAXY_STAGE_DUMP] layers=$DumpStages dir=$($env:LLAMA_GALAXY_STAGE_DUMP_DIR)"
    }
    if ($DumpFfnOps) {
        if ($DumpFfnOps -ne "all") {
            foreach ($layer in $DumpFfnOps.Split(",")) {
                foreach ($stage in @("ffn_up", "ffn_gate", "ffn_swiglu")) {
                    $snapshot = Join-Path $env:LLAMA_GALAXY_FFN_OP_DUMP_DIR "$stage-$layer.f32"
                    if (-not (Test-Path -LiteralPath $snapshot) -or
                            (Get-Item -LiteralPath $snapshot).Length -le 8) {
                        throw "Missing Galaxy FFN operator snapshot: $snapshot"
                    }
                }
            }
        }
        Write-Host "[GALAXY_FFN_OP_DUMP] layers=$DumpFfnOps dir=$($env:LLAMA_GALAXY_FFN_OP_DUMP_DIR)"
    }
    if ($ProbeQkvLayers) {
        $probeRoot = $env:LLAMA_GALAXY_QKV_PROBE_DIR
        $tokenRuns = @(Get-ChildItem -LiteralPath $probeRoot -Directory -Filter "tokens-*" |
            Where-Object { $_.Name -match '^tokens-([0-9]+)$' } |
            Sort-Object @{Expression = { [int64]$_.Name.Substring(7) }; Descending = $true})
        if ($tokenRuns.Count -eq 0) {
            throw "No multi-token QKV snapshots found: $probeRoot"
        }
        $latest = $tokenRuns[0].FullName
        if ($ProbeQkvLayers -ne "all") {
            foreach ($layer in $ProbeQkvLayers.Split(",")) {
                foreach ($stage in @("Qcur", "Kcur", "Vcur", "kqv_out")) {
                    $snap = Join-Path $latest "$stage-$layer.f32"
                    if (-not (Test-Path -LiteralPath $snap) -or
                            (Get-Item -LiteralPath $snap).Length -le 8) {
                        throw "Missing real-model QKV probe snapshot: $snap"
                    }
                }
            }
        }
        Write-Host "[GALAXY_REAL_QKV] layers=$ProbeQkvLayers largest_tokens=$($tokenRuns[0].Name) dir=$latest"
    }
    $spEvidence = if ($PureTP) { "N/A" } elseif ($spWitness) { "CONFIRMED" } else { "UNVERIFIED" }
    Write-Host "[GALAXY_SP_SMOKE] PASS mode=$mode inference=PASS sp_collectives=$spEvidence"
}
finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $old[$key], "Process") }
}

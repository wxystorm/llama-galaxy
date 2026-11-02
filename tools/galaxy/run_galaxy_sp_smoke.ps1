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
if ($Ctx -le 0 -or $Batch -le 0 -or $UBatch -le 0 -or $Threads -le 0 -or $NPredict -le 0) {
    throw "Context, batch, ubatch, threads and predict must be positive"
}

$keys = @("LLAMA_GALAXY_TP", "LLAMA_GALAXY_TP_HEADS", "LLAMA_GALAXY_TP_FFN", "LLAMA_GALAXY_SP", "LLAMA_GALAXY_SP_TRACE", "LLAMA_GALAXY_FIRST_LOGITS_PATH")
$old = @{}
foreach ($key in $keys) { $old[$key] = [Environment]::GetEnvironmentVariable($key, "Process") }

try {
    $env:LLAMA_GALAXY_TP       = "1"
    $env:LLAMA_GALAXY_TP_HEADS = $HeadSplit
    $env:LLAMA_GALAXY_TP_FFN   = $FfnSplit
    $env:LLAMA_GALAXY_SP       = if ($PureTP) { "0" } else { "1" }
    $env:LLAMA_GALAXY_SP_TRACE = if ($Trace) { "1" } else { "0" }

    $mode = if ($PureTP) { "tp" } else { "sp" }
    $dir = Join-Path $OutputDir "$mode-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $stdout = Join-Path $dir "stdout.log"
    $stderr = Join-Path $dir "stderr.log"
    $dumpPath = Join-Path (Resolve-Path -LiteralPath $dir).Path "first_logits.f32"
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
    Write-Host "[GALAXY_SP_SMOKE] mode=$mode devices=$Devices rpc=$Rpc"
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
    $spEvidence = if ($PureTP) { "N/A" } elseif ($spWitness) { "CONFIRMED" } else { "UNVERIFIED" }
    Write-Host "[GALAXY_SP_SMOKE] PASS mode=$mode inference=PASS sp_collectives=$spEvidence"
}
finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $old[$key], "Process") }
}

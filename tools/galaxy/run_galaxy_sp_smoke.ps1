<#
Galaxy PC + Phone TP/SP smoke on Windows PowerShell.
Uses Start-Process to capture stdout/stderr without NativeCommandError.
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
    [int]$NPredict = 1,
    [int]$Ctx = 256,
    [int]$Batch = 256,
    [int]$UBatch = 256,
    [int]$Threads = 8
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Exe)) { throw "Missing executable: $Exe" }
if (-not (Test-Path -LiteralPath $Model)) { throw "Missing model: $Model" }
if (-not (Test-Path -LiteralPath $PromptFile)) { throw "Missing prompt: $PromptFile" }
if ($Devices.Split(",").Count -ne 2) { throw "Devices must contain exactly PC,RPC" }
if ($Ctx -le 0 -or $Batch -le 0 -or $UBatch -le 0 -or $Threads -le 0 -or $NPredict -le 0) {
    throw "Context, batch, ubatch, threads and predict must be positive"
}

$keys = @("LLAMA_GALAXY_TP", "LLAMA_GALAXY_TP_HEADS", "LLAMA_GALAXY_TP_FFN", "LLAMA_GALAXY_SP")
$old = @{}
foreach ($key in $keys) { $old[$key] = [Environment]::GetEnvironmentVariable($key, "Process") }

try {
    $env:LLAMA_GALAXY_TP       = "1"
    $env:LLAMA_GALAXY_TP_HEADS = $HeadSplit
    $env:LLAMA_GALAXY_TP_FFN   = $FfnSplit
    $env:LLAMA_GALAXY_SP       = if ($PureTP) { "0" } else { "1" }

    $mode = if ($PureTP) { "tp" } else { "sp" }
    $dir = Join-Path $OutputDir "$mode-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $stdout = Join-Path $dir "stdout.log"
    $stderr = Join-Path $dir "stderr.log"

    $modelArg = '"' + (Resolve-Path -LiteralPath $Model).Path + '"'
    $promptArg = '"' + (Resolve-Path -LiteralPath $PromptFile).Path + '"'
    $arguments = @(
        "-m", $modelArg, "--rpc", $Rpc, "-dev", $Devices, "-sm", "tensor",
        "-ts", "1,1", "-ngl", "all", "-fa", "on", "-ctk", "f16", "-ctv", "f16",
        "-fit", "off", "-c", $Ctx.ToString(), "-b", $Batch.ToString(),
        "-ub", $UBatch.ToString(), "-t", $Threads.ToString(),
        "-n", $NPredict.ToString(), "--temp", "0", "-f", $promptArg
    )
    Write-Host "[GALAXY_SP_SMOKE] mode=$mode devices=$Devices rpc=$Rpc"
    $proc = Start-Process -FilePath (Resolve-Path -LiteralPath $Exe).Path -ArgumentList $arguments -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-Host "[GALAXY_SP_SMOKE] exit=$($proc.ExitCode) logs=$dir"

    $hits = @(Select-String -Path $stderr, $stdout -Pattern "GALAXY_SP_SUM","GALAXY_SP]","GALAXY_TP]","prompt eval time","eval time","ERROR","failed" -ErrorAction SilentlyContinue)
    foreach ($hit in $hits) { Write-Host $hit.Line.Trim() }

    if ($proc.ExitCode -ne 0) { throw "Smoke run failed; examine $stderr and $stdout" }
    if (-not $PureTP -and -not ($hits | Where-Object { $_.Line -match 'GALAXY_SP_SUM' })) {
        throw "No GALAXY_SP_SUM found: verify multi-token prompt and Meta execution"
    }
    Write-Host "[GALAXY_SP_SMOKE] PASS mode=$mode"
}
finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $old[$key], "Process") }
}

<#
Run pure TP then TP/SP on the same prompt with the same model/options.
Dump and numerically compare the complete first-token FP32 logits.
Correctness experiment only; not a clean performance benchmark.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [string]$Exe = ".\build-galaxy\bin\Release\llama-cli.exe",
    [string]$Comparator = ".\build-galaxy\bin\Release\galaxy-logits-compare.exe",
    [string]$Devices = "CUDA0,RPC0",
    [string]$HeadSplit = "20,20",
    [string]$FfnSplit = "8704,8704",
    [string]$OutputDir = ".\galaxy-sp-results",
    [double]$MaxAbs = 0.10,
    [double]$MaxRmse = 0.02,
    [switch]$Trace,
    [int]$TimeoutSec = 180,
    [int]$NPredict = 1,
    [int]$Ctx = 256,
    [int]$Batch = 256,
    [int]$UBatch = 256,
    [int]$Threads = 8
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "galaxy-process.ps1")
if ($MaxAbs -lt 0 -or $MaxRmse -lt 0) { throw "Error tolerances cannot be negative" }
if (-not (Test-Path -LiteralPath $Comparator)) { throw "Build galaxy-logits-compare first: $Comparator" }
& $Comparator --selftest
if ($LASTEXITCODE -ne 0) { throw "Galaxy logits comparator selftest failed" }
# Do not load Qwen3 until the Windows process exit-code regression is checked.
& (Join-Path $PSScriptRoot "test_galaxy_process.ps1")
$smoke = Join-Path $PSScriptRoot "run_galaxy_sp_smoke.ps1"
if (-not (Test-Path -LiteralPath $smoke)) { throw "Missing smoke script: $smoke" }
$root = Join-Path $OutputDir ("compare-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force -Path $root | Out-Null
$root = (Resolve-Path -LiteralPath $root).Path

$opts = @{
    Model = $Model
    PromptFile = $PromptFile
    Rpc = $Rpc
    Exe = $Exe
    Devices = $Devices
    HeadSplit = $HeadSplit
    FfnSplit = $FfnSplit
    OutputDir = $root
    DumpLogits = $true
    Trace = $Trace
    TimeoutSec = $TimeoutSec
    NPredict = $NPredict
    Ctx = $Ctx
    Batch = $Batch
    UBatch = $UBatch
    Threads = $Threads
}

Write-Host "[GALAXY_PARITY] stage=TP_REFERENCE dir=$root"
& $smoke @opts -PureTP
$tpDir = Get-ChildItem -LiteralPath $root -Directory -Filter "tp-*" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $tpDir) { throw "No TP output directory found in $root" }

Write-Host "[GALAXY_PARITY] stage=SP_TEST dir=$root"
& $smoke @opts
$spDir = Get-ChildItem -LiteralPath $root -Directory -Filter "sp-*" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $spDir) { throw "No SP output directory found in $root" }

$tpLogits = Join-Path $tpDir.FullName "first_logits.f32"
$spLogits = Join-Path $spDir.FullName "first_logits.f32"
$json = Join-Path $root "summary.json"
$stdout = Join-Path $root "compare.log"
$stderr = Join-Path $root "compare.stderr.log"
if (-not (Test-Path -LiteralPath $tpLogits) -or -not (Test-Path -LiteralPath $spLogits)) {
    throw "Missing full logits file from TP or SP: $root"
}
$ci = [System.Globalization.CultureInfo]::InvariantCulture
$maxAbsText = $MaxAbs.ToString("G17", $ci)
$maxRmseText = $MaxRmse.ToString("G17", $ci)
$arguments = @(
    "--tp", ('"' + $tpLogits + '"'),
    "--sp", ('"' + $spLogits + '"'),
    "--max-abs", $maxAbsText,
    "--max-rmse", $maxRmseText,
    "--json", ('"' + $json + '"')
)
$compareResult = Invoke-GalaxyProcess `
    -Executable (Resolve-Path -LiteralPath $Comparator).Path `
    -ArgumentList $arguments -StdoutPath $stdout -StderrPath $stderr
$compareExitCode = $compareResult.ExitCode
Get-Content -LiteralPath $stdout | ForEach-Object { Write-Host $_ }
if ((Get-Item -LiteralPath $stderr).Length -gt 0) {
    Get-Content -LiteralPath $stderr | ForEach-Object { Write-Host $_ }
}
$manifest = @{
    model = (Resolve-Path -LiteralPath $Model).Path
    prompt = (Resolve-Path -LiteralPath $PromptFile).Path
    prompt_sha256 = (Get-FileHash -LiteralPath $PromptFile -Algorithm SHA256).Hash
    rpc = $Rpc
    devices = $Devices
    heads = $HeadSplit
    ffn = $FfnSplit
    tp_dir = $tpDir.FullName
    sp_dir = $spDir.FullName
    logits_summary = $json
    comparator_exit_code = $compareExitCode
    max_abs_threshold = $MaxAbs
    rmse_threshold = $MaxRmse
    note = "FP32 first-token logits comparison; not a speed benchmark"
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $root "manifest.json") -Encoding UTF8
if ($compareExitCode -ne 0) {
    throw "Galaxy TP/SP numerical parity failed: $root (exit=$compareExitCode)"
}
Write-Host "[GALAXY_PARITY] PASS summary=$json"

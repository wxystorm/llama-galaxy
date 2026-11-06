<#
Run the first TWO prompts from each 128/256/512/1024 group (8 cases total)
with the existing llama-galaxy PC+Phone smoke runner.

PowerShell 5.1+; does not rely on $common or any persisted environment variable.

Example (from E:\llama\llama-galaxy):
  .\tools\galaxy\run_galaxy_14b_prompt2_sweep.ps1 -PromptDir 'D:\YOUR_PROMPT_FOLDER'
  .\tools\galaxy\run_galaxy_14b_prompt2_sweep.ps1 -PromptDir 'D:\YOUR_PROMPT_FOLDER' -Mode Galaxy2
  .\tools\galaxy\run_galaxy_14b_prompt2_sweep.ps1 -PromptDir 'D:\YOUR_PROMPT_FOLDER' -Transport HOST
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PromptDir,
    [string]$RepoRoot = 'E:\llama\llama-galaxy',
    [string]$Model = 'E:\llama\models\Qwen3-14B-Q4_K_M.gguf',
    [string]$Rpc = '10.39.52.114:50052',
    [ValidateSet('Galaxy', 'Galaxy2', 'Both')]
    [string]$Mode = 'Galaxy',
    [ValidateSet('HOST', 'RANK_LOCAL')]
    [string]$Transport = 'RANK_LOCAL',
    [ValidateRange(1, 40)]
    [int]$Galaxy2Layers = 1,
    [ValidateRange(1, 1024)]
    [int]$CollectiveTileTokens = 256,
    [int]$Ctx = 2048,
    [int]$Batch = 512,
    [int]$UBatch = 256,
    [int]$Threads = 8,
    [int]$TimeoutSec = 900,
    [string]$OutputRoot = '.\galaxy-sp-results\prompt2-14b',
    [switch]$FinalLayerSP,
    [switch]$DumpLogits,
    [switch]$Profile
)

$ErrorActionPreference = 'Stop'

# Resolve all paths once: the underlying runner invokes an external binary.
$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$prompts = (Resolve-Path -LiteralPath $PromptDir).Path
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$runner = Join-Path $repo 'tools\galaxy\run_galaxy_sp_smoke.ps1'
$exe = Join-Path $repo 'build-galaxy\bin\Release\llama-cli.exe'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Missing Galaxy smoke runner: $runner"
}
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "Missing llama-cli executable: $exe"
}
if ($Ctx -le 0 -or $Batch -le 0 -or $UBatch -le 0 -or $UBatch -gt $Batch) {
    throw 'Require Ctx > 0 and 0 < UBatch <= Batch.'
}

# Fail before model loading if any of the 8 expected files are missing.
$cases = @()
$missing = @()
foreach ($group in @(128, 256, 512, 1024)) {
    foreach ($index in @(1, 2)) {
        $filename = 'prompt_{0}_{1:D2}.txt' -f $group, $index
        $path = Join-Path $prompts $filename
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            $missing += $path
        }
        $cases += [pscustomobject]@{
            Group = $group
            Index = $index
            FileName = $filename
            Path = $path
        }
    }
}
if ($missing.Count -gt 0) {
    throw "Expected 8 prompt files; missing:`n$($missing -join "`n")"
}

$baseOutput = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot
} else {
    Join-Path $repo $OutputRoot
}
$sessionDir = Join-Path $baseOutput ('run-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null
$sessionDir = (Resolve-Path -LiteralPath $sessionDir).Path
$csv = Join-Path $sessionDir 'summary.csv'
$results = New-Object 'System.Collections.Generic.List[object]'
$modes = if ($Mode -eq 'Both') { @('Galaxy', 'Galaxy2') } else { @($Mode) }

Write-Host "[GALAXY_14B_SWEEP] cases=$($cases.Count) modes=$($modes -join ',') transport=$Transport output=$sessionDir"
Write-Host '[GALAXY_14B_SWEEP] groups=128,256,512,1024 indices=01,02'
if ($Mode -ne 'Galaxy' -and $Galaxy2Layers -ge 20) {
    Write-Warning 'Many replicated Galaxy2 FFN layers may exceed 6 GB CUDA / Phone memory on Qwen3-14B.'
}

foreach ($entry in $cases) {
    foreach ($caseMode in $modes) {
        $tag = ('{0}-{1}' -f $caseMode.ToLowerInvariant(), [System.IO.Path]::GetFileNameWithoutExtension($entry.FileName))
        $caseRoot = Join-Path $sessionDir $tag
        New-Item -ItemType Directory -Path $caseRoot -Force | Out-Null

        # Explicit hashtable: no @common, no duplicate PowerShell switches.
        $opts = @{
            Model = $modelPath
            PromptFile = $entry.Path
            Rpc = $Rpc
            Exe = $exe
            Devices = 'CUDA0,RPC0'
            HeadSplit = '20,20'
            FfnSplit = '8704,8704'
            OutputDir = $caseRoot
            CollectiveTileTokens = $CollectiveTileTokens
            NPredict = 1
            Ctx = $Ctx
            Batch = $Batch
            UBatch = $UBatch
            Threads = $Threads
            TimeoutSec = $TimeoutSec
        }
        if ($Transport -eq 'RANK_LOCAL') { $opts['RankLocalCollectives'] = $true }
        if ($caseMode -eq 'Galaxy2') {
            $opts['Galaxy2'] = $true
            $opts['Galaxy2FfnSpLayers'] = [string]$Galaxy2Layers
        }
        if ($FinalLayerSP) { $opts['FinalLayerSP'] = $true }
        if ($DumpLogits) { $opts['DumpLogits'] = $true }
        if ($Profile) { $opts['Profile'] = $true }

        Write-Host "`n[GALAXY_14B_SWEEP] START $tag transport=$Transport"
        $status = 'PASS'
        $errorText = ''
        try {
            & $runner @opts
        }
        catch {
            $status = 'FAIL'
            $errorText = $_.Exception.Message -replace '[\r\n]+', ' '
            Write-Warning "[GALAXY_14B_SWEEP] FAIL $tag : $errorText"
        }

        # Existing runner creates one sp-* or galaxy2-* folder in this case root.
        $latest = Get-ChildItem -LiteralPath $caseRoot -Directory |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        $logDir = if ($null -ne $latest) { $latest.FullName } else { $caseRoot }
        $logPaths = @(
            (Join-Path $logDir 'stderr.log'),
            (Join-Path $logDir 'stdout.log')
        ) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }

        $actualTokens = $null
        $prefillMs = $null
        $prefillTPS = $null
        if ($logPaths.Count -gt 0) {
            $timingLines = @(Select-String -LiteralPath $logPaths -Pattern 'prompt eval time\s*=' -ErrorAction SilentlyContinue)
            if ($timingLines.Count -gt 0) {
                $line = $timingLines[-1].Line
                $metric = [regex]::Match($line, 'prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens')
                if ($metric.Success) {
                    $prefillMs = [double]::Parse($metric.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
                    $actualTokens = [int]$metric.Groups[2].Value
                }
                $rate = [regex]::Match($line, '([0-9.]+)\s*tokens per second')
                if ($rate.Success) {
                    $prefillTPS = [double]::Parse($rate.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
                }
            }
        }
        if ($status -eq 'PASS' -and $null -eq $prefillMs) {
            $status = 'PASS_NO_TIMING'
        }

        $results.Add([pscustomobject]@{
            Group = $entry.Group
            Prompt = $entry.FileName
            Mode = $caseMode
            Transport = $Transport
            Status = $status
            ActualTokens = $actualTokens
            PrefillMs = $prefillMs
            PrefillTPS = $prefillTPS
            LogDir = $logDir
            Error = $errorText
        })
        # Save after every case, including failures: Ctrl+C will not lose all rows.
        $results | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
        Write-Host "[GALAXY_14B_SWEEP] END $tag status=$status tokens=$actualTokens prefill_ms=$prefillMs prefill_tps=$prefillTPS"
    }
}

Write-Host "`n[GALAXY_14B_SWEEP] SUMMARY: $csv"
$results | Format-Table Group, Prompt, Mode, Transport, Status, ActualTokens, PrefillMs, PrefillTPS -AutoSize
$failed = @($results | Where-Object { $_.Status -eq 'FAIL' })
if ($failed.Count -gt 0) {
    throw "$($failed.Count) cases failed. See summary.csv and per-case stderr.log under $sessionDir"
}

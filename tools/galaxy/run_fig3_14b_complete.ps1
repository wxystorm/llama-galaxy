<#
One-command Galaxy Fig.3 14B collection after the matching Phone RPC server is
already running with LLAMA_GALAXY_PHONE_PROFILE_FILE set to
$HOME/fig3-galaxy-phone-v3.csv. The Phone binary requires NO new changes
from the original unified-timeline branch.
This script builds PC, runs one unprofiled and one profiled inference,
copies the Phone timeline and runs STRICT gap-free 6-way validation.
#>
[CmdletBinding()]
param(
  [string]$Model = "E:\llama\models\Qwen3-14B-Q4_K_M.gguf",
  [string]$PromptFile = "E:\llama\datasets\sharegpt\sharegpt50_short\prompt_1024_01.txt",
  [string]$Rpc = "10.238.229.239:50052",
  [string]$PhoneHost = "10.238.229.239",
  [string]$PhoneUser = "u0_a283",
  [int]$PhoneSshPort = 8022,
  [string]$PhoneProfilePath = "/data/data/com.termux/files/home/fig3-galaxy-phone-v3.timeline.csv",
  [string]$PhoneTimeline = "",
  [string]$OutputRoot = ".\fig3-galaxy14b-complete",
  [int]$NormalRepeats = 1,
  [int]$TimeoutSec = 900,
  [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
# Do not enable StrictMode here: the legacy smoke runner uses scalar/optional
# PowerShell .Count accesses and must run in its original non-strict semantics.
Push-Location (Join-Path $PSScriptRoot "..\..")
try {
  if ($NormalRepeats -lt 1) { throw "NormalRepeats must be >= 1" }
  $branch = (& git branch --show-current).Trim()
  if ($branch -ne "exp/fig3-unified-phone-timeline-20261118") {
    throw "Wrong branch '$branch'. Switch to exp/fig3-unified-phone-timeline-20261118 first."
  }
  $commit = (& git rev-parse --short HEAD).Trim()
  if (-not $SkipBuild) {
    # Web UI is not needed, and the user's previous UI dist was stale.
    Remove-Item ".\build-galaxy\tools\ui\dist" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item ".\build-galaxy\tools\ui\.ui-stamp" -Force -ErrorAction SilentlyContinue
    & cmake -S . -B build-galaxy -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_UI_GZIP=OFF
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
    & cmake --build build-galaxy --config Release --target llama-cli --parallel 4
    if ($LASTEXITCODE -ne 0) { throw "PC build failed" }
  }

  # Catch parser/schema bugs using synthetic clocks, graphs and events BEFORE
  # spending minutes on the real normal+Profile inference passes.
  & python (Join-Path $PSScriptRoot "test_fig3_unified_timeline.py")
  if ($LASTEXITCODE -ne 0) { throw "Fig.3 analyzer synthetic preflight failed; model run not started" }

  $runId = "fig3-full-" + (Get-Date -Format "yyyyMMddHHmmss")
  New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
  $root = (Resolve-Path -LiteralPath $OutputRoot).Path
  $runner = Join-Path $PSScriptRoot "run_galaxy_sp_smoke.ps1"
  $normalPath = Join-Path $root "normal"
  $profilePath = Join-Path $root "profiled"
  $common = @{
    Model=$Model; PromptFile=$PromptFile; Rpc=$Rpc
    PureTP=$true; Ctx=2048; Batch=2048; UBatch=256; NPredict=1
    Threads=8; TimeoutSec=$TimeoutSec
  }

  # Run the ACTUAL Galaxy PowerShell smoke script through its argument
  # validation and setup, without starting any model inference. This catches
  # inherited StrictMode and parameter binding failures that synthetic CSV
  # tests alone cannot detect.
  Write-Host "[FIG3_FULL] PowerShell smoke preflight (no inference)"
  & $runner @common -DryRun -OutputDir (Join-Path $root "preflight")
  Write-Host "[FIG3_FULL] PowerShell smoke preflight passed"

  $measured = @()
  for ($i=1; $i -le $NormalRepeats; ++$i) {
    Write-Host "[FIG3_FULL] normal run $i / $NormalRepeats"
    & $runner @common -OutputDir $normalPath
    $lastDir = Get-ChildItem -LiteralPath $normalPath -Directory |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $lastDir) { throw "Normal output directory missing" }
    $text = (Get-Content -LiteralPath (Join-Path $lastDir.FullName "stdout.log") -Raw) +
      [Environment]::NewLine +
      (Get-Content -LiteralPath (Join-Path $lastDir.FullName "stderr.log") -Raw)
    $timeHits = [regex]::Matches($text, 'prompt eval time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*(\d+)\s*tokens')
    if ($timeHits.Count -eq 0) { throw "Unprofiled Prefill timing missing in $($lastDir.FullName)" }
    $hit = $timeHits[$timeHits.Count - 1]
    $value = [double]::Parse($hit.Groups[1].Value,
      [System.Globalization.CultureInfo]::InvariantCulture)
    $tokens = [int]::Parse($hit.Groups[2].Value)
    Write-Host "[FIG3_FULL] normal prefill_ms=$value tokens=$tokens"
    $measured += $value
  }
  $ordered = @($measured | Sort-Object)
  if ($ordered.Count % 2 -eq 1) {
    $median = $ordered[[int][math]::Floor($ordered.Count/2)]
  } else {
    $k = [int]($ordered.Count / 2)
    $median = ($ordered[$k-1] + $ordered[$k])/2.0
  }

  Write-Host "[FIG3_FULL] profiling run_id=$runId"
  $profArgs = @{}
  foreach ($key in $common.Keys) { $profArgs[$key] = $common[$key] }
  $profArgs.Profile = $true
  $profArgs.ProfileRunId = $runId
  $profArgs.ProfilePrompt = "prompt_1024_01"
  $profArgs.ProfileTokenGroup = 1024
  $profArgs.OutputDir = $profilePath
  & $runner @profArgs
  $profRun = Get-ChildItem -LiteralPath $profilePath -Directory |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($null -eq $profRun) { throw "Profile run output missing" }
  $dir = Join-Path $profRun.FullName "profile"
  $required = @(
    "summary.csv", "graph_timeline.csv", "clock_sync.csv",
    "allreduce_breakdown.csv", "rpc_transfer_timeline.csv",
    "pc_cuda_events.csv", "pc_cpu_events.csv"
  )
  foreach ($file in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $dir $file))) {
      throw "INCOMPLETE PROFILE: missing $file at $dir. Do not label missing measurements as 0."
    }
  }

  $destPhone = Join-Path $root "phone.timeline.csv"
  if ($PhoneTimeline) {
    if (-not (Test-Path -LiteralPath $PhoneTimeline)) { throw "PhoneTimeline absent: $PhoneTimeline" }
    Copy-Item -LiteralPath $PhoneTimeline -Destination $destPhone -Force
  } else {
    Write-Host "[FIG3_FULL] Archiving all three Phone telemetry CSVs over SSH..."
    $remote = $PhoneUser + "@" + $PhoneHost + ":"
    $nativePath = $PhoneProfilePath.Replace(".timeline.csv", ".csv")
    $phoneFiles = @(
      @{ Path=$PhoneProfilePath; Local=$destPhone }
      @{ Path=$nativePath; Local=(Join-Path $root "phone.native.csv") }
      @{ Path=$nativePath.Replace(".csv", ".events.csv"); Local=(Join-Path $root "phone.events.csv") }
    )
    foreach ($item in $phoneFiles) {
      & scp -P $PhoneSshPort ($remote + $item.Path) $item.Local
      if ($LASTEXITCODE -ne 0) {
        throw "Phone SCP failed for $($item.Path). PC logs are saved at $dir. Copy Phone CSVs manually and rerun only the analyzer, not inference."
      }
    }
  }

  $normalMs = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture,
    "{0:F5}", $median)
  $pythonArgs = @(
    (Join-Path $PSScriptRoot "build_fig3_unified_timeline.py"),
    "--profile-dir", $dir, "--phone-timeline", $destPhone,
    "--run-id", $runId, "--last-graphs", "5",
    "--normal-prefill-ms", $normalMs
  )
  & python @pythonArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Strict Fig.3 analysis failed. Raw events remain available at $dir, no model rerun needed for a parser fix."
  }
  $breakdown = @(Import-Csv -LiteralPath (Join-Path $dir "fig3_unified_breakdown.csv"))
  if ($breakdown.Count -ne 1 -or $breakdown[0].complete_inputs -ne "1" -or
      $breakdown[0].full_coverage -ne "1") {
    throw "Fig.3 source validation did not pass"
  }
  $v = $breakdown[0]
  $row = [PSCustomObject]@{
    Baseline = "Galaxy"
    NormalPrefillMs = $median
    ProfileWindowMs = $v.profile_envelope_ms
    GPUComputePct = $v.gpu_compute_pct
    PhoneComputePct = $v.phone_compute_pct
    CPUComputePct = $v.cpu_compute_pct
    PCIeCopyPct = $v.pcie_copy_pct
    WiFiSyncPct = $v.wi_fi_sync_pct
    WaitingOtherPct = $v.waiting_other_pct
    OverlapMs = $v.overlap_ms
    GraphGapMs = $v.graph_gap_ms
    ClockUncertaintyMs = $v.clock_uncertainty_ms
    Classification = "exclusive priority projection, not hardware utilization"
  }
  $row | Export-Csv -LiteralPath (Join-Path $root "fig3_galaxy_14b_paper_row.csv") -Encoding UTF8 -NoTypeInformation
  @(
    "GitBranch=$branch"
    "GitCommit=$commit"
    "RunId=$runId"
    "Model=$Model"
    "PromptFile=$PromptFile"
    "Rpc=$Rpc"
    "ProfileDir=$dir"
    "PhoneTimeline=$destPhone"
    "NormalPrefillMs=$median"
    "NormalRepeats=$NormalRepeats"
    "ClockUncertaintyMs=$($v.clock_uncertainty_ms)"
    "ClassSemantics=exclusive priority projection; WiFi category includes cross-device synchronization"
  ) | Set-Content -LiteralPath (Join-Path $root "manifest.txt") -Encoding UTF8
  Write-Host "[FIG3_FULL] PASS. Timelines and breakdown: $dir"
  Write-Host "[FIG3_FULL] Paper row: $(Join-Path $root 'fig3_galaxy_14b_paper_row.csv')"
  Write-Host "[FIG3_FULL] Normal Prefill total: $median ms"
} finally {
  Pop-Location
}

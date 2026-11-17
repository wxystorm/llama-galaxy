# Fig. 3 Galaxy 14B: unified PC/Phone timeline

Phone CPU/GPU/HTP are now represented by ONE category: Phone Compute.

## Timing semantics

This branch adds an optional RPC clock-sync command (feature negotiated) and
a Phone native graph timeline, plus real PC Meta graph start/end time.
Clock synchronization uses a four-timestamp ping to estimate Phone minus PC
steady-clock offset. The timeline script uses the minimum-uncertainty sample.

Output is a single, mutually exclusive, priority-based diagnostic projection.
When Phone Compute overlaps a PC AllReduce wait, the interval is assigned to
Phone Compute ONCE. Wi-Fi Sync means cross-device communication/sync path,
not pure Wi-Fi transmission. Waiting/Other is the residual and, without
independent PC GPU/CPU/PCIe traces, is NOT verified hardware idle.
This is NOT an independently verified critical-path attribution.

Existing (pre-patch) CSVs do not contain calibrated Phone start/end
timestamps; they cannot produce a verified single-axis timeline.

## Phone Termux

Stop the old Galaxy RPC server. Reuse the same build configuration and backend.

    cd ~/llama-galaxy
    git fetch origin
    git switch exp/fig3-unified-phone-timeline-20261118
    git pull --ff-only
    cmake --build build-galaxy --target ggml-rpc-server -j 4

    export LLAMA_GALAXY_PHONE_PROFILE_FILE="$HOME/fig3-galaxy-phone-v3.csv"
    rm -f "$HOME/fig3-galaxy-phone-v3.csv" "$HOME/fig3-galaxy-phone-v3.events.csv" "$HOME/fig3-galaxy-phone-v3.timeline.csv"
    ./build-galaxy/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf" -t 8

New Phone file: fig3-galaxy-phone-v3.timeline.csv with true Phone graph
prepare and compute intervals, in Phone monotonic nanoseconds.

## Windows PowerShell

    cd E:\llama\llama-galaxy
    git fetch origin
    git switch exp/fig3-unified-phone-timeline-20261118
    git pull --ff-only
    cmake --build build-galaxy --config Release --target llama-cli -j 8

    $rpc = "10.238.229.239:50052"
    $params = @{
      PureTP = $true
      Profile = $true
      ProfileRunId = "fig3-unified-14b-01"
      ProfilePrompt = "prompt_1024_01"
      ProfileTokenGroup = 1024
      Model = "E:\llama\models\Qwen3-14B-Q4_K_M.gguf"
      PromptFile = "E:\llama\datasets\sharegpt\sharegpt50_short\prompt_1024_01.txt"
      Rpc = $rpc
      Ctx = 2048
      Batch = 2048
      UBatch = 256
      NPredict = 1
      TimeoutSec = 900
      OutputDir = ".\fig3-galaxy-14b-timeline-v3"
    }
    .\tools\galaxy\run_galaxy_sp_smoke.ps1 @params

New PC files: profile/clock_sync.csv and profile/graph_timeline.csv.
Existing PC files: profile/summary.csv, profile/allreduce_breakdown.csv,
profile/rpc_transfer_timeline.csv.

If PC clock_sync.csv is absent, verify that Phone ggml-rpc-server has been
rebuilt from this branch (a legacy server lacks capability bit 2).

## Transfer Phone timeline to PC

With Termux sshd running (commonly SSH port 8022):

    scp -P 8022 u0_a283@10.238.229.239:/data/data/com.termux/files/home/fig3-galaxy-phone-v3.timeline.csv .

Adjust username, IP and port to current Termux configuration.

## One command to build single-axis timeline

    $dir = Get-ChildItem ".\fig3-galaxy-14b-timeline-v3" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $inputs = @{
      "profile-dir" = "$($dir.FullName)\profile"
      "phone-timeline" = ".\fig3-galaxy-phone-v3.timeline.csv"
      "run-id" = "fig3-unified-14b-01"
      "last-graphs" = 5
    }
    python .\tools\galaxy\build_fig3_unified_timeline.py --profile-dir $inputs["profile-dir"] --phone-timeline $inputs["phone-timeline"] --run-id $inputs["run-id"] --last-graphs 5

The analyzer produces: fig3_unified_timeline.csv (one contiguous categorical
timeline), fig3_unified_timeline.svg (visual preview),
fig3_unified_breakdown.csv (normalized proportions) and
fig3_source_events.csv (overlapping raw events for audit).

Optionally --pc-events may supply a CSV with
category,start_steady_ns,end_steady_ns in PC monotonic nanoseconds,
where category is GPU Compute, CPU Compute or PCIe Copy.
--normal-prefill-ms can store the unprofiled latency as separate metadata;
-Profile modifies scheduling, so its total is not a formal latency.

Final common Fig. 3 labels: GPU Compute / Phone Compute / CPU Compute /
PCIe Copy / Wi-Fi Sync / Waiting (operational). Do not use the resulting
Waiting/Other as a measured hardware idle fraction without PC device tracing.

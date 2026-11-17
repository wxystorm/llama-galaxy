# Fig.3 Galaxy 14B - one-command full profiling

Branch: exp/fig3-unified-phone-timeline-20261118

This branch adds PC CUDA/CPU/PCIe measurements to the existing Phone/AllReduce
timeline, plus a one-command Windows runner. It replaces the earlier incomplete
Phone-vs-AllReduce diagnostic breakdown.

## Classification and measurement definitions

- Phone Compute: all Phone native graph executions, backend type not guessed.
- GPU Compute: PC CUDA stream event intervals for native backend graph execution.
  This is stream device time, NOT per-kernel active utilization.
- CPU Compute: PC CPU backend native graph execution intervals.
- PCIe Copy: PC CUDA device-time events of standard H2D/D2H synchronous,
  asynchronous and 2D copies, with direction and bytes. D2D is not PCIe.
- Wi-Fi Sync: RPC GET/SET / AllReduce synchronization after subtracting
  intervals that the projection assigns to device work. NOT RF-only wire time.
- Waiting/Other: the remaining Prefill envelope, including Meta graph gaps
  and host bookkeeping not measured by CPU-backend graph intervals.

All 6 classes are mutually exclusive via a deterministic projection priority:
Phone Compute > PCIe Copy > GPU Compute > CPU Compute > Wi-Fi Sync > Waiting.
This policy does not prove causal critical-path attribution. Raw overlapping
device activities are preserved and reported separately. Device CUDA events
may exclude some host pageable-memory staging or scheduling overhead.

Strict mode FAILS on missing PC CUDA or CPU files, missing Phone native events,
clock sync error, Graph coverage error or AllReduce count discrepancy. This
prevents accidentally treating missing measurements as 0%.

## Phone setup (only once)

The original v3 Phone RPC profiling and clock-sync code is UNCHANGED by this
PC telemetry patch. If your Phone already generates
fig3-galaxy-phone-v3.timeline.csv, do NOT rebuild the Phone again.

For a fresh Phone setup, stop the old server, then run in Termux:

    cd ~/llama-galaxy
    git fetch origin exp/fig3-unified-phone-timeline-20261118
    git switch exp/fig3-unified-phone-timeline-20261118
    git pull --ff-only
    cmake --build build-galaxy --target ggml-rpc-server -j 4
    export LD_LIBRARY_PATH="$PWD/build-galaxy/bin:$LD_LIBRARY_PATH"
    export LLAMA_GALAXY_PHONE_PROFILE_FILE="$HOME/fig3-galaxy-phone-v3.csv"
    ./build-galaxy/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf" -t 8

Phone files (all under Termux home):
    fig3-galaxy-phone-v3.csv
    fig3-galaxy-phone-v3.events.csv
    fig3-galaxy-phone-v3.timeline.csv

For automatic PC SCP, enable Termux sshd (commonly port 8022).

## PC Windows PowerShell: ONE command

    cd E:\llama\llama-galaxy
    git fetch origin exp/fig3-unified-phone-timeline-20261118
    git switch exp/fig3-unified-phone-timeline-20261118
    git pull --ff-only
    .\tools\galaxy\run_fig3_14b_complete.ps1

The wrapper performs ALL of these automatically:
  1. Fix stale CMake Web UI dist and compile PC llama-cli with existing
     CUDA and RPC settings, but without embedded Web UI assets.
  2. Run one normal -PureTP inference and record genuine unprofiled
     Prefill wall-time for the bar top label.
  3. Run one profiled -PureTP inference with a unique Phone run_id and
     automatically capture all PC/Phone timing sources.
  4. Verify PC file completeness and SCP all three Phone CSVs.
  5. Run strict analysis and output plot-ready 100% totals, raw concurrent
     event lanes, all inter-Graph gap time and a versioned manifest.

The wrapper uses the user's 14B model path and prompt_1024_01 by default.
Change -PhoneHost, -PhoneUser, -PhoneSshPort or -PhoneProfilePath if needed.
Use -NormalRepeats 3 for a median of 3 normal runs (default 1 to save time).

There are two automatic inference passes by design: one normal and one
Profile. Profiling inserts synchronization, so its wall-time is not a fair
normal baseline latency. No further manual model re-runs should be needed
for missing columns when all sources are emitted correctly.

If SSH copy fails, the PC profile already exists and is preserved. Copy
the Phone CSVs manually and run only the analyzer, not another inference.

## Outputs

PC profile folder:
- summary.csv: Meta graph counts/durations.
- graph_timeline.csv: PC monotonic graph start/end for full coverage.
- clock_sync.csv: PC/Phone monotonic clock calibration samples.
- allreduce_breakdown.csv: all collective intervals.
- rpc_transfer_timeline.csv: PC RPC request intervals.
- pc_cuda_events.csv: GPU backend stream compute and PCIe transfer device time.
- pc_cpu_events.csv: CPU native backend graph compute time (header-only
  if the PC CPU backend performs no timed graph).
- fig3_source_events.csv: raw concurrent aligned sources.
- fig3_unified_timeline.csv: 100% covered, exclusive time segments.
- fig3_graph_coverage.csv: selected Meta Graph windows and gaps.
- fig3_unified_breakdown.csv: source coverage checks, overlap duration,
  all six proportions and cross-device clock uncertainty.
- fig3_unified_timeline.svg: GPU, Phone, CPU, PCIe and synchronization
  time lanes plus an exclusive normalized time line.

Top-level experiment root:
- phone.native.csv / phone.events.csv / phone.timeline.csv
- fig3_galaxy_14b_paper_row.csv (six category percentages plus separate
  unprofiled wall-time from the same automated experiment)
- manifest.txt with run_id, Git revision, model/prompt and source paths.

Analyzer manually (if needed):

    python .\tools\galaxy\build_fig3_unified_timeline.py --profile-dir "YOUR\profile" --phone-timeline "phone.timeline.csv" --run-id "YOUR_RUN_ID" --last-graphs 5 --normal-prefill-ms 102639.99

Only use --allow-incomplete while debugging, never for the paper.

## Important scientific caveats

This is a hardware-instrumented diagnostic occupancy projection, NOT proof
of an exact causal critical path, radio-only Wi-Fi time, per-kernel busy time
or real physical Idle rate. Raw overlapping event times are separately
exported so analysis can be audited. The phone backend category is
intentionally unified.

Code is committed and statically reviewed, but it has NOT yet been compiled
and run on the specific Windows NVIDIA plus Termux setup by the assistant.
If validation fails after the single run, preserve the CSVs and patch the
analyzer; do not rerun model inference unless the missing data came from a
failed backend binary.

# Fig. 3(a): Galaxy 14B pure TP diagnostic breakdown

This isolated branch adds opt-in diagnostics to Meta AllReduce and the RPC client.
Run -PureTP, not the default TP/SP implementation.

## PC: Windows PowerShell

Phone Galaxy RPC server must already listen on port 50052.

    cd E:\llama\llama-galaxy
    git fetch origin
    git switch exp/fig3-galaxy-tp-breakdown-20261117
    git pull --ff-only
    cmake --build build-galaxy --config Release -j 8

    $rpc = "10.238.229.239:50052"
    $model = "E:\llama\models\Qwen3-14B-Q4_K_M.gguf"
    $prompt = "E:\llama\datasets\sharegpt\sharegpt50_short\prompt_1024_01.txt"

    # Formal latency: 3 repetitions, profiling OFF.
    1..3 | ForEach-Object {
      .\tools\galaxy\run_galaxy_sp_smoke.ps1 -PureTP -Model $model -PromptFile $prompt -Rpc $rpc -Ctx 2048 -Batch 2048 -UBatch 256 -NPredict 1 -TimeoutSec 900 -OutputDir ".\fig3-galaxy-14b-formal"
    }

    # One separate intrusive diagnostic pass.
    .\tools\galaxy\run_galaxy_sp_smoke.ps1 -PureTP -Profile -ProfileRunId "fig3-galaxy-14b" -ProfilePrompt "prompt_1024_01" -ProfileTokenGroup 1024 -Model $model -PromptFile $prompt -Rpc $rpc -Ctx 2048 -Batch 2048 -UBatch 256 -NPredict 1 -TimeoutSec 900 -OutputDir ".\fig3-galaxy-14b-profile"

    $dir = Get-ChildItem ".\fig3-galaxy-14b-profile" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    python .\tools\galaxy\analyze_fig3_tp.py --profile-dir "$($dir.FullName)\profile" --last-graphs 5

Check that the final five graph IDs are 1041 tokens / 256 ubatch = 5
microbatches. Warmup or decode may alter graph numbering in other builds.
For explicit graph selection, use --graph-ids 3,4,5,6,7.

## New profile artifacts

The runner preserves existing communication_timeline.csv, compute_timeline.csv,
and summary.csv. This branch also creates:

- allreduce_breakdown.csv: one row per graph/subgraph AllReduce with
  native_call_ms, copy_call_ms, add_dispatch_ms, completion_wait_ms,
  other_ms, wall_ms, copy_calls, copy_bytes, and PC steady-clock interval.
- rpc_transfer_timeline.csv: PC socket-level GET/SET/COPY/GRAPH RPC commands,
  their input/output bytes, send_ms, recv_wait_ms, and PC steady-clock interval.
- fig3_tp_diagnostic.csv: produced by analyze_fig3_tp.py, with an AllReduce
  count consistency check and the RPC events nested in AllReduce windows.

The script stops on inconsistent counts and warns if the new RPC file is
missing. Both the PC llama-cli and dynamically loaded RPC library must be
rebuilt and deployed for complete instrumentation.

## Phone-side native timing (existing feature)

Restart the existing phone Galaxy RPC server after exporting:

    export LLAMA_GALAXY_PHONE_PROFILE_FILE="$HOME/fig3-galaxy-phone.csv"

Keep the original phone device configuration. The existing RPC server has
native graph profiling if the deployed binary includes the hook. Its CSV
records graph UID, prepare_ms, compute_ms, and wall_ms; a sidecar events
CSV records the ProfileRunId when both endpoints support it.

PC and Phone steady clocks are NOT synchronized. Do not subtract their raw
timestamps. Record whether RPC0 actually uses HTP, OpenCL, or CPU.

## Measurement semantics (important for paper)

- Normal unprofiled prompt eval time is the formal Prefill latency.
- -Profile forces additional backend fences, and therefore changes latency.
- AllReduce consists of backend tensor copies and device-side ADD dispatch
  (or a native collective). It is NOT necessarily Host CPU reduction.
- RPC send_ms can include socket backpressure. RPC recv_wait_ms can include
  phone server processing and blocking waits, NOT just one-way Wi-Fi latency.
- The RPC timing events are NESTED inside copy/ADD/AllReduce timers; never
  add these overlapping totals.
- Rank0 observed time is not isolated CUDA kernel time; rank1 RPC observed
  time includes Phone compute + RPC round trips.
- No new OOM test is part of this branch.

Earlier reference from this prompt: normal Prefill 102639.99 ms (10.14 tok/s);
profiling Graph 3-7: 396 AllReduce, 105347.67 ms in 108144.3 ms graph wall
(97.41%). These figures were measured BEFORE the present instrumentation
changes. They are not results from the patched branch.

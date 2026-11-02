# Galaxy static tensor parallelism - phase 1

Original Galaxy-LM source: https://github.com/ysyisyourbrother/Galaxy-LM
Reference commit: 7b3dcae9b7dbec8ec173b71bc6460dfedfe8ee36.

This phase implements the static Attention-head and FFN-intermediate-size
allocation used by the Galaxy-LM source, but on the existing llama.cpp
Meta Device execution substrate (PC backend + phone RPC backend).

THIS IS NOT A COMPLETE GALAXY PERFORMANCE BASELINE YET:
- llama.cpp Meta Device still performs the reductions, not Galaxy-LM Gloo.
- SP, dynamic planning, tiled overlap and Chunk FIFO are NOT implemented.
- No validated PC/Phone correctness or timing measurements have been made.

Activation on the PC (not on the phone RPC server):

  $env:LLAMA_GALAXY_TP = '1'
  $env:LLAMA_GALAXY_TP_HEADS = '20,20'
  $env:LLAMA_GALAXY_TP_FFN = '8704,8704'

Then run llama-cli in --split-mode tensor with an explicit device list
PC first, RPC second, e.g. --device CUDA0,RPC0, and --rpc PHONE_IP:50052.
The mode is opt-in: the old behavior is unchanged when GALAXY_TP is unset.

Example for Qwen3-14B (check your GGUF model hparams first):
  n_head = 40; n_gqa = 5; n_ff = 17408
  heads = [20,20]; FFN units = [8704,8704]
  Every head count must be divisible by n_gqa; FFN counts must respect
  lcm(weight_quant_block_size, 128). Incorrect splits fail instead of rounding.

After CMake configuration, build the isolated model-free unit self-test:

  cmake --build build-galaxy --config Release --target galaxy-tp-selftest
  .\build-galaxy\bin\Release\galaxy-tp-selftest.exe

Expected: GALAXY_TP_SELFTEST PASS.

Next, compile PC and Phone, test local phone GGUF loads, verify that
[GALAXY_TP_SPLIT] logs match intended assignment and compare numerical
outputs with single-device inference before any performance claim.

## Phase 1A: ReduceScatter + AllGather standalone test

Implemented in ggml/src/ggml-backend-galaxy-collectives.h as standalone C++17
backend operations. These are NOT yet wired to model execution; pure TP
continues to use its original AllReduce. Inputs are GGML contiguous
[hidden, tokens] F32/F16. Each rank starts with its own partial contribution;
reduce_scatter sums the two contributions but retains only the rank-assigned
token slice. all_gather copies both reduced slices to both ranks.
Uneven splits are supported without padding. The host-staging implementation
uses backend tensor get/set (CPU, CUDA, RPC) and does not claim optimal
communication. A token tile bounds scratch space; Decode with a single token
should stay on the existing TP path.

Build on Windows:
  cmake --build build-galaxy --config Release --target galaxy-collectives-selftest -j 8
  .\build-galaxy\bin\Release\galaxy-collectives-selftest.exe

Expected: six GALAXY_COLLECTIVES_CASE PASS lines and a final
GALAXY_COLLECTIVES_SELFTEST PASS.

Optional phone RPC test (phone Galaxy ggml-rpc-server must be running):
  .\build-galaxy\bin\Release\galaxy-collectives-selftest.exe --pc CUDA0 --rpc 10.39.52.114:50052

The RPC test uses only synthetic tensors; the GGUF is not needed for this
test. Phase 1B will integrate these collectives into Qwen3 TP/SP graphs.

## Phase 1B: experimental Qwen3 TP/SP graph integration

Runtime markers are now emitted only for multi-token Qwen3 inference
when LLAMA_GALAXY_SP=1 AND LLAMA_GALAXY_TP=1. Default pure TP is unchanged.
Last layer and single-token Decode remain on pure TP.

Dataflow per intermediate layer:
  ATT TP_PARTIAL -> ReduceScatter -> Residual Sequence Scatter
  -> SP residual add / RMSNorm -> AllGather -> FFN TP_PARTIAL
  -> ReduceScatter -> SP residual add -> AllGather -> next layer.

Meta skips marker DUP kernels, materializes proper per-rank token shapes,
performs the three host-staged collective operations, and skips the old
AllReduce only when followed immediately by a ReduceScatter marker.
PC receives ceil(T/2) tokens and Phone floor(T/2), without a Planner.

THIS STAGE IS EXPERIMENTAL AND NOT YET VERIFIED BY REAL QWEN3 INFERENCE.
The standalone F32/F16 collective tests do not prove graph-level correctness.

Build the updated Windows targets:
  git pull --ff-only origin feat/galaxy-baseline
  cmake --build build-galaxy --config Release --target galaxy-collectives-selftest llama-cli -j 8
  .\build-galaxy\bin\Release\galaxy-collectives-selftest.exe

Then run the new script from the repo root (PowerShell one-line example):
  .\tools\galaxy\run_galaxy_sp_smoke.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -Rpc "10.39.52.114:50052" -PromptFile "E:\llama\llama-tensor-v2\single-prompt\prompt_64.txt"

Compare against the same command with -PureTP. Both modes use greedy temp=0.
The script saves separate stdout.log and stderr.log under galaxy-sp-results.
A successful SP run emits [GALAXY_SP_SUM] with transition counts. For the
40-layer Qwen3-14B model, the intended 39 SP layers imply per complete
graph: RS=78, ResidualScatter=39, AG=78. Verify rather than assume counts.

Only after numerical correctness has been established should speed or
communication costs be evaluated. The current collectives use host staging.

### Fix: callback rename should not hide a Galaxy transition

The Qwen3 graph builder tags each layout transition in its GGML DUP
`op_params` with a magic value and explicit operation kind. Meta reads this
metadata instead of inferring transitions from tensor names. This fixes
the case where `graph_get_cb()` relabels `GALAXY_AG_FFN_0` to `l_out-0`,
causing ordinary DUP split inference to reject a sequence-sharded input.

`galaxy-collectives-selftest` also checks that all three kinds survive
renaming, and that a normal DUP named `GALAXY_AG_FFN_0` does not accidentally
become a collective.


### Debugging a suspected hang after l_out-38

PC-only tracing is now opt-in. It logs GRAPH_BEGIN, SUBGRAPH_BEGIN,
DISPATCH_BEGIN/RETURN for each rank, COLLECTIVE_BEGIN/END,
ALLREDUCE_BEGIN/END and GRAPH_END. Note: In this fork RPC DISPATCH_RETURN follows the synchronous
RPC response, so the phone graph has completed on the server.
If the PC stops between COLLECTIVE_BEGIN and COLLECTIVE_END, the collective
is waiting for a backend get/set; if it stops at ALLREDUCE_BEGIN, the
Meta fallback reduction or its backend copy is waiting.

Run the usual smoke script with `-Trace -TimeoutSec 300` (or a longer
explicit limit for slow phone CPU). This saves stdout.log and stderr.log;
check the final 30-60 `[GALAXY_META_TRACE]` lines from stderr.log. The
tracing flag does not change tensor layouts, weights, or scheduling.
The phone RPC server does not need to be rebuilt.

### PC CLI single-turn smoke fix

The smoke script now passes `--single-turn`: the CLI was correctly producing
logits and a first response, but its interactive prompt `>` kept the process
alive until the script timeout. `-n 1` only limits tokens, not conversation
turns. The script also supplies `-lv 3` (INFO) by default so the existing
`[GALAXY_SP_SUM]` line can be captured, and `-lv 4` (TRACE) with `-Trace`.
On a genuine timeout, the final matching trace events are printed.
Both stdout.log and stderr.log continue to be saved under the run directory.


## Phase 1C: verify numerical parity of Qwen3 TP/SP against pure TP

The Qwen3 TP/SP path has produced a complete first-token logits vector
and a response, but its **numeric parity with pure TP is not certified**
until both paths are compared on exactly the same model, prompt and runtime
settings. The smoke CLI now uses \`--single-turn\` and log verbosity 3/4.

### Full-logits correctness experiment

The existing server already prints the first generated token's top-10.
For a stricter test, the opt-in environment variable
\`LLAMA_GALAXY_FIRST_LOGITS_PATH\` saves the complete first-token vocabulary
logits: 4-byte magic \`GLG1\`, int32 vocab size, then contiguous float32
logits. The variable is scoped by the smoke script and is disabled by
default. Only the first generated token's logits are dumped; the file
remains local to the PC.

The model-free \`galaxy-logits-compare\` executable loads two such dumps and
reports Top-1 equality, Top-10 overlap, max absolute error, mean absolute
error, RMSE, cosine similarity, and any differing non-finite entries.
Its numeric thresholds are configurable; the exploratory defaults are
max-abs 0.10 and RMSE 0.02. A matching Top-1 alone is NOT sufficient to
declare numerical parity.

Build:
 
    git pull --ff-only origin feat/galaxy-baseline
    cmake --build build-galaxy --config Release --target llama-cli galaxy-logits-compare -j 8
    .\build-galaxy\bin\Release\galaxy-logits-compare.exe --selftest

One matched TP/SP pair (PowerShell, from repository root):

    .\tools\galaxy\run_galaxy_tp_sp_compare.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -Rpc "10.39.52.114:50052" -PromptFile "D:\prima.cpp\prompts\prompt_64.txt" -TimeoutSec 180

This script runs pure TP first, then TP/SP with the SAME options, and
captures \`first_logits.f32\` for both. It calls the comparator and stores:
- \`galaxy-sp-results/compare-*/summary.json\` for metric values and thresholds
- \`compare.log\`, \`compare.stderr.log\` for comparison output
- \`manifest.json\` with source paths, prompt SHA-256, split config, RPC endpoint
- \`tp-*/stdout.log\`, \`tp-*/stderr.log\`, \`tp-*/first_logits.f32\`
- \`sp-*/stdout.log\`, \`sp-*/stderr.log\`, \`sp-*/first_logits.f32\`

Use \`-Trace\` for detailed Meta execution logs, which increase logging
overhead. A failed numerical comparison is reported as FAIL and the files
are retained for diagnosis. If the mismatch is above an exploratory
threshold, inspect the actual error values and token rankings before
deciding whether it represents a meaningful correctness bug.

This is a correctness check, NOT a reliable decode or prefill speed
benchmark; binary file output and verbose logs add overhead. Keep phone
RPC and model weight configuration unchanged between the two runs.


### Windows process exit-code regression (fixed)

A TP run can successfully finish, print first-token logits and Prefill timing,
but the original PowerShell smoke wrapper may show \`exit=\` (empty) and
erroneously fail. \`Start-Process -NoNewWindow\` with stdout/stderr
redirection can return a Process object with a null \`ExitCode\` on
Windows; it is a wrapper problem, not evidence of a Qwen3 TP/SP failure.

Both smoke and numerical-parity scripts now use the shared
\`tools/galaxy/galaxy-process.ps1\` helper, which creates
\`System.Diagnostics.Process\` directly, drains stdout/stderr
asynchronously to files, enforces a timeout and returns a real integer
exit code. Logs remain available after successful completion or timeout.

The parity script runs \`test_galaxy_process.ps1\` before loading the
model. The small test checks exit codes 0 and 7 and one-second timeout
handling with the native Windows \`cmd.exe\`; expected output:
\`GALAXY_PROCESS_SELFTEST PASS exit0=0 exit7=7 timeout=1\`.
Run the test alone if debugging process handling. These changes are
PowerShell-only and require no PC/Phone recompilation.

### Correctness summary logging under -lv 3

The common CLI maps `GGML_LOG_LEVEL_INFO` to internal TRACE verbosity 4,
not 3. Previously `[GALAXY_SP_SUM]` was written as `GGML_LOG_INFO`, so
a completed SP inference under the standard smoke runner (`-lv 3`) could
miss the summary and cause a false script failure. The Meta Device now
writes the short post-execution `[GALAXY_SP_SUM]` witness directly to stderr
and flushes it without relying on a verbosity level. Detailed
`[GALAXY_META_TRACE]` remains optional (`-Trace` uses `-lv 4`).

The smoke script no longer blocks an otherwise successful numerical
comparison when the witness is missing: it marks SP execution
`UNVERIFIED` and continues. The paired comparison records
`sp_collectives_confirmed` and the witness lines in its manifest.
A numerical match with no witness is reported as
`NUMERICAL_PASS sp_collectives=UNVERIFIED`, **not** as fully verified
Galaxy mixed TP/SP execution.

For runs completed with the older binary, compare already-exported
`tp-*/first_logits.f32` and `sp-*/first_logits.f32` directly with
`galaxy-logits-compare` without reloading the model.


## Phase 2: Galaxy2 full-weight SP FFN + opt-in performance profiling

### What is actually implemented

Galaxy2 keeps the ATTENTION projection in the original fixed head-parallel
TP layout. After Attention ReduceScatter, both ranks operate on different
token intervals. For each Galaxy2-enabled FFN layer, the full gate/up/down
weight matrices are MIRRORED on PC and Phone:

    Attention TP_PARTIAL
        -> RS
        -> residual/norm in SP_SEQ
        -> FFN in SP_SEQ (each rank has complete gate/up/down weights)
        -> residual add on local sequence
        -> AG
        -> next Attention TP

The FFN-side AG and RS are removed ONLY for selected layers; the
connection output still requires the final AG before the next TP Attention.
The last Qwen3 layer stays pure TP because of output-token selection;
single-token decode stays pure TP even when its FFN weights are mirrored.
Unselected layers retain the original mixed Galaxy FFN TP path.

Activation is explicit on the PC:
- LLAMA_GALAXY_TP=1
- LLAMA_GALAXY_SP=1
- LLAMA_GALAXY2=1
- LLAMA_GALAXY2_FFN_SP_LAYERS=all (the complete Galaxy2 implementation),
  or N (the first N non-final layers, as an experimental memory-capped
  Galaxy+Galaxy2 hybrid).

**IMPORTANT memory constraint:** Galaxy2 full-FFN replication is not just
a graph substitution. For Qwen3-14B with hidden=5120 and FFN=17408,
three FFN weight matrices across the 39 eligible layers contain approximately
10.4 billion parameters (before quantization). In Q4_K_M this means
several additional GiB of weights on EACH rank relative to split-FFN TP.
Full Galaxy2 is therefore unlikely to fit a 6 GiB RTX 4050 or the
phone's available memory at the same time as attention and KV tensors.
Actual memory depends on GGUF quantization and backend allocations.
The code does NOT stream these replicated FFN weights; do not interpret
the "all" mode as an offload/streaming solution.

Use N=1 for first integration testing on 14B; this is a PARTIAL hybrid,
not the paper's full Galaxy2. Use a smaller GGUF that fits both rank
memories to exercise "all" end-to-end. Trying "all" with insufficient
memory may fail model initialization.

### Profiling CSV (PC)

PC profiling is opt-in via LLAMA_GALAXY_PROFILE_DIR. The smoke runner's
-Profile switch sets that directory to its per-run subfolder.
Three separate files are generated:

- profile/compute_timeline.csv: graph, subgraph, rank=0/1, backend,
  last tensor, node count, dispatch_ms, sync_wait_ms, observed_ms.
- profile/communication_timeline.csv: graph, collective kind (RS, AG,
  RESIDUAL_SCATTER, ALLREDUCE), tensor, token count, read/write bytes,
  read_ms, host_reduce_ms, write_ms, sync_wait_ms, wall_ms.
- profile/summary.csv: graph, mode (TP, GALAXY, GALAXY2_PARTIAL or
  GALAXY2_FULL), subgraph count, collective counts, graph_wall_ms.

IMPORTANT: For per-rank actual elapsed observations, -Profile dispatches
both ranks and then synchronizes them at each subgraph; this is invasive.
The resulting Prefill tok/s and the CSV service times are not the same as
normal asynchronous inference. Communication get/set timings for Phone
include RPC/network time; they cannot uniquely isolate phone compute.
ALLREDUCE wall_ms also includes a forced completion fence, and its internal
network-byte breakdown is not separately observed here (0 bytes recorded
in the fields when not measured). Stage times can overlap, so do not sum
them to infer graph wall time.

### Phone native graph timing (optional; requires new phone build)

The RPC server can independently record its real native graph timings.
In Termux on the Phone, rebuild ggml-rpc-server from the updated branch,
then set LLAMA_GALAXY_PHONE_PROFILE_FILE to an absolute writable path
before launching the server. For example:

    export LLAMA_GALAXY_PHONE_PROFILE_FILE="$HOME/galaxy-phone-profile.csv"
    ./build-galaxy/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf" -t 8

The file records GRAPH_COMPUTE (graph preparation + native compute) and
GRAPH_RECOMPUTE (native compute, reused graph) separately, keyed by RPC
graph UID. PC and Phone timestamps are NOT automatically time-synchronized.

IMPORTANT RPC timing detail: ggml_backend_rpc_graph_compute sends a
GRAPH_COMPUTE RPC and waits for the server response, which the server
returns only after native backend computation finishes. Thus on PC,
rank=1 RPC dispatch_ms measures an RPC round trip INCLUDING phone
computation and server graph work; it is NOT the cheap queuing cost.
ggml_backend_rpc_synchronize is a no-op in this fork. The phone-side
compute_ms in galaxy-phone-profile.csv is the separate native compute
measurement. rank=0 CUDA observed_ms is a launch-to-post-RPC-fence
completion WINDOW that includes potential overlap with the phone RPC,
not a kernel-only CUDA time. Do not sum per-rank observed_ms to infer
graph wall time.

### Three-mode experiment (Windows PC)

Compile the updated inference and comparison tools:

    git pull --ff-only origin feat/galaxy-baseline
    cmake --build build-galaxy --config Release --target llama-cli galaxy-tp-selftest galaxy-collectives-selftest galaxy-logits-compare -j 8
    .\build-galaxy\bin\Release\galaxy-tp-selftest.exe
    .\build-galaxy\bin\Release\galaxy-collectives-selftest.exe

The following command runs exactly one TP, one Galaxy and one
layer-capped Galaxy2 pass on the same prompt, and compares the complete
first-token logits against pure TP, saving two comparison JSONs:

    .\tools\galaxy\run_galaxy_three_modes.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -Rpc "10.39.52.114:50052" -PromptFile "D:\prima.cpp\prompts\prompt_64.txt" -Galaxy2FfnSpLayers 1 -TimeoutSec 300

Output directory: galaxy-sp-results/three-YYYYMMDD-HHMMSS, including
tp-*, sp-* and galaxy2-* folders, each with first_logits.f32,
stdout.log, stderr.log and a profile/ subfolder. The comparisons are
compare-sp.json and compare-galaxy2.json; manifest.json records
layer-capped versus full Galaxy2.

Normal (not instrumented) performance runs should use
run_galaxy_sp_smoke.ps1 with -PureTP, no mode flag, or -Galaxy2
respectively, without -Profile. Do not interpret a 1-token generation
speed as a decode throughput measurement.

For the 40-layer Qwen3-14B, a single full multi-token graph with
39 eligible intermediate SP layers would perform:
- Pure Galaxy: RS=78, residual scatter=39, AG=78.
- Galaxy2 with N=1 selected: RS=77, residual scatter=39, AG=77.
- Full Galaxy2 with N=39: RS=39, residual scatter=39, AG=39.

Verify these numbers from [GALAXY_SP_SUM]; a model with fewer layers or
additional microbatches will have different counts. Do not claim SP/Galaxy2
correctness before full-logit parity and the actual collective witness pass.

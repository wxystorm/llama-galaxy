# Galaxy tensor/sequence parallelism + Galaxy2 (experimental)

Original Galaxy-LM source: https://github.com/ysyisyourbrother/Galaxy-LM
Reference commit: 7b3dcae9b7dbec8ec173b71bc6460dfedfe8ee36.

This fork implements three opt-in Qwen3 inference modes on the llama.cpp
Meta Device: static Attention+FFN TP; Galaxy mixed TP/SP with
ReduceScatter and AllGather; and Galaxy2 with replicated FFN weights and
token-sharded FFN computation. Galaxy2 can be limited to N layers to fit
a smaller PC/Phone memory budget (this is a partial hybrid, not full Galaxy2).

STATUS: pure TP and earlier mixed TP/SP forward inference have run on PC+Phone;
the new Galaxy2 and detailed CSV profiling must be compiled and validated.
The fork still uses Meta/RPC transport, not Galaxy-LM Gloo. It does not
implement a dynamic Planner, CPU/GPU weight streaming, or Chunk FIFO.
All initial speed measurements and numerical comparisons need repeatable
tests on the actual hardware.

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


## Phase 3: Host Staging RPC call reduction (A/B)

This phase intentionally does **not** gate performance development on
the existing TP-versus-SP full-vocabulary logit deviation. The prior
72-token experiment completed all modes and confirmed the expected
collective counts, but strict numerical parity remains an unresolved,
documented limitation. Do NOT describe it as fully equivalent TP/SP.

### How the communication optimization works

Previously each ReduceScatter, AllGather or Residual Scatter split a
device's local token interval into 32-token tiles. For a 72-token
prompt, the rank split is approximately 36+36 tokens, so *each* rank
required two tiles even though the second tile held only four tokens.
For longer prompts, the number of small RPC get/set calls grows rapidly.

The new environment variable
\`LLAMA_GALAXY_COLLECTIVE_TILE_TOKENS\` supports integers 1..1024:
- unset or 32: original correctness/performance baseline
- 256: experiment with fewer, larger RPC requests
- 512 or 1024: larger allocations; test only after 256 is stable

A fixed amount of Host buffer memory is reserved once per collective,
and reused across the tiles. For ReduceScatter this is three buffers
(partial rank 0, partial rank 1, reduced result). AllGather and Residual
Scatter each use one. All operations preserve exactly the same token
interval ordering, FP16/F32 addition order, and completion fences.
The corresponding phone RPC server protocol does not change.

For the 72-token / [36,36] case, the expected per-collective backend
tensor_get/set call counts are:

| Kind | tile=32 reads | tile=32 writes | tile=256 reads | tile=256 writes |
|------|---------------|----------------|----------------|-----------------|
| RS | 8 | 4 | 4 | 2 |
| AG | 4 | 8 | 2 | 4 |
| Residual Scatter | 4 | 4 | 2 | 2 |

These are *calls into ggml's backend tensor_get/set API*, not
necessarily distinct network packets. RPC requests on the Phone do
normally correspond to one network command per call; local CUDA
tensor reads/writes are counted as well. The number of bytes and the
arithmetic remain identical. Actual speed benefits require measurement.

The communication_timeline.csv profiler now includes:
\`read_calls,write_calls,sync_calls,tile_tokens\`, appended after
\`wall_ms\`. Existing numeric columns retain their meaning; old CSV
files have the old header and should not be combined without checking
their schema.

### Reproducible experiments

Update the PC (no phone rebuild needed for tile optimization):

    git pull --ff-only origin feat/galaxy-baseline
    cmake --build build-galaxy --config Release --target llama-cli galaxy-collectives-selftest galaxy-logits-compare -j 8
    .\build-galaxy\bin\Release\galaxy-collectives-selftest.exe

For the strongest direct PC/Phone primitive check, use:

    .\build-galaxy\bin\Release\galaxy-collectives-selftest.exe --pc CUDA0 --rpc 10.39.52.114:50052

Compare baseline tile=32 and optimized tile=256 with the SAME Galaxy
and one-layer Galaxy2 modes. Profiling is OFF by default so the
Prefill times exclude forced per-subgraph synchronization:

    .\tools\galaxy\run_galaxy_collective_tile_sweep.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -PromptFile "D:\prima.cpp\prompts\prompt_64.txt" -Rpc "10.39.52.114:50052" -Galaxy2FfnSpLayers 1 -TimeoutSec 300

The four runs produce:
- \`galaxy-sp-results/tile-sweep-*/timing.csv\` (Prefill latency and tps)
- \`tile-parity-sp.json\` and \`tile-parity-galaxy2.json\` comparing
  *within the same execution mode* (tile=32 against tile=256)
- \`manifest.json\` with prompt SHA-256, tile parameters, profile flag
- all four run folders, individual logs and first-token logits.

The tile A/B comparator still requires stable numerical output within
the same execution mode; this is a stronger requirement than accepting
the already observed SP-vs-TP deviations. A failed A/B comparator marks
the optimization as unsafe until investigated.

To inspect RPC call counts instead of unbiased latency, repeat with
\`-Profile\`. The CSVs then appear in each run's \`profile\` subfolder.
Profile mode is intentionally intrusive: do not use its tok/s to claim
end-to-end inference speedups.

Alternatively, select a tile for a single existing smoke or three-mode
run using \`-CollectiveTileTokens 256\`. The SP summary line now
includes \`tile_tokens=256\` as an execution witness.

The Galaxy2-FFN layer cap remains independent of the tile size; keep
\`Galaxy2FfnSpLayers=1\` on Qwen3-14B first. Full Galaxy2 is still
memory-limited and should be tested with a model small enough for
replicated FFN weights on both devices.


## Phase 4: optional fused ReduceScatter input reads

A successful first A/B run showed Galaxy 16,493.36 -> 15,111.96 ms
(1.091x) and Galaxy2 one-layer 16,687.60 -> 15,664.37 ms (1.065x)
when changing the Host Staging tile from 32 to 256 tokens. Both runs
preserved their mode's full first-token logits exactly (max_abs=0).
These are single measurements and are not yet confidence intervals.

The next opt-in optimization changes only ReduceScatter's source reads:
\`LLAMA_GALAXY_RS_FUSED_READ=1\` makes the PC fetch the *entire* partial
input once from CUDA and once from RPC, then reduce the same byte ranges
into each rank's existing output shards. Tensor arithmetic, F16 rounding
and destination writes remain in the original order. The old tiled
ReduceScatter path is the default and can be restored by setting 0.

Safety: a fused read is used only when EACH source is <=16 MiB.
It holds two inputs (<=32 MiB combined) plus a bounded tile-sized
output Host buffer. Larger inputs silently use the old tiled read
path; whether fused was actually used is recorded as \`rs_fused_read\`
in each communication_timeline.csv record. With a 72-token prompt
split 36+36 and tile=256, expected ReduceScatter backend tensor_get
calls drop from **4 to 2**, while output tensor_set calls remain 2.
The byte volume is unchanged. \`[GALAXY_SP_SUM]\` now reports
\`rs_fused_requested=1\`, which only indicates the requested mode;
the CSV reports whether a particular RS qualified for fused reads.

First verify \`galaxy-collectives-selftest\` (local and PC+Phone RPC).
Its new F16/F32 byte-for-byte RS parity checks compare fused vs
unfused reads on identical backend tensor buffers.

To run a single mode (PC only needs a rebuild):

    .\tools\galaxy\run_galaxy_sp_smoke.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -PromptFile "D:\prima.cpp\prompts\prompt_64.txt" -Rpc "10.39.52.114:50052" -CollectiveTileTokens 256 -RsFusedRead

To record \`rs_fused_read\`, \`read_calls\`, \`write_calls\` and
\`read_ms\` at the operation level, add \`-Profile\`. As before this
introduces forced synchronization, so compare non-profile Prefill
times separately. The phone server does not require protocol changes.

Accept the previously documented TP-vs-SP numerical deviations for
performance exploration, but require fused-vs-unfused *within-mode*
numerical parity before reporting a fused-read speedup.


For a matched four-run A/B harness with identical tile sizes and fused
ReduceScatter toggled off/on for both Galaxy and Galaxy2:

    .\tools\galaxy\run_galaxy_rs_fused_sweep.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -PromptFile "D:\prima.cpp\prompts\prompt_64.txt" -Rpc "10.39.52.114:50052" -Galaxy2FfnSpLayers 1 -CollectiveTileTokens 256 -TimeoutSec 300

It writes \`galaxy-sp-results/rs-fused-*/timing.csv\`,
\`parity-sp.json\`, \`parity-galaxy2.json\` and \`manifest.json\`, plus
all full first-token logits. The A/B numerical checks use max_abs=0 and
RMSE=0: exact equality is expected because fused input reads change
only data movement, not the reduction arithmetic.

To check the actual read/write call counts instead of clean Prefill
performance, run the same sweep with \`-Profile\`.
Its \`rpc_call_counts.csv\` summarizes total backend read/write calls
for RS and the number of RS operations that actually used fused reads.
These counts include BOTH backend ranks and may include initialization
and repeated graph execution; they are not unique TCP packets.
Do not compare Profile-on time against Profile-off time.


### Resume an interrupted fused-ReduceScatter sweep (Windows)

Older run_galaxy_rs_fused_sweep.ps1 passed Galaxy2FfnSpLayers twice:
once through the @opts hashtable and once explicitly. PowerShell
rejected the Galaxy2 invocation before model loading. The duplicate
argument has been removed.

An optional -ResumeDir parameter lets the fixed script reuse completed
mode/fused cases from the same sweep directory and run missing cases.
It checks each candidate's [GALAXY_SP_SUM] witness for the same tile
size and fused setting, its full first-token logits, and Prefill timing.
It also preserves the Profile-on/off distinction. Always use the SAME
model, prompt, RPC backend, head/FFN split, thread and batch settings
as the original interrupted run; old run directories do not carry a
complete per-case argument manifest.

For the interrupted 2026-11-03 experiment:

    .\tools\galaxy\run_galaxy_rs_fused_sweep.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -Rpc "10.39.52.114:50052" -PromptFile "D:\prima.cpp\prompts\prompt_64.txt" -Galaxy2FfnSpLayers 1 -CollectiveTileTokens 256 -TimeoutSec 300 -ResumeDir ".\galaxy-sp-results\rs-fused-20261103-090908"

Two completed Galaxy cases should show [GALAXY_RS_FUSED_SWEEP] REUSE.
The missing Galaxy2 OFF/ON cases will then execute; afterwards the
comparator and timing summary are computed from all four runs.
Resume does not require recompiling either endpoint since this patch
changes only PowerShell orchestration.


### Profile smoke: Windows relative log paths

When a user launches PowerShell from a different drive than the .NET
process current directory, \`New-Item .\galaxy-sp-results\...\` can create
the run directory on E: while \`System.IO.File.Create\` interprets
\`.\galaxy-sp-results\...\stdout.log\` under C:\Users\<account>, causing
DirectoryNotFoundException BEFORE any inference launches.

Both paths are now normalized to absolute PowerShell provider paths:
\`run_galaxy_sp_smoke.ps1\` resolves its new run directory immediately
and \`galaxy-process.ps1\` resolves both log streams and creates their
parent directories before opening the files. The independent
\`test_galaxy_process.ps1\` checks a nested *relative* log output path
and reports \`relative_path=PASS\` on success.

Run (no rebuild required):

    git pull --ff-only origin feat/galaxy-baseline
    .\tools\galaxy\test_galaxy_process.ps1
    .\tools\galaxy\run_galaxy_sp_smoke.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -Rpc "10.39.52.114:50052" -PromptFile "D:\prima.cpp\prompts\prompt_64.txt" -CollectiveTileTokens 256 -RsFusedRead -Profile -TimeoutSec 300

For old scripts, passing
\`-OutputDir "E:\llama\llama-galaxy\galaxy-sp-results"\`
also avoids this particular path mismatch.


## Phase 5: source-defined asymmetric sequence allocation

The upstream Galaxy-LM BERT configuration exposes \`seq_scatter_list\`
(e.g. [20,12]) alongside static attention-head and FFN split lists.
Previous Qwen3 Meta graphs hard-coded balanced (ceil(N/2),floor(N/2))
ownership. Asymmetric sequence ownership is now configurable through
\`LLAMA_GALAXY_SP_SEQ_WEIGHTS\`, exposed on the smoke runner as
\`-SeqWeights "20,12"\` (defaults to 1,1).

The values are **weights**, not fixed absolute token counts, because
llama.cpp also executes two-token warmup graphs. For an N-token graph,
the split is round(N*w0/(w0+w1)), clamped to [1,N-1].
Examples: N=72 with 20,12 -> [45,27]; N=32 -> [20,12]; N=2 ->
[1,1]. The same function is used for all SP transitions in each graph.
The existing attention-head and FFN splits are independent of these
sequence weights; this matches the distinct configuration dimensions
in the upstream implementation.

This does not add SP Attention with global K/V Gather. Upstream
Galaxy-LM defines that path for encoder-only BERT, but Qwen3's causal
KV-cache model needs separate masking, position and cache-ownership
handling. Until implemented, the Qwen3 modes remain TP, Galaxy,
Galaxy2 (full or per-layer-capped).


## Full last-layer SP (source CON2 AllGather semantics)

By default Qwen3 retains llama.cpp's last-layer optimization:
\`inp_out_ids\` selects only output tokens *before* the final FFN.
This means the last layer remains TP even when Galaxy2FfnSpLayers=all.
For a source-aligned, full-layer Galaxy/Galaxy2 prefill, use
\`-FinalLayerSP\` in the smoke runner
(\`LLAMA_GALAXY_SP_FINAL_LAYER=1\`).

This keeps the last layer's Attention/CON/FFN processing on the full
multi-token sequence, performs the final CON2 AllGather, then applies
\`inp_out_ids\` to the now-mirrored output. **It changes performance
characteristics:** the final layer computes all input tokens rather
than only the output token. Single-token decode remains pure TP.

Expected single-graph Qwen3-14B operation counts:
- Galaxy with FinalLayerSP: RS 80, AG 80, Residual Scatter 40
- Galaxy2 with FinalLayerSP and N=1: RS 79, AG 79, scatter 40
- Galaxy2 with FinalLayerSP and N=all: RS 40, AG 40, scatter 40

The last configuration mirrors all 40 FFN weight sets and is likely
out-of-memory on the user's 14B hardware; test with a smaller GGUF
first. Existing 39-eligible-layer experiments are unchanged unless
the switch is explicitly enabled.

Example:

    .\tools\galaxy\run_galaxy_sp_smoke.ps1 -Model "E:\llama\models\Qwen3-0.6B-Q4_K_M.gguf" -Rpc "10.39.52.114:50052" -PromptFile "D:\prima.cpp\prompts\prompt_64.txt" -HeadSplit "8,8" -FfnSplit "1536,1536" -Galaxy2 -Galaxy2FfnSpLayers all -FinalLayerSP -SeqWeights "20,12" -TimeoutSec 300


## SP Attention K/V collective from source Galaxy-LM

The original \`SPBertAttention.forward\` runs Q/K/V projections on a
per-rank sequence shard, then separately AllGathers K and V to full
sequence tensors before computing local-query attention. The fork now
includes \`all_gather_attention_kv\` as an independent backend/RPC
primitive, with F16/F32 unequal-shard and distinct-K/V tests in
\`galaxy-collectives-selftest\`. Unlike general SP connection AG, this
pair always gathers **both** K and V.

IMPORTANT: This is a primitive-level port, NOT an enabled Qwen3
pure-SP Attention mode. Qwen3 uses grouped-query causal attention,
rotary positions and a stateful KV cache; SP Q ownership and the causal
mask must be adapted as well. The existing TP, Galaxy and Galaxy2
Qwen3 paths do not call this new primitive. The test provides an
independent transport baseline for implementing that remaining mode.


## Source mode matrix and remaining scope

This fork uses Qwen3 on a PC+Phone Meta backend as its inference target.
The upstream Galaxy-LM repository implements the following BERT
model-parallel configurations with \`att_parallel_method\`,
\`mlp_parallel_method\` and \`con_parallel_method\`:

| Upstream mode | Attention | Connection | FFN | Qwen3 runtime status |
|---|---|---|---|---|
| TP | TP | None | TP | Implemented |
| Galaxy / Scheme-1 | TP | SP | TP | Implemented (prefill multi-token) |
| Galaxy2 / Scheme-2 | TP | SP | SP | Implemented (per-layer or all eligible; optional final layer) |
| SP | SP (full K/V gathers) | SP | SP | K/V collective ported/testable; Qwen3 graph not yet integrated |
| Mixed variant | SP (full K/V gathers) | SP | TP | Source transition contract only; Qwen3 graph not yet integrated |

A standalone \`galaxy-source-plan-selftest\` now checks all of those
transition decisions independently of model inference, so the runtime
port is evaluated against the upstream design rather than invented
collective counts. Build it with:

    cmake --build build-galaxy --config Release --target galaxy-source-plan-selftest -j 8
    .\build-galaxy\bin\Release\galaxy-source-plan-selftest.exe

**Do not claim complete source parity yet**: Qwen3's causal attention
uses stateful KV cache and attention masks, whereas the upstream
\`SPBertAttention\` is encoder-only and gathers K/V for local queries.
Porting the two K/V gathers is not enough to make SP Attention
correct for Qwen3; the Qwen3 attention graph, mask and cache placement
still need integration and validation. The upstream repository also
contains BERT training, adapters and PP variants; these are separate
from this fork's PC+Phone Qwen3 HMP inference goal.

Source references:
- https://github.com/ysyisyourbrother/Galaxy-LM/blob/main/README.md
- https://github.com/ysyisyourbrother/Galaxy-LM/blob/main/galaxy/models/bert/galaxy_bert_model.py
- https://github.com/ysyisyourbrother/Galaxy-LM/blob/main/galaxy/models/bert/sp_bert_model.py


### Source-aligned full Galaxy2 small-model validation

Qwen3-0.6B uses 28 transformer blocks, 16 attention heads, 8 KV heads
and a 3072-wide intermediate FFN. The 14B defaults (20,20 attention
heads and 8704,8704 FFN channels) are **not** valid for this model.
Use \`-HeadSplit "8,8" -FfnSplit "1536,1536"\` when validating the
small-model full Galaxy2 mode. These sums are from the official
Qwen3-0.6B model configuration.

Run once on the same prompt with pure TP, Galaxy Scheme-1 and full
Galaxy2 Scheme-2 (all FFN layers, including last), with a 20:12
sequence allocation and WITHOUT intrusive profiling:

    .\tools\galaxy\run_galaxy_three_modes.ps1 -Model "E:\llama\models\Qwen3-0.6B-Q4_K_M.gguf" -Rpc "10.39.52.114:50052" -PromptFile "D:\prima.cpp\prompts\prompt_64.txt" -HeadSplit "8,8" -FfnSplit "1536,1536" -Galaxy2FfnSpLayers all -FinalLayerSP -SeqWeights "20,12" -Profile $false -TimeoutSec 300

The full-graph expected counts for 28 layers are:
- Galaxy: Attention RS 28 + FFN RS 28 = 56, AG 56, Residual Scatter 28
- Galaxy2: Attention RS 28, AG 28, Residual Scatter 28

These values are for an actual full-sequence multi-token graph; the
smoke runner also emits 2-token warmup graphs. Full Galaxy2 reduces
collectives but replicates all FFN weights. If the 0.6B model is not
present at the specified path, change only the -Model argument to the
actual GGUF path; do not reuse the 14B attention/FFN splits.

The CLI experiment records TP-vs-SP numerical deviations separately
and does not require exact full-logit parity by default. Pure TP and
Galaxy modes still use their respective runtime paths; the source
reference-mode plan selftest has no influence on inference.


## Dense Llama 3.2 3B port (PC + Phone)

The dense \`LLM_ARCH_LLAMA\` graph is now wired to the same Attention
ReduceScatter -> Connection1 SP -> FFN TP/SP -> Connection2 AllGather
markers used in Qwen3. Its Llama-specific RoPE and causal KV cache
path are unchanged. A single-token decode graph remains pure TP.
The loader rejects MoE Llama when Galaxy SP is requested, rather than
silently inserting transitions across unsupported expert execution.

**Llama-3.2-3B-Instruct (GGUF Q4_K_M) model dimensions:**
- 28 layers; 24 attention heads; 8 KV heads; hidden=3072; FFN=8192.
- Equal static partitions: \`-HeadSplit "12,12"\` and
  \`-FfnSplit "4096,4096"\`.
- \`-SeqWeights "1,1"\` produces a balanced token split; \`"20,12"\`
  tests source-inspired nonuniform ownership.
- For the full Scheme-2 Galaxy2 configuration, use
  \`-Galaxy2FfnSpLayers all -FinalLayerSP\`. This replicates all 28 FFN
  layers on *both* ranks, and should be more feasible than the 14B
  model, but actual Meta/CUDA/Phone memory use must still be tested.

Suggested sequence: first baseline \`-PureTP\`, then one-layer
\`-Galaxy2 -Galaxy2FfnSpLayers 1\`, then full
\`-Galaxy2 -Galaxy2FfnSpLayers all -FinalLayerSP\`.

For one Llama 3.2 3B full-Galaxy2 smoke run:

    .\tools\galaxy\run_galaxy_sp_smoke.ps1 -Model "E:\llama\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf" -Rpc "10.39.52.114:50052" -PromptFile "D:\prima.cpp\prompts\prompt_64.txt" -HeadSplit "12,12" -FfnSplit "4096,4096" -Galaxy2 -Galaxy2FfnSpLayers all -FinalLayerSP -TimeoutSec 300

Expected per full 72-token Prefill graph (28 transformer blocks):
- Galaxy Scheme-1 with \`-FinalLayerSP\`: RS=56, AG=56, scatter=28
- Galaxy2 Scheme-2 all layers: RS=28, AG=28, scatter=28
- Galaxy2 N=1 with final SP: RS=55, AG=55, scatter=28

This port is **not yet Windows/Phone runtime validated**. Compilation,
correctness witnesses and output logits still need live checks on the
user's hardware. The upstream independent SP Attention/Q-KV strategy
remains separate work, not silently supported by the Llama port.


## Dense Llama FFN TP/SP stage numerical diagnosis (opt-in)

The cap=0/1/2 experiment on Llama-3.2-3B isolated the earliest discrepancy:
cap=0 matches Galaxy; cap=1 already differs at layer 0, and that difference
persists into layer 1. To distinguish FFN output discrepancies from
next-layer Attention amplification, `run_galaxy_sp_smoke.ps1` now accepts
`-DumpStages "0,1"`. It pins **only** `ffn_inp` (post-Attention residual)
and `ffn_norm` (normalized FFN input) of selected dense Llama layers,
and writes `stages/ffn_inp-N.f32` and `stages/ffn_norm-N.f32`.
Use it with the existing `-DumpLayers "0,1"` to also obtain
`layers/l_out-N.f32`.

Stage dumps use GLG1/F32 and reassemble both rank-local SP sequence shards
into logical `[hidden,tokens]` order. Pure TP's MIRRORED activations are
copied from rank 0. Unsupported layouts or incomplete shards cause a
**diagnostic failure** instead of emitting misleading values. No extra
Collective is introduced. The extra output pins, synchronization, and
readbacks are invasive: do not benchmark performance with `-DumpStages`.
The phone RPC server protocol is unchanged; only the Windows PC binary
needs rebuilding. These stage snapshots are currently enabled for the
dense Llama graph, not MoE.

For Llama 3.2 3B, with PC+Phone RPC running, from the repository root:

```powershell
git pull --ff-only origin feat/galaxy-baseline
cmake --build build-galaxy --config Release --target llama-cli -j 8
$common = @{
  Model = "E:\llama\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
  Rpc = "10.39.52.114:50052"
  PromptFile = "D:\prima.cpp\prompts\prompt_64.txt"
  HeadSplit = "12,12"
  FfnSplit = "4096,4096"
  SeqWeights = "1,1"
  DumpStages = "0,1"
  DumpLayers = "0,1"
  OutputDir = ".\galaxy-sp-results\stage-diagnostic"
  TimeoutSec = 300
}
.\tools\galaxy\run_galaxy_sp_smoke.ps1 @common -Galaxy2 -Galaxy2FfnSpLayers 0
.\tools\galaxy\run_galaxy_sp_smoke.ps1 @common -Galaxy2 -Galaxy2FfnSpLayers 1
```

Use the run directories printed by the wrapper:

```powershell
python .\tools\galaxy\galaxy-stage-compare.py `
  --reference ".\galaxy-sp-results\stage-diagnostic\galaxy2-<CAP0_RUN>" `
  --candidate ".\galaxy-sp-results\stage-diagnostic\galaxy2-<CAP1_RUN>" `
  --layers "0,1" --hidden 3072 --seq-weights "1,1" `
  --output ".\galaxy-sp-results\stage-diagnostic\stage-compare.json"
```

The analyzer reports FFN input and normalized input RMSE, plus two derived
logical contributions:

- `FFN_contribution[N] = l_out[N] - ffn_inp[N]`. This avoids comparing the
  TP per-rank partial FFN tensor against the SP complete local FFN output.
  Use this for layer 0 when deciding whether FFN TP/SP differ.
- `Attention_contribution[1] = ffn_inp[1] - l_out[0]`. This uses the
  previous full layer output and isolates the logical Attention addition
  in layer 1. Both run modes must use the same prompt/model settings.

All derived quantities are post-residual logical activations, not raw
operator dumps. The subtraction adds minor float32 rounding. Large
differences in `ffn_norm-0` would implicate inputs/normalization before
the FFN rather than the FFN itself. The stage comparator does not use
the logits Top-1 threshold as a correctness criterion.



## Final Llama3.2 Galaxy2 FFN Up/Gate/SwiGLU numerical experiment

The cap=0/1 stage diagnosis found identical `ffn_inp-0` and
`ffn_norm-0`, while layer-0 FFN contributions differ with
RMSE ~= 0.00044864. Layer-1 Attention contributions remain close, so the
last diagnostic tests whether the first divergence is in the FFN Up/Gate
quantized GEMMs or in the Down projection/ReduceScatter.

Opt-in `-DumpFfnOps "0"` pins precisely three layer-0 activations in
dense Llama and dumps them **in common logical [8192, tokens] order**:

- `ffn_ops/ffn_up-0.f32` -- Up projection before activation
- `ffn_ops/ffn_gate-0.f32` -- Gate projection before activation
- `ffn_ops/ffn_swiglu-0.f32` -- fused SwiGLU (Gate/Up activation product)

Llama 3.2 uses a fused `ggml_swiglu_split` implementation. An
`ffn_gate_par` node is **not** executed in its ordinary SwiGLU path.
The diagnostic therefore captures `ffn_swiglu` instead.

In FFN TP, the output is sharded across FFN feature dim 0; in FFN SP,
it is sharded across token dim 1. The exporter checks split axis and
source dimensions, performs the corresponding reconstruction, writes
GLG1/F32, and fails explicitly on unsupported/incomplete layouts.
The pre-existing stage comparator uses the final post-RS
`FFN_contribution` (= `l_out-0 - ffn_inp-0`) as the logical Down
projection result; the TP raw partial Down is NOT directly compared
against SP Down.

This does not change collectives or Phone RPC protocol. The output
pins and readback are intrusive; do not use these runs for tok/s claims.
Only the Windows PC `llama-cli` needs rebuilding.

```powershell
git pull --ff-only origin feat/galaxy-baseline
cmake --build build-galaxy --config Release --target llama-cli -j 8
python -m py_compile .\tools\galaxy\galaxy-stage-compare.py

$common = @{
  Model = "E:\llama\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
  Rpc = "10.39.52.114:50052"
  PromptFile = "D:\prima.cpp\prompts\prompt_64.txt"
  HeadSplit = "12,12"
  FfnSplit = "4096,4096"
  SeqWeights = "1,1"
  DumpStages = "0"
  DumpLayers = "0"
  DumpFfnOps = "0"
  OutputDir = ".\galaxy-sp-results\ffn-final-diagnostic"
  TimeoutSec = 300
}
.\tools\galaxy\run_galaxy_sp_smoke.ps1 @common -Galaxy2 -Galaxy2FfnSpLayers 0
.\tools\galaxy\run_galaxy_sp_smoke.ps1 @common -Galaxy2 -Galaxy2FfnSpLayers 1

# Set these to the real run folders printed in the two smoke logs:
$cap0 = ".\galaxy-sp-results\ffn-final-diagnostic\galaxy2-CAP0"
$cap1 = ".\galaxy-sp-results\ffn-final-diagnostic\galaxy2-CAP1"
python .\tools\galaxy\galaxy-stage-compare.py `
    --reference $cap0 --candidate $cap1 --layers 0 `
    --hidden 3072 --ffn-width 8192 --ffn-ops `
    --seq-weights "1,1" `
    --output ".\galaxy-sp-results\ffn-final-diagnostic\comparison.json"
```

Interpretation: If Up/Gate diverge, inspect FFN weight partitioning and
quantized projection numerics; if Up/Gate agree but SwiGLU diverges,
inspect fused activation; if all three agree within measurement noise
but `FFN_contribution` diverges, investigate Down projection/TP
ReduceScatter precision. This is a numerical diagnostic, not a proof
of correctness or exact output equivalence.


## Framework: noncontiguous per-layer FFN TP/SP schedule

Galaxy2 previously supported only the leading `N` SP-FFN layers, via
`LLAMA_GALAXY2_FFN_SP_LAYERS=N|all`. The framework now supports an
**explicit zero-based layer set**:

- Environment: `LLAMA_GALAXY2_FFN_SP_LAYER_SET=0,2,5-7`
- Smoke runner: `-Galaxy2 -Galaxy2FfnSpLayerSet "0,2,5-7"`
- Three-mode runner: `-Galaxy2FfnSpLayerSet "0,2,5-7"`
- `all` selects all eligible layers; `none` selects none.
- An explicit set must not be combined with the legacy leading-N option.
- The final layer is not eligible unless `-FinalLayerSP` is supplied.
- Decode remains on the existing one-token TP graph (unchanged).

Layer selection is resolved by one shared policy in
`src/llama-galaxy-tp.h`. Both model-weight placement and transformer
graph construction call `galaxy2_ffn_is_sp(il, n_layers)`, so an
explicit SP layer always mirrors its Up/Gate/Down FFN weights and avoids
its FFN-side AllGather/ReduceScatter, while every unselected layer uses
the original FFN-TP path. Attention-TP and connection collectives are
unchanged. This is not yet a pure-SP Attention implementation.

Example on Llama-3.2-3B with 28 layers and final-layer SP disabled:

```powershell
git pull --ff-only origin feat/galaxy-baseline
cmake --build build-galaxy --config Release --target llama-cli galaxy-tp-selftest -j 8
.\build-galaxy\bin\Release\galaxy-tp-selftest.exe

$common = @{
  Model = "E:\llama\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
  Rpc = "10.39.52.114:50052"
  PromptFile = "D:\prima.cpp\prompts\prompt_64.txt"
  HeadSplit = "12,12"
  FfnSplit = "4096,4096"
  SeqWeights = "1,1"
  OutputDir = ".\galaxy-sp-results\layer-schedule"
  TimeoutSec = 300
}
.\tools\galaxy\run_galaxy_sp_smoke.ps1 @common -Galaxy2 -Galaxy2FfnSpLayerSet "0,2,5-7"
```

On each complete multi-token prefill graph, with 27 eligible layers
and five chosen SP FFN layers, expected counts are:

- Attention RS = 27; FFN TP RS = 22; total RS = 49.
- Attention/connection-2 gathers = 27; FFN-TP input AG = 22;
  total AG = 49.
- Residual scatter = 27.

Multiple `[GALAXY_SP_SUM]` lines may be printed because the runtime
constructs multiple prefill graphs. These counts apply per full graph,
not to an aggregate over the process.

Cross-mode equivalence smoke: `-Galaxy2FfnSpLayerSet "0"` must choose
the same layer assignment as `-Galaxy2FfnSpLayers 1`. Both branches
should produce the same logits under identical execution settings.
Use the established first-logits dumps for comparison, rather than
assuming an ordinary PASS exit proves numerical correctness.
Selecting `"2"` instead also removes exactly one FFN-side RS and AG,
but selects a different layer with a different computation path.
There is no automatic performance planner in this patch; profiles
will be needed before choosing an optimal layer set.

The added tests cover noncontiguous sets, ranges, duplicates, bad
syntax, out-of-range layers, final-layer eligibility, `all`/`none`,
and compatibility with the original N-layer behavior.



## Causal SP Attention correctness baseline (standalone; runtime not enabled)

After verifying the noncontiguous Galaxy2 FFN schedule
(`0,2,5-7`, full-prefill collectives 49/27/49 on Llama-3.2-3B),
the next source-parity gap is *SP Attention* rather than FFN.

The original Galaxy-LM SPBertAttention gathers full K and V for each
rank's local Query shard. Directly copying that graph into Qwen3 or
Llama causal inference would be incorrect: existing `build_attn`
writes persistent KV-cache entries, then reads a cached K/V tensor
with a causal mask and absolute RoPE positions. A BERT-style one-shot
K/V gather omits cached history, and naive local Query offsets produce
wrong rotary positions and future-key visibility.

This stage therefore adds a **standalone, executable C++17 reference
contract**, not a switch to a partially correct production graph:

- `ggml/src/ggml-backend-galaxy-causal-plan.h`
  defines rank0/rank1 local Query ownership, absolute Query positions
  including the cached prefix, full logical K/V extent, causal
  visibility and GQA Query-to-KV head mapping.
- `tools/galaxy/galaxy-causal-attention-selftest.cpp`
  compares full unsharded attention against local Query shards with
  globally reconstructed K/V. Data uses deterministic per-head
  values, **absolute-position RoPE** and a numerically stable softmax.
  Cases include symmetric/asymmetric splits, nonzero historical cache
  (13 and 31 tokens), GQA (24/8 and 16/8), ordinary MHA (8/8) and a
  one-token decode fallback contract. The independent full-reference
  mask is compared with the SP helper's global-position visibility.
  A negative control detects accidental future-token access.
- The first implementation is intentionally limited to a single
  logical sequence with contiguous logical KV positions. Paged KV
  slots, multiple packed sequences, sliding windows, speculative or
  recurrent memory are **not** supported by this reference.

Build and run on Windows without starting Phone RPC:

```powershell
git pull --ff-only origin feat/galaxy-baseline
cmake --build build-galaxy --config Release --target galaxy-causal-attention-selftest -j 8
.\build-galaxy\bin\Release\galaxy-causal-attention-selftest.exe
```

Expected final output: `GALAXY_CAUSAL_ATTENTION_SELFTEST PASS`,
plus `[GALAXY_CAUSAL_SP_REFERENCE]` numerical comparisons.

**Runtime scope:** The new contract is not yet called by the Qwen3
or Llama computation graph, and does not modify existing TP, Galaxy,
Galaxy2, FFN layer schedules, RPC collectives or Decode. Passing the
standalone test is a prerequisite, not proof of correct distributed
causal inference.

For actual runtime SP Attention the remaining implementation is:
(1) full-width Q/K/V weights and local sequence-sharded Q with the
correct absolute position handling, (2) gather current K/V plus
historical cached K/V or establish a validated replicated cache,
(3) select the correct per-rank causal Mask rows without altering
global key visibility, (4) gather/shard residuals and FFN transitions,
and (5) verify first-token logits and chunked-prefill/decode
continuations. Inference paths must reject unsupported combinations
rather than silently pretending that source BERT-SP is already active.


## Backend-stage causal SP Attention on actual PC+Phone buffers (not model graph)

The standalone mathematical reference has now been extended to a **real
ggml-backend host-staged execution primitive**. This is the second stage of
the SP-Attention port, not a replacement for Qwen3/Llama inference.

Files:

- `ggml/src/ggml-backend-galaxy-causal-runtime.h`:
  `gather_kv_with_history()` and `host_staged_attention()`.
- `tools/galaxy/galaxy-causal-backend-selftest.cpp`:
  CPU+CPU and optional CUDA+RPC end-to-end checks.

Contract and actual data movement:

1. Rank 0 and rank 1 each own contiguous, already-RoPE-rotated Query,
   new Key and new Value tokens. Each rank's Q projection includes all
   Query heads; K/V include all KV heads. This requires full projected
   weights and **is not** compatible with the existing head-sharded TP
   Attention outputs without changing their weight layout.
2. A canonical historical K/V prefix lives on rank 0 as contiguous
   logical positions (when the prefix length is nonzero). The primitive
   stages that prefix, and each rank's new K and V, to *both* backends.
   The full logical K/V `[n_kv_heads*head_dim, cached+new_tokens]`
   must match byte-for-byte on each device.
3. For each destination rank, it reads that rank's local Q and full
   K/V back to the PC host, applies grouped-query causal attention
   using **absolute query positions** and a stable F64-accumulation
   softmax, then writes only that rank's local output back to its
   backend. Inputs can be F32 or F16.
4. The independent golden reference uses an unsharded causal mask and
   checks each output element, both K/V buffers, asymmetric splits,
   nonzero history, GQA, F16/F32 and invalid alias/shape settings.

This is intentionally a *slow correctness bridge*: K/V history is
retransmitted on every call, and Q/K/V are read back for host softmax.
It does not implement device-local fused kernels, real llama.cpp
paged KV slots, multi-sequence batching, asynchronous overlap or
fast cache replication. One-token decode remains TP.

First test locally, with no RPC server:

```powershell
git pull --ff-only origin feat/galaxy-baseline
cmake --build build-galaxy --config Release --target galaxy-causal-backend-selftest -j 8
.\build-galaxy\bin\Release\galaxy-causal-backend-selftest.exe
```

Then run the same test with the real PC+Phone backend:

```powershell
.\build-galaxy\bin\Release\galaxy-causal-backend-selftest.exe `
    --pc CUDA0 --rpc 10.39.52.114:50052
```

Expected final line: `GALAXY_CAUSAL_BACKEND_SELFTEST PASS mode=PC_RPC`.
The test runs a few *small deterministic tensors*, not an entire
Llama-3.2 model; it does not require a GGUF or recompiled Phone RPC
server. The companion mathematical reference
`galaxy-causal-attention-selftest` is still independent and should
also pass.

Next integration gate: before enabling SP Attention in Qwen3/Llama,
the Meta graph must support mirrored full Q/K/V projection weights,
rank-local Q/K/V sequence inputs, global absolute positions and
mask rows, and validated KV-cache physical-to-logical ownership.
Until then the existing TP/Galaxy/Galaxy2 graph is unchanged.


## Real GGUF QKV -> causal SP backend replay (graph-to-backend bridge)

After `GALAXY_CAUSAL_BACKEND_SELFTEST PASS mode=PC_RPC`, the next
integration gate captures **actual** Llama-3.2 3B graph activations and
replays them on the independent PC+Phone SP backend. This is a diagnostic
bridge, not an activated SP-Attention decoder.

The opt-in `-ProbeQkvLayers "0"` parameter captures four real tensors:

- `Qcur-0`: real post-RoPE query tensor
- `Kcur-0`: real post-RoPE key tensor
- `Vcur-0`: real projected value tensor
- `kqv_out-0`: original model's Attention output **before W_o**

The Meta backend reconstructs TP head shards from both ranks, in
canonical `[head_dim*heads, tokens]` order. It accepts only its
expected 3D head-sharded Q/K/V or 2D feature-sharded pre-Wo Attention
output, aborting the *diagnostic run* if shape/layout assumptions fail.
Each prefill token count is isolated under `real-qkv/tokens-N/` so
the warmup graph cannot overwrite the 98-token snapshot. No Q/K/V
capture is performed unless `-ProbeQkvLayers` is explicitly enabled.

The replay program reads the snapshots, splits each canonical tensor by
Token, and invokes the **real CUDA+RPC** host-staged causal Attention
executor. It asserts that both ranks received byte-exact K/V and prints
the RMSE, MaxAbs, cosine and worst token/channel compared with the
model's own `kqv_out`. Full attention arithmetic may differ because
the model uses Flash Attention and F16 cached K/V, whereas this replay
uses input F32 values and host F64 softmax; the replay's final PASS
means transport and execution succeeded, **not** numerical equivalence.

Build (only the Windows PC executable needs recompilation):

```powershell
git pull --ff-only origin feat/galaxy-baseline
cmake --build build-galaxy --config Release --target llama-cli galaxy-causal-model-replay -j 8
```

Run the real model in its existing pure-TP mode, with diagnostics only:

```powershell
$common.DumpLayers = ""
$common.DumpStages = ""
$common.DumpFfnOps = ""
$common.ProbeQkvLayers = "0"
$common.OutputDir = ".\galaxy-sp-results\real-qkv-replay"

.\tools\galaxy\run_galaxy_sp_smoke.ps1 @common -PureTP

$run = Get-ChildItem $common.OutputDir -Directory -Filter "tp-*" |
    Sort-Object CreationTime -Descending | Select-Object -First 1
$probe = Get-ChildItem (Join-Path $run.FullName "real-qkv") -Directory |
    Where-Object Name -match '^tokens-[0-9]+$' |
    Sort-Object @{Expression = { [int]($_.Name.Substring(7)) }; Descending = $true} |
    Select-Object -First 1
$tokens = [int]$probe.Name.Substring(7)
$rank0 = [int][math]::Ceiling($tokens / 2)
```

Run the independent cross-backend replay with the Llama 3.2 3B
24/8 GQA head configuration (128 channels per head):

```powershell
.\build-galaxy\bin\Release\galaxy-causal-model-replay.exe `
    --probe-dir $probe.FullName --layer 0 --tokens $tokens `
    --heads 24 --kv-heads 8 --head-dim 128 --rank0 $rank0 `
    --pc CUDA0 --rpc 10.39.52.114:50052
```

Expected end marker:
`GALAXY_CAUSAL_REAL_REPLAY PASS transport_verified numerical_parity_not_asserted`.
Inspect the preceding `[GALAXY_REAL_REPLAY]` lines to establish
whether actual GGUF Q/K/V agree with the standalone SP executor at the
pre-Wo boundary. If the probe logs `unexpected shape/type` or
`expected axis`, do not remove the layout guard blindly: capture the
actual Meta source states before extending the probe.

Restrictions: supported initial experiment is dense Llama without
Q/K RMSNorm, first prefill, a *logical* contiguous KV sequence with
no historic cache in replay, single sequence, and no positional bias
or soft-capping. This bypasses the model's *physical* KV cache on
purpose; it does not prove continuation/decode parity and does not
change the existing TP/Galaxy/Galaxy2 inference path. The final graph
integration remains separate.


## Real-model replay mismatch: investigate the source QKV snapshots first

Observed on CUDA0+Phone RPC, Llama-3.2-3B, first 98-token prompt,
Layer 0, TP model QKV -> SP host-staged replay:

- PC: RMSE=0.227125, MaxAbs=4.48022, Cosine=0.00028346
- Phone: RMSE=0.271659, MaxAbs=4.66108, Cosine=-0.00699348
- Combined: RMSE=0.250384, MaxAbs=4.66108, Cosine=-0.00397957
- K/V transport from the GLG1 snapshots to both ranks was byte-exact.

This is a **severe numerical mismatch**, not ordinary F16/Flash Attention
rounding; do not enable SP Attention in inference on this evidence.
The current replay's `PASS transport_verified numerical_parity_not_asserted`
is deliberately **not** an Attention correctness verdict.

`galaxy-real-qkv-sanity.py` now checks an independent invariant that
requires no GPU, RPC or model rerun: for the FIRST token of a fresh
single-sequence causal prefill, `kqv_out[0,query_head,:]` must equal
`Vcur[0,query_head/(n_q_heads/n_kv_heads),:]` (modulo cache-type
conversion). It prints group-vs-interleaved head-mapping metrics,
the best matching KV head for every Query head, and whether the first
Attention output more closely resembles V from another token.

Run against the **already exported** 98-token probe:

```powershell
git pull --ff-only origin feat/galaxy-baseline
python -m py_compile .\tools\galaxy\galaxy-real-qkv-sanity.py
$probe = "E:\llama\llama-galaxy\galaxy-sp-results\real-qkv-replay\tp-20261103-190442\real-qkv\tokens-98"
python .\tools\galaxy\galaxy-real-qkv-sanity.py `
    --probe-dir $probe --layer 0 --tokens 98 `
    --heads 24 --kv-heads 8 --head-dim 128
```

Interpretation:
- If token0 matches grouped V[0] closely, the export and initial
  Attention value layout pass a strong one-token sanity test.
  Continue by comparing the actual KV-cache mask/slot indices and
  the multi-key softmax execution.
- If it matches a *different* KV head or Token, inspect the Meta
  head/slot reconstruction before attempting a numerical fix.
- If it matches NO V candidate, first check that `kqv_out` and
  `Vcur` are sourced from the same graph and actual post-cache
  execution; inspect physical cache V, mask and output tensor
  lifetime. Do not infer that the transport primitive is incorrect.

`galaxy-causal-model-replay` also prints
`check=first_token_v_identity` and a severe mismatch warning after
rebuilding, but the Python check works immediately on existing files.


## TP Attention raw-vs-reshape duplication boundary (98-token Llama 3.2)

A later real-QKV experiment found that **every element** of the second
12-head half of the exported `kqv_out-0` was exactly equal to the
corresponding element in its first half, over all 98 tokens:
`exact=150528/150528`, RMSE=0, cosine=1.
The real projected `Qcur`, `Kcur`, and `Vcur` halves were *not*
repeated. The first-token causal V identity was also clearly violated
(grouped-GQA RMSE=0.0918303923, cosine=0.00513757887).

This is **not** numerical parity, and is incompatible with treating
the exported `kqv_out` as a trusted full-head Attention reference.
It does not yet prove whether the duplicate originated in the device
`FLASH_ATTN_EXT` output, its reshape/view, or the Meta exporter.

An opt-in five-tensor probe now preserves `fattn_raw-<layer>` just after
the `FLASH_ATTN_EXT` node (logical 4D `[v_head_dim,q_heads,tokens,1]`)
and before `ggml_reshape_2d` produces `kqv_out`. Both exports are
reassembled in logical `[heads*dim,tokens]` order. Meta additionally
logs `[GALAXY_REAL_QKV_RANK]` for each rank-local **raw** Flash Attention
and reshaped tensor, including backend, local dimensions, bytes and a
separate 64-bit FNV-1a hash of the bytes read from that backend.

The same `galaxy-real-qkv-sanity.py` now prints:

- `GALAXY_QKV_RAW_RESHAPE` for bitwise/raw-vs-2D equality and per-rank error
- `GALAXY_QKV_RAW_DUPLICATE` for whether *raw* Flash Attention halves repeat
- `GALAXY_QKV_DUPLICATE` for existing Q/K/V/output comparisons

This diagnostic change only runs with the existing `-ProbeQkvLayers`
option, does not change model Attention math or the Phone RPC protocol,
and may change allocation lifetimes when pinning the Flash Attention
node. Never use the timing from a probe run for performance claims.

One targeted rerun using the existing 98-token prompt:

```powershell
git pull --ff-only origin feat/galaxy-baseline
cmake --build build-galaxy --config Release --target llama-cli -j 8
$common.ProbeQkvLayers = "0"
$common.DumpLayers = ""
$common.DumpStages = ""
$common.DumpFfnOps = ""
$common.OutputDir = ".\galaxy-sp-results\raw-attn-boundary"
.\tools\galaxy\run_galaxy_sp_smoke.ps1 @common -PureTP

$run = Get-ChildItem $common.OutputDir -Directory -Filter "tp-*" |
    Sort-Object CreationTime -Descending | Select-Object -First 1
$probe = Join-Path $run.FullName "real-qkv\tokens-98"
python .\tools\galaxy\galaxy-real-qkv-sanity.py `
    --probe-dir $probe --layer 0 --tokens 98 `
    --heads 24 --kv-heads 8 --head-dim 128 |
    Select-String "GALAXY_QKV_RAW|GALAXY_QKV_DUPLICATE"
Select-String -Path (Join-Path $run.FullName "stderr.log") `
    -Pattern "\[GALAXY_REAL_QKV_RANK\]"
```

Interpretation:

- Raw halves **not duplicated**, but reshaped output duplicated:
  isolate the reshape/view/Meta per-rank source mapping. Check FNV hashes.
- Raw halves **already duplicated** and per-rank FNV hashes match:
  investigate Flash Attention execution or pre-reshape local tensor
  memory, rather than the 2D exporter.
- Raw and reshaped values do not match within the same rank:
  investigate tensor pinning/lifetimes and wrong/aliased buffer reads.
- All five snapshots become nonduplicated with the new pin:
  suspect temporary tensor lifetime/reuse previously masked by the
  unpinned raw intermediate. Verify again without the diagnostic
  before accepting a production fix.

Stop before enabling production SP-Attention until this boundary is
understood and the reference Attention is trustworthy.


## Post raw-Flash pinning: cross-run QKV capture stability

With `fattn_raw` pinned, a Llama-3.2 3B TP diagnostic run again
passed inference, and the new raw Attention and reshaped output are
**exactly identical** (`301056/301056`). Also, the prior exact
12-head duplication of `kqv_out` disappeared: the two halves now
differ (RMSE=0.0211013231; zero exact elements).

However the newly captured `Vcur` two 4-head halves are almost
identical (RMSE=0.0263049762, cosine=0.999928845) whereas the previous
run had RMSE=0.729279249, cosine=-0.0361658005. Since the main
change was adding the raw Attention diagnostic output pin, a
lifetime/aliasing or snapshot-timing problem is plausible. This is
**not** evidence to change GQA, RoPE or the production Attention mask.

Before another remote inference, compare the *existing* full probes:

```powershell
git pull --ff-only origin feat/galaxy-baseline

$old = "E:\llama\llama-galaxy\galaxy-sp-results\real-qkv-replay\tp-20261103-190442\real-qkv\tokens-98"
$new = "E:\llama\llama-galaxy\galaxy-sp-results\raw-attn-boundary\tp-20261103-204930\real-qkv\tokens-98"

python .\tools\galaxy\galaxy-real-qkv-sanity.py `
    --probe-dir $new --compare-probe-dir $old `
    --layer 0 --tokens 98 --heads 24 --kv-heads 8 --head-dim 128 |
    Select-String "GALAXY_QKV_CROSS_RUN|grouped_GQA|interleaved_GQA|tensor=Vcur|tensor=kqv_out"
```

This prints old-vs-new RMSE/MaxAbs/Cosine and exact element counts for
Qcur, Kcur, Vcur and kqv_out (whole graph, token 0, each head half).
If Q/K are bitwise identical but V/Attention are not, it narrows the
problem to the data captured after Q/K projection and validates the
hypothesis that diagnostic pinning changes snapshot observability.
It does not by itself identify a single invalid buffer lifetime.

A later probe run now emits `[GALAXY_REAL_QKV_RANK]` with explicit
`stderr` writes instead of verbosity-filtered INFO logs. Both
`stderr.log` and `stdout.log` can be searched for older runs, but
missing entries in an old log are not evidence of equal device hashes.


## Fix: capture QKV at the producing Meta subgraph (not graph completion)

The old/new 98-token diagnostic captures established that `Qcur`
and `Kcur` are **bitwise identical** across runs (0 differences), while
`Vcur` differs massively (cross-run RMSE=2.26481915, token-0
RMSE=3.77976845) and `kqv_out` also changes (RMSE=0.188672215).
Previous end-of-model snapshots were therefore **not trustworthy as a
numerical reference**. These measurements alone do not prove whether
the V computation or an overwritten temporary caused the change.

The Meta QKV probe has now been corrected to:

1. Treat `Qcur`, `Kcur`, `Vcur`, `fattn_raw`, and `kqv_out`
   (only when marked OUTPUT by the opt-in layer selector) as producer
   boundaries in the Meta execution partitioner.
2. When a boundary completes, synchronize both CUDA/RPC backends and
   immediately copy each rank's *actual local tensor bytes* to its own
   host-owned vector. Print one `[GALAXY_QKV_CAPTURE]` line per rank,
   including the producing subgraph and FNV-1a hash. No per-element
   logging or permanent changes to the runtime model buffers.
3. Skip TP AllReduce at the new diagnostic boundaries when their split
   state is a normal feature/head shard, not PARTIAL. Disable the
   pure-TP delayed-partial boundary optimization while probing, to
   avoid accidentally moving a requested capture point.
4. At the end of the run, reconstruct GLG1 snapshots **only from
   producer-time host copies**. If any required snapshot is missing or
   its length differs from the underlying rank-local shape, fail
   explicitly instead of silently reading a reused backend buffer.
5. Emit `[GALAXY_QKV_CAPTURE_SUM]` with `late_backend_reads=0`.
   Without `-ProbeQkvLayers` no diagnostic boundaries or readbacks
   are inserted.

Build only the PC client; Phone RPC is unchanged:

```powershell
git pull --ff-only origin feat/galaxy-baseline
cmake --build build-galaxy --config Release --target llama-cli -j 8

$common.DumpLayers = ""
$common.DumpStages = ""
$common.DumpFfnOps = ""
$common.ProbeQkvLayers = "0"
$common.OutputDir = ".\galaxy-sp-results\qkv-producer-capture"
.\tools\galaxy\run_galaxy_sp_smoke.ps1 @common -PureTP

$run = Get-ChildItem $common.OutputDir -Directory -Filter "tp-*" |
    Sort-Object CreationTime -Descending | Select-Object -First 1
$probe = Join-Path $run.FullName "real-qkv\tokens-98"

Get-Content (Join-Path $run.FullName "stderr.log") |
    Select-String "GALAXY_QKV_CAPTURE"
python .\tools\galaxy\galaxy-real-qkv-sanity.py `
    --probe-dir $probe --layer 0 --tokens 98 `
    --heads 24 --kv-heads 8 --head-dim 128 |
    Select-String "GALAXY_QKV_DUPLICATE|GALAXY_QKV_RAW|grouped_GQA|interleaved_GQA"
```

One diagnostic 98-token TP run is sufficient to check whether the
capture-time V/Attention outputs are stable and satisfy the
first-token invariant. These runs deliberately add backend fences and
additional subgraphs, so their `tok/s` must not be used in performance
comparisons. **This does not itself prove production Attention numerical
parity.** If the model still violates the causal token-0 V identity with
producer-time copies, proceed to the true Flash Attention mask, source
KV-cache placement and per-rank GQA indexing before enabling SP
Attention in the inference graph.


## Native CUDA + Phone Flash Attention replay: first real device-kernel SP stage

The producer-time Q/K/V correction is now numerically verified for
Llama-3.2-3B Layer 0, 98 fresh prefill tokens (split 49/49).
On CUDA0 + RPC0 the reference HOST-SOFTMAX replay produced:

- Rank 0 RMSE 6.18443e-6, MaxAbs 2.48551e-4
- Rank 1 RMSE 5.71072e-6, MaxAbs 1.9154e-4
- Overall RMSE 5.95229e-6, MaxAbs 2.48551e-4, Cosine approximately 1
- Both K/V transfers were byte-exact, but host softmax is not device-local execution.

The new galaxy-causal-native-flash-replay executable uses the same
already verified GLG1 activation snapshots, but performs real native
device Attention computation rather than the host softmax:

1. PC and Phone own distinct contiguous Query TOKEN intervals, with
   all query heads on each device, unlike existing TP head splitting.
   The input is the existing post-RoPE model Q snapshot.
2. F32 K/V are converted to F16, split by the same token intervals,
   then transferred with all_gather_attention_kv to BOTH backends.
   Each backend's gathered F16 K/V is checked byte for byte.
3. Each device's full K/V tensor has 256-token-padded cache capacity,
   matching the first model prefill cache convention. The populated
   prefix is gathered, the remaining padding is zero-filled and masked.
4. Each backend builds its own GGML FLASH_ATTN_EXT graph: Q is
   [dim,local_tokens,query_heads]; K/V use
   [dim,kv_tokens_padded,kv_heads]. The causal F16 mask uses GLOBAL
   Query positions. V follows the non-transposed F16 Flash cache path.
5. PC CUDA and Phone RPC independently run native Flash Attention.
   The combined outputs are compared to producer-time kqv_out-0.f32,
   reporting RMSE, MaxAbs, cosine, gather and kernel timings.

No new GGUF model run is required. Phone RPC needs no new server
build because the new executable uses existing standard GGML ops.

Windows build and run:

    git pull --ff-only origin feat/galaxy-baseline
    cmake --build build-galaxy --config Release --target galaxy-causal-native-flash-replay -j 8
    $probe = "E:\llama\llama-galaxy\galaxy-sp-results\qkv-producer-capture\tp-20261103-210405\real-qkv\tokens-98"
    .\build-galaxy\bin\Release\galaxy-causal-native-flash-replay.exe --probe-dir $probe --layer 0 --tokens 98 --heads 24 --kv-heads 8 --head-dim 128 --rank0 49 --pc CUDA0 --rpc 10.39.52.114:50052

For fail-closed numerical parity after observing backend differences,
optionally add --require-rmse 0.01. Without that flag, a final
PASS kernel_executed=1 production_graph_enabled=0 means K/V
transport, native graph execution and numerical MEASUREMENT completed,
not that numerical parity was asserted.

If RPC reports unsupported FLASH_ATTN_EXT, record the status and
retain the original host-softmax replay as the correctness baseline.
Never silently calculate the Phone output on CPU.

SCOPE: this is a native-kernel stage, NOT the full transformer graph.
Full GGUF QKV weight mirroring, rank-local QKV projection, absolute
RoPE positions, dynamic cache ownership, output/FFN transitions,
multi-sequence scheduling and decode remain unimplemented.

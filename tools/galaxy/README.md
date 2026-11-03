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

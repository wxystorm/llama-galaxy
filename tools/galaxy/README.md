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

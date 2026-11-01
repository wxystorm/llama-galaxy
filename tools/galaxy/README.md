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

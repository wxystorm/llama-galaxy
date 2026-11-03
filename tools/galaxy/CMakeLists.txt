add_executable(galaxy-tp-selftest galaxy-tp-selftest.cpp)
target_compile_features(galaxy-tp-selftest PRIVATE cxx_std_17)

# Standalone TP/SP collective correctness tests; no inference graph changes.
add_executable(galaxy-collectives-selftest galaxy-collectives-selftest.cpp)
target_link_libraries(galaxy-collectives-selftest PRIVATE ggml)
target_compile_features(galaxy-collectives-selftest PRIVATE cxx_std_17)
if (GGML_RPC)
    target_compile_definitions(galaxy-collectives-selftest PRIVATE GALAXY_COLLECTIVES_HAVE_RPC=1)
endif()

# Full-vocabulary numerical comparison: standalone C++17 executable.
add_executable(galaxy-logits-compare galaxy-logits-compare.cpp)
target_compile_features(galaxy-logits-compare PRIVATE cxx_std_17)

# Model-independent reference contract for the upstream Galaxy-LM BERT
# configuration modes: TP, Galaxy, Galaxy2, full SP and mixed variants.
add_executable(galaxy-source-plan-selftest galaxy-source-plan-selftest.cpp)
target_include_directories(galaxy-source-plan-selftest PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
target_compile_features(galaxy-source-plan-selftest PRIVATE cxx_std_17)

# Causal SP Attention reference: local Q, global gathered K/V, absolute
# RoPE positions, GQA head mapping and a nonzero historical KV prefix.
# This target does not activate the unfinished GPU/RPC SP Attention graph.
add_executable(galaxy-causal-attention-selftest galaxy-causal-attention-selftest.cpp)
target_compile_features(galaxy-causal-attention-selftest PRIVATE cxx_std_17)

# Real-backend correctness test for local causal Q, global KV with canonical
# cached prefix, and CPU-host staging. Can use CPU+CPU or CUDA+Phone RPC.
# Does not yet activate the Qwen3/Llama inference graph's SP Attention.
add_executable(galaxy-causal-backend-selftest galaxy-causal-backend-selftest.cpp)
target_link_libraries(galaxy-causal-backend-selftest PRIVATE ggml)
target_compile_features(galaxy-causal-backend-selftest PRIVATE cxx_std_17)
if (GGML_RPC)
    target_compile_definitions(galaxy-causal-backend-selftest PRIVATE GALAXY_CAUSAL_HAVE_RPC=1)
endif()

# Replays real GGUF Q/K/V exported by a TP Llama graph on CPU+CPU or
# CUDA+Phone RPC using the causal SP backend. Reports numerical error
# versus the model's pre-Wo kqv_out without claiming exact parity.
add_executable(galaxy-causal-model-replay galaxy-causal-model-replay.cpp)
target_link_libraries(galaxy-causal-model-replay PRIVATE ggml)
target_compile_features(galaxy-causal-model-replay PRIVATE cxx_std_17)
if (GGML_RPC)
    target_compile_definitions(galaxy-causal-model-replay PRIVATE GALAXY_CAUSAL_HAVE_RPC=1)
endif()

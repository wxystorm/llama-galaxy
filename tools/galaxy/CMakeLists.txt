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

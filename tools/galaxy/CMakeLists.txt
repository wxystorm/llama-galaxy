add_executable(galaxy-tp-selftest galaxy-tp-selftest.cpp)
target_compile_features(galaxy-tp-selftest PRIVATE cxx_std_17)

# Standalone TP/SP collective correctness tests; no inference graph changes.
add_executable(galaxy-collectives-selftest galaxy-collectives-selftest.cpp)
target_link_libraries(galaxy-collectives-selftest PRIVATE ggml)
target_compile_features(galaxy-collectives-selftest PRIVATE cxx_std_17)
if (GGML_RPC)
    target_compile_definitions(galaxy-collectives-selftest PRIVATE GALAXY_COLLECTIVES_HAVE_RPC=1)
endif()

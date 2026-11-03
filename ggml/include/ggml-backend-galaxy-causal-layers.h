#pragma once

// Shared, deliberately conservative layer-selection contract for the first
// production causal SP-Attention integration. Both llama-graph and GGML Meta
// read exactly the same environment and therefore pin/execute the same
// model layers. Never enable all layers or silently accept sparse sets.
//
// Rollout order: 1 -> 2 -> 4 consecutive leading layers. Later, once
// single-layer REPLACE has independently passed full-logits parity, enable
// the next prefix size explicitly. Unset by default.

#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_galaxy_causal_layers {

struct plan {
    std::vector<int> layers;
    std::string error;
    bool requested = false;

    bool valid() const { return error.empty(); }
    bool contains(int64_t layer) const {
        if (layer < 0 || !valid()) return false;
        for (const int l : layers) if (layer == l) return true;
        return false;
    }
    int match_boundary(const char * name, const char * stem) const {
        if (!name || !stem || !valid()) return -1;
        for (const int layer : layers) {
            if (std::string(name) == std::string(stem)+"-"+std::to_string(layer)) {
                return layer;
            }
        }
        return -1;
    }
    std::string description() const {
        std::string result;
        for (const int layer : layers) {
            if (!result.empty()) result += ",";
            result += std::to_string(layer);
        }
        return result;
    }
};

inline plan parse(const char * legacy_layer, const char * prefix_layers) {
    const bool legacy = legacy_layer && *legacy_layer;
    const bool prefix = prefix_layers && *prefix_layers;
    plan result;
    result.requested = legacy || prefix;
    if (legacy && prefix) {
        result.error="choose legacy LLAMA_GALAXY_CAUSAL_LIVE_LAYER or "
                     "LLAMA_GALAXY_CAUSAL_LIVE_PREFIX_LAYERS, not both";
        return result;
    }
    if (legacy) {
        if (std::string(legacy_layer)!="0") {
            result.error="legacy live causal layer supports only 0";
            return result;
        }
        result.layers={0};
        return result;
    }
    if (prefix) {
        const std::string count(prefix_layers);
        if (count=="1") result.layers={0};
        else if (count=="2") result.layers={0,1};
        else if (count=="4") result.layers={0,1,2,3};
        else {
            result.error="causal SP prefix must be 1, 2, or 4 leading layers";
        }
    }
    return result;
}

inline plan from_environment() {
    return parse(std::getenv("LLAMA_GALAXY_CAUSAL_LIVE_LAYER"),
                 std::getenv("LLAMA_GALAXY_CAUSAL_LIVE_PREFIX_LAYERS"));
}

inline bool enabled(int64_t layer) {
    return from_environment().contains(layer);
}

} // namespace ggml_galaxy_causal_layers

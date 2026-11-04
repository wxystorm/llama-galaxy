#pragma once

// Shared, deliberately conservative layer-selection contract for the first
// production causal SP-Attention integration. Both llama-graph and GGML Meta
// read exactly the same environment and therefore pin/execute the same
// model layers. Never enable all layers or silently accept sparse sets.
//
// Rollout order: 1 -> 2 -> 4 -> 8 -> 16 -> 28 consecutive leading
// layers (Llama-3.2-3B has 28 layers). This selection means execution
// coverage only: it does NOT assert TP/SP numerical equivalence. Any
// model with fewer than the selected prefix must fail the graph-wide
// per-layer completeness contract. Unset by default.

#include <cstdlib>
#include <cstddef>
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
        // Decimal only, exact supported prefixes, no sparse selection,
        // leading zeros or accidental "all" activation. These sizes
        // deliberately correspond to staged experiments on 28-layer Llama.
        const std::string count(prefix_layers);
        int n = 0;
        if (count=="1") n=1;
        else if (count=="2") n=2;
        else if (count=="4") n=4;
        else if (count=="8") n=8;
        else if (count=="16") n=16;
        else if (count=="28") n=28;
        if (n==0) {
            result.error="causal SP prefix must be one of 1,2,4,8,16,28 leading layers";
        } else {
            result.layers.reserve(static_cast<size_t>(n));
            for (int il=0;il<n;++il) result.layers.push_back(il);
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

#pragma once

// Galaxy-LM assigns fixed attention-head and FFN-feature counts to each rank.
// This file contains only static allocation policy, not collective transport.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend-galaxy-causal-layers.h"

namespace llama_galaxy_tp {

struct policy {
    bool active = false;
    std::vector<int64_t> heads;
    std::vector<int64_t> ffn;
};

inline bool enabled() {
    const char * value = std::getenv("LLAMA_GALAXY_TP");
    return value != nullptr && std::string(value) == "1";
}

// Experimental TP -> SP graph transitions, disabled by default.
inline bool sp_enabled() {
    const char * value = std::getenv("LLAMA_GALAXY_SP");
    return value != nullptr && std::string(value) == "1";
}

// Galaxy2: Attention TP -> RS -> SP connection + full-weight SP FFN -> AG.
// Full FFN replication may not fit on a 6 GiB device. For incremental
// validation, LLAMA_GALAXY2_FFN_SP_LAYERS can cap the number of leading
// prefill layers with complete FFN replicas. "all" selects every eligible
// layer. The last layer is excluded to preserve the output-token path.
inline bool galaxy2_enabled() {
    const char * value = std::getenv("LLAMA_GALAXY2");
    return value != nullptr && std::string(value) == "1";
}

// Default OFF: legacy optimization selects only the last token before
// the final FFN. Source Galaxy-LM restores its full sequence after CON2;
// enabling this keeps the final layer in SP and postpones output selection
// until after the last AllGather. Qwen3 decoding (one token) stays TP.
inline bool sp_include_final_layer() {
    const char * value = std::getenv("LLAMA_GALAXY_SP_FINAL_LAYER");
    return value != nullptr && std::string(value) == "1";
}

inline int64_t galaxy2_eligible_ffn_layers(int64_t n_layers, bool include_final) {
    if (n_layers < 1) {
        throw std::invalid_argument("Galaxy2 requires a positive layer count");
    }
    return n_layers - (include_final ? 0 : 1);
}

inline int64_t galaxy2_parse_ffn_layers(const char * raw, int64_t eligible) {
    if (eligible < 0) {
        throw std::invalid_argument("Galaxy2 requires a positive layer count");
    }
    if (raw == nullptr || std::string(raw) == "all") {
        return eligible;
    }
    if (raw[0] == '\0') {
        throw std::invalid_argument("LLAMA_GALAXY2_FFN_SP_LAYERS must be 'all' or a nonnegative integer");
    }
    int64_t value = 0;
    for (const char * p = raw; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            throw std::invalid_argument("LLAMA_GALAXY2_FFN_SP_LAYERS must be 'all' or a nonnegative integer");
        }
        const int digit = *p - '0';
        if (value > (std::numeric_limits<int64_t>::max() - digit) / 10) {
            throw std::invalid_argument("LLAMA_GALAXY2_FFN_SP_LAYERS overflows int64");
        }
        value = value*10 + digit;
    }
    if (value > eligible) {
        throw std::invalid_argument("Galaxy2 SP FFN count exceeds eligible non-final layers");
    }
    return value;
}

// A per-layer Galaxy2 FFN schedule is the common source of truth for
// graph construction AND weight placement. Explicit layer sets use zero-based
// indices or inclusive ranges, e.g. "0,3,6-8". "none" and "all" are allowed.
// The legacy leading-N FFN_SP_LAYERS setting remains backward compatible.
// Do not silently combine the two settings: inconsistent ownership would
// cause a per-rank tensor layout mismatch.
inline int64_t galaxy2_parse_layer_index(const std::string & spec, size_t & cursor) {
    if (cursor >= spec.size() || spec[cursor] < '0' || spec[cursor] > '9') {
        throw std::invalid_argument("Galaxy2 layer set requires an integer index");
    }
    int64_t value = 0;
    while (cursor < spec.size() && spec[cursor] >= '0' && spec[cursor] <= '9') {
        const int digit = spec[cursor] - '0';
        if (value > (std::numeric_limits<int64_t>::max() - digit)/10) {
            throw std::invalid_argument("Galaxy2 layer index overflows int64");
        }
        value = value * 10 + digit;
        ++cursor;
    }
    return value;
}

inline std::vector<uint8_t> galaxy2_parse_layer_set(const char * raw, int64_t eligible) {
    if (eligible < 0) {
        throw std::invalid_argument("Galaxy2 layer set requires nonnegative eligible layers");
    }
    if (raw == nullptr || *raw == '\0') {
        throw std::invalid_argument("Galaxy2 layer set must be 'all', 'none', or indices/ranges");
    }
    // An index that is out of range is always an error, including the
    // excluded last layer when FinalLayerSP is not enabled.
    std::vector<uint8_t> selected(static_cast<size_t>(eligible), uint8_t(0));
    const std::string spec(raw);
    if (spec == "none") {
        return selected;
    }
    if (spec == "all") {
        for (auto & flag : selected) {
            flag = 1;
        }
        return selected;
    }
    size_t cursor = 0;
    while (cursor < spec.size()) {
        const int64_t first = galaxy2_parse_layer_index(spec, cursor);
        int64_t last = first;
        if (cursor < spec.size() && spec[cursor] == '-') {
            ++cursor;
            last = galaxy2_parse_layer_index(spec, cursor);
            if (last < first) {
                throw std::invalid_argument("Galaxy2 layer set has a descending range");
            }
        }
        if (last >= eligible) {
            throw std::invalid_argument("Galaxy2 layer set selects a noneligible layer");
        }
        for (int64_t il = first; il <= last; ++il) {
            uint8_t & flag = selected[static_cast<size_t>(il)];
            if (flag) {
                throw std::invalid_argument("Galaxy2 layer set contains a duplicate layer");
            }
            flag = 1;
        }
        if (cursor == spec.size()) {
            break;
        }
        if (spec[cursor] != ',') {
            throw std::invalid_argument("Galaxy2 layer set must separate ranges with commas");
        }
        ++cursor;
        if (cursor == spec.size()) {
            throw std::invalid_argument("Galaxy2 layer set must not end with a comma");
        }
    }
    return selected;
}

// Pure resolver for testing and future planners. Both model loading and
// graph building consume exactly the same effective mask.
inline std::vector<uint8_t> galaxy2_resolve_ffn_sp_mask(
        const char * layer_set, const char * leading_count, int64_t eligible) {
    if (layer_set != nullptr) {
        if (leading_count != nullptr && *leading_count != '\0') {
            throw std::invalid_argument(
                    "Galaxy2 FFN_SP_LAYER_SET and FFN_SP_LAYERS cannot both be configured");
        }
        return galaxy2_parse_layer_set(layer_set, eligible);
    }
    const int64_t count = galaxy2_parse_ffn_layers(leading_count, eligible);
    std::vector<uint8_t> selected(static_cast<size_t>(eligible), uint8_t(0));
    for (int64_t il = 0; il < count; ++il) {
        selected[static_cast<size_t>(il)] = 1;
    }
    return selected;
}

inline std::vector<uint8_t> galaxy2_ffn_sp_mask(int64_t n_layers) {
    if (!galaxy2_enabled()) {
        return {};
    }
    const int64_t eligible = galaxy2_eligible_ffn_layers(n_layers, sp_include_final_layer());
    return galaxy2_resolve_ffn_sp_mask(
            std::getenv("LLAMA_GALAXY2_FFN_SP_LAYER_SET"),
            std::getenv("LLAMA_GALAXY2_FFN_SP_LAYERS"), eligible);
}

inline int64_t galaxy2_ffn_sp_count(int64_t n_layers) {
    int64_t count = 0;
    for (uint8_t flag : galaxy2_ffn_sp_mask(n_layers)) {
        count += flag ? 1 : 0;
    }
    return count;
}

inline bool galaxy2_ffn_is_sp(int64_t il, int64_t n_layers) {
    if (il < 0) {
        return false;
    }
    const auto mask = galaxy2_ffn_sp_mask(n_layers);
    return static_cast<size_t>(il) < mask.size() && mask[static_cast<size_t>(il)] != 0;
}

// Comma-delimited diagnostic layer selection, or 'all'. Keep the original
// layer dump and the opt-in intermediate-stage dump independently selectable.
inline bool layer_selection_contains(const char * value, int64_t il) {
    if (value == nullptr || *value == '\0' || il < 0) {
        return false;
    }
    const std::string spec(value);
    if (spec == "all") {
        return true;
    }
    const std::string needle = std::to_string(il);
    size_t start = 0;
    while (start < spec.size()) {
        const size_t end = spec.find(',', start);
        const size_t len = end == std::string::npos ? spec.size()-start : end-start;
        if (spec.substr(start, len) == needle) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end+1;
    }
    return false;
}

// Existing l_out snapshot: pin the selected final layer outputs.
inline bool dump_layer_enabled(int64_t il) {
    return layer_selection_contains(std::getenv("LLAMA_GALAXY_LAYER_DUMP"), il);
}

// Diagnostic-only snapshots of comparable logical intermediate activations.
// Only the dense Llama Galaxy TP/SP path currently opts into this facility.
inline bool dump_stage_enabled(int64_t il) {
    return layer_selection_contains(std::getenv("LLAMA_GALAXY_STAGE_DUMP"), il);
}

// Opt-in: capture the first dense-Llama FFN projection/activation stages.
// Disabled by default because the 8192-feature snapshots are relatively large.
inline bool dump_ffn_op_enabled(int64_t il) {
    return layer_selection_contains(std::getenv("LLAMA_GALAXY_FFN_OP_DUMP"), il);
}

 // Opt-in: reconstruct actual model Q, K, V and pre-output-projection
 // Attention results. This is a diagnostic bridge to the independent causal
 // SP backend, NOT a switch to SP inference. Never active unless requested.
inline bool qkv_probe_enabled(int64_t il) {
    return layer_selection_contains(std::getenv("LLAMA_GALAXY_QKV_PROBE_LAYERS"), il);
}

// Explicitly selected causal SP Attention layers. Both the SHADOW
// numerical validation and the REPLACE producer path use the same pinning
// contract as the Meta executor. The normal TP graph is unchanged unless
// the legacy layer=0 or new 1/2/4 prefix env selector is set.
inline bool causal_live_shadow_enabled(int64_t il) {
    return ggml_galaxy_causal_layers::enabled(il);
}

// Native SP QKV is an independently selected leading-layer prefix.
// The original TP KV-cache path is preserved until SP Decode is implemented.
inline bool native_sp_qkv_enabled(int64_t il) {
    return ggml_galaxy_causal_layers::native_qkv_enabled(il);
}

inline std::vector<int64_t> parse_counts(const char * text, const char * key, size_t n_devices) {
    if (text == nullptr || text[0] == '\0') {
        throw std::invalid_argument(std::string(key) + " is required in Galaxy TP mode");
    }
    std::vector<int64_t> counts;
    int64_t value = 0;
    bool have_digits = false;
    for (const char * p = text; ; ++p) {
        const char c = *p;
        if (c >= '0' && c <= '9') {
            const int digit = c - '0';
            if (value > (std::numeric_limits<int64_t>::max() - digit)/10) {
                throw std::invalid_argument(std::string(key) + " contains an overflowing count");
            }
            value = 10*value + digit;
            have_digits = true;
        } else if (c == ',' || c == '\0') {
            if (!have_digits || value <= 0) {
                throw std::invalid_argument(std::string(key) + " needs positive comma-separated counts");
            }
            counts.push_back(value);
            value = 0;
            have_digits = false;
            if (c == '\0') {
                break;
            }
        } else {
            throw std::invalid_argument(std::string(key) + " needs positive comma-separated counts");
        }
    }
    if (counts.size() != n_devices) {
        throw std::invalid_argument(std::string(key) + " count must match number of devices");
    }
    return counts;
}

inline policy from_environment(size_t n_devices) {
    policy result;
    result.active = enabled();
    if (!result.active) {
        return result;
    }
    if (n_devices != 2) {
        throw std::invalid_argument("Galaxy TP phase 1 requires exactly two devices");
    }
    result.heads = parse_counts(std::getenv("LLAMA_GALAXY_TP_HEADS"), "LLAMA_GALAXY_TP_HEADS", n_devices);
    result.ffn = parse_counts(std::getenv("LLAMA_GALAXY_TP_FFN"), "LLAMA_GALAXY_TP_FFN", n_devices);
    return result;
}

inline void validate_counts(const std::vector<int64_t> & counts, int64_t total, int64_t granularity, const char * name) {
    if (counts.empty() || total <= 0 || granularity <= 0) {
        throw std::invalid_argument(std::string("invalid Galaxy TP ") + name + " dimensions");
    }
    int64_t sum = 0;
    for (int64_t count : counts) {
        if (count <= 0 || count % granularity != 0 || count > total - sum) {
            throw std::invalid_argument(std::string("invalid Galaxy TP ") + name + " split or granularity");
        }
        sum += count;
    }
    if (sum != total) {
        throw std::invalid_argument(std::string("Galaxy TP ") + name + " split does not sum to model dimension");
    }
}

} // namespace llama_galaxy_tp

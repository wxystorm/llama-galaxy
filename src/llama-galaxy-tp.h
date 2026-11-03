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

inline int64_t galaxy2_ffn_sp_count(int64_t n_layers) {
    if (!galaxy2_enabled()) {
        return 0;
    }
    return galaxy2_parse_ffn_layers(std::getenv("LLAMA_GALAXY2_FFN_SP_LAYERS"),
            galaxy2_eligible_ffn_layers(n_layers, sp_include_final_layer()));
}

inline bool galaxy2_ffn_is_sp(int64_t il, int64_t n_layers) {
    return il >= 0 && il < galaxy2_ffn_sp_count(n_layers);
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

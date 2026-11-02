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

inline int64_t galaxy2_ffn_sp_count(int64_t n_layers) {
    if (!galaxy2_enabled()) {
        return 0;
    }
    if (n_layers < 1) {
        throw std::invalid_argument("Galaxy2 requires a positive layer count");
    }
    const int64_t eligible = n_layers - 1;
    const char * raw = std::getenv("LLAMA_GALAXY2_FFN_SP_LAYERS");
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

inline bool galaxy2_ffn_is_sp(int64_t il, int64_t n_layers) {
    return il >= 0 && il < galaxy2_ffn_sp_count(n_layers);
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

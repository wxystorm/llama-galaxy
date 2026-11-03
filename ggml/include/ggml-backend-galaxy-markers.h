#pragma once

// Galaxy TP/SP graph-only marker identity. Unlike a tensor name, ggml DUP
// op_params are not changed by llama_context::graph_get_cb(), which may
// rename an output tensor (e.g. GALAXY_AG_FFN_0 -> l_out-0).
//
// The Meta backend consumes these DUP nodes itself; they are never executed
// as ordinary DUP kernels. The metadata is carried on the original ggml
// tensor and copied to per-device descriptors during graph preparation.

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace ggml_galaxy {

enum class sp_transition : int32_t {
    none             = 0,
    reduce_scatter   = 1,
    scatter_mirrored = 2,
    all_gather       = 3,
};

constexpr int32_t marker_magic = 0x47535031; // ASCII "GSP1"

// Source Galaxy-LM offers per-rank seq_scatter_list such as [20,12].
// llama.cpp builds multiple graph sizes (including two-token warmups),
// so the two positive weights are scaled to each graph's live token count.
// When unset, 1:1 reproduces the previous split.
// Minimum one token per rank; tie goes to rank 0.
inline std::array<int64_t, 2> parse_seq_weights(const char * raw) {
    if (raw == nullptr) return {1, 1};
    const std::string value(raw);
    if (value.empty()) {
        throw std::invalid_argument("LLAMA_GALAXY_SP_SEQ_WEIGHTS needs two positive weights");
    }
    std::array<int64_t, 2> result = {0, 0};
    size_t component = 0;
    bool have_digit = false;
    for (const char c : value) {
        if (c >= '0' && c <= '9') {
            const int64_t digit = c-'0';
            if (result[component] > (std::numeric_limits<int64_t>::max()-digit)/10) {
                throw std::invalid_argument("LLAMA_GALAXY_SP_SEQ_WEIGHTS integer overflow");
            }
            result[component] = result[component]*10 + digit;
            have_digit = true;
        } else if (c == ',' && component == 0 && have_digit && result[0] > 0) {
            component = 1;
            have_digit = false;
        } else {
            throw std::invalid_argument("LLAMA_GALAXY_SP_SEQ_WEIGHTS must be positive a,b");
        }
    }
    if (component != 1 || !have_digit || result[1] <= 0) {
        throw std::invalid_argument("LLAMA_GALAXY_SP_SEQ_WEIGHTS must be positive a,b");
    }
    return result;
}

inline std::array<int64_t, 2> split_sequence(int64_t tokens, std::array<int64_t, 2> weights) {
    if (tokens < 2 || weights[0] <= 0 || weights[1] <= 0) {
        throw std::invalid_argument("Galaxy SP needs >=2 tokens and positive rank weights");
    }
    // Long double avoids int64 overflow from N*weight; MSVC uses double
    // here, which is sufficient for the prompt lengths and weighting ratios.
    const long double ratio = static_cast<long double>(weights[0]) /
            (static_cast<long double>(weights[0]) + static_cast<long double>(weights[1]));
    const long double ideal = static_cast<long double>(tokens) * ratio + 0.5L;
    const int64_t first = std::max<int64_t>(1, std::min<int64_t>(
            tokens - 1, static_cast<int64_t>(ideal)));
    return {first, tokens - first};
}

inline std::array<int64_t, 2> configured_sequence_split(int64_t tokens) {
    return split_sequence(tokens, parse_seq_weights(std::getenv("LLAMA_GALAXY_SP_SEQ_WEIGHTS")));
}


inline void mark_transition(ggml_tensor * tensor, sp_transition transition) {
    if (tensor == nullptr || tensor->op != GGML_OP_DUP || transition == sp_transition::none) {
        throw std::invalid_argument("Galaxy SP marker requires a DUP tensor and a nonzero transition");
    }
    tensor->op_params[0] = marker_magic;
    tensor->op_params[1] = static_cast<int32_t>(transition);
}

inline sp_transition get_transition(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->op != GGML_OP_DUP || tensor->op_params[0] != marker_magic) {
        return sp_transition::none;
    }
    switch (tensor->op_params[1]) {
        case 1: return sp_transition::reduce_scatter;
        case 2: return sp_transition::scatter_mirrored;
        case 3: return sp_transition::all_gather;
        default: return sp_transition::none;
    }
}

} // namespace ggml_galaxy

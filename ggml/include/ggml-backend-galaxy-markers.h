#pragma once

// Galaxy TP/SP graph-only marker identity. Unlike a tensor name, ggml DUP
// op_params are not changed by llama_context::graph_get_cb(), which may
// rename an output tensor (e.g. GALAXY_AG_FFN_0 -> l_out-0).
//
// The Meta backend consumes these DUP nodes itself; they are never executed
// as ordinary DUP kernels. The metadata is carried on the original ggml
// tensor and copied to per-device descriptors during graph preparation.

#include "ggml.h"

#include <cstdint>
#include <stdexcept>

namespace ggml_galaxy {

enum class sp_transition : int32_t {
    none             = 0,
    reduce_scatter   = 1,
    scatter_mirrored = 2,
    all_gather       = 3,
};

constexpr int32_t marker_magic = 0x47535031; // ASCII "GSP1"

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

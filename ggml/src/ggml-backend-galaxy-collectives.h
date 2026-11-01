#pragma once

// Experimental, correctness-first Galaxy TP/SP communication primitives.
// Implemented using ggml backend tensor_get/set (CPU/CUDA/RPC compatible).
// These functions are NOT yet integrated with the Meta Device graph.
//
// GGML contiguous 2-D layout: [hidden, tokens], rank 0 first in sequence.
// Each rank provides its own backend-allocated tensors.

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ggml_galaxy_collectives {

using devices_t = std::array<ggml_backend_t, 2>;
using input_t = std::array<const ggml_tensor *, 2>;
using output_t = std::array<ggml_tensor *, 2>;
using counts_t = std::array<int64_t, 2>;

namespace detail {

inline void demand(bool condition, const std::string & text) {
    if (!condition) {
        throw std::invalid_argument("Galaxy collective: " + text);
    }
}

inline size_t element_bytes(ggml_type type) {
    demand(type == GGML_TYPE_F32 || type == GGML_TYPE_F16, "only F32/F16 are supported");
    return type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
}

inline size_t multiply(int64_t count, size_t stride) {
    demand(count >= 0 && stride != 0 &&
           static_cast<uint64_t>(count) <= std::numeric_limits<size_t>::max()/stride,
           "byte offset/size overflow");
    return static_cast<size_t>(count)*stride;
}

inline int64_t total_tokens(counts_t counts) {
    demand(counts[0] > 0 && counts[1] > 0,
           "both ranks need tokens (single-token decode must use TP)");
    demand(counts[0] <= std::numeric_limits<int64_t>::max()-counts[1],
           "sequence count overflow");
    return counts[0]+counts[1];
}

inline void tensor_shape(const ggml_tensor * tensor, ggml_type type,
                         int64_t hidden, int64_t tokens, const char * role) {
    demand(tensor != nullptr && tensor->buffer && tensor->data,
           std::string(role) + ": tensor is missing or unallocated");
    demand(tensor->type == type && tensor->ne[0] == hidden && tensor->ne[1] == tokens &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1,
           std::string(role) + ": expected [hidden,tokens] with matching dtype");
    demand(ggml_is_contiguous(tensor), std::string(role) + ": must be contiguous");
    const size_t row = multiply(hidden, element_bytes(type));
    demand(multiply(tokens, row) == ggml_nbytes(tensor),
           std::string(role) + ": invalid dense byte layout");
}

inline void validate(const devices_t & devices, const input_t & in, const output_t & out,
                     counts_t counts, size_t tile_tokens, bool scatter) {
    demand(devices[0] && devices[1], "both backends must exist");
    demand(tile_tokens > 0 && tile_tokens <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
           "tile_tokens must be in 1..INT64_MAX");
    const int64_t total = total_tokens(counts);
    demand(in[0] && in[1], "both input tensors must exist");
    const ggml_type type = in[0]->type;
    element_bytes(type);
    const int64_t hidden = in[0]->ne[0];
    demand(hidden > 0, "hidden dimension must be positive");
    for (int i = 0; i < 2; ++i) {
        tensor_shape(in[i], type, hidden, scatter ? total : counts[i], "input");
        tensor_shape(out[i], type, hidden, scatter ? counts[i] : total, "output");
        for (int j = 0; j < 2; ++j) {
            demand(in[i] != out[j], "input/output aliasing is unsupported");
        }
    }
}

inline float read(ggml_type type, const uint8_t * ptr) {
    if (type == GGML_TYPE_F32) {
        float v;
        std::memcpy(&v, ptr, sizeof(v));
        return v;
    }
    ggml_fp16_t h;
    std::memcpy(&h, ptr, sizeof(h));
    return ggml_fp16_to_fp32(h);
}

inline void write(ggml_type type, uint8_t * ptr, float value) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(ptr, &value, sizeof(value));
    } else {
        const ggml_fp16_t h = ggml_fp32_to_fp16(value);
        std::memcpy(ptr, &h, sizeof(h));
    }
}

} // namespace detail

// Input rank tensors each contain a partial contribution for ALL tokens.
// Output rank tensors contain only the reduced tokens assigned to that rank.
// This implementation only creates each destination's reduced slice, not an
// AllReduce(full) intermediate. Host staging is intentionally not optimized.
inline void reduce_scatter(const devices_t & devices, const input_t & partial,
                           const output_t & shard, counts_t counts, size_t tile_tokens = 32) {
    detail::validate(devices, partial, shard, counts, tile_tokens, true);
    const ggml_type type = partial[0]->type;
    const size_t item = detail::element_bytes(type);
    const size_t row = detail::multiply(partial[0]->ne[0], item);

    ggml_backend_synchronize(devices[0]);
    ggml_backend_synchronize(devices[1]);

    int64_t global_start = 0;
    for (int rank = 0; rank < 2; ++rank) {
        for (int64_t done = 0; done < counts[rank]; ) {
            const int64_t n = std::min<int64_t>(counts[rank]-done, static_cast<int64_t>(tile_tokens));
            const size_t bytes = detail::multiply(n, row);
            const size_t src_offset = detail::multiply(global_start+done, row);
            const size_t dst_offset = detail::multiply(done, row);
            std::vector<uint8_t> a(bytes), b(bytes), reduced(bytes);
            ggml_backend_tensor_get(partial[0], a.data(), src_offset, bytes);
            ggml_backend_tensor_get(partial[1], b.data(), src_offset, bytes);
            for (size_t k = 0; k < bytes/item; ++k) {
                detail::write(type, reduced.data()+k*item,
                        detail::read(type, a.data()+k*item) +
                        detail::read(type, b.data()+k*item));
            }
            ggml_backend_tensor_set(shard[rank], reduced.data(), dst_offset, bytes);
            ggml_backend_synchronize(devices[rank]);
            done += n;
        }
        global_start += counts[rank];
    }
}

// Input rank tensors each hold their own reduced sequence interval.
// Output rank tensors each receive the full original sequence in rank order.
// Unequal splits are handled without padding.
inline void all_gather(const devices_t & devices, const input_t & shard,
                       const output_t & replicated, counts_t counts, size_t tile_tokens = 32) {
    detail::validate(devices, shard, replicated, counts, tile_tokens, false);
    const size_t row = detail::multiply(shard[0]->ne[0], detail::element_bytes(shard[0]->type));

    ggml_backend_synchronize(devices[0]);
    ggml_backend_synchronize(devices[1]);

    int64_t global_start = 0;
    for (int rank = 0; rank < 2; ++rank) {
        for (int64_t done = 0; done < counts[rank]; ) {
            const int64_t n = std::min<int64_t>(counts[rank]-done, static_cast<int64_t>(tile_tokens));
            const size_t bytes = detail::multiply(n, row);
            const size_t src_offset = detail::multiply(done, row);
            const size_t dst_offset = detail::multiply(global_start+done, row);
            std::vector<uint8_t> data(bytes);
            ggml_backend_tensor_get(shard[rank], data.data(), src_offset, bytes);
            for (int dest = 0; dest < 2; ++dest) {
                ggml_backend_tensor_set(replicated[dest], data.data(), dst_offset, bytes);
                // The host buffer is reused after the backend write completes.
                ggml_backend_synchronize(devices[dest]);
            }
            done += n;
        }
        global_start += counts[rank];
    }
}

// Copy the sequence interval owned by each rank out of an already mirrored
// full activation. This is needed for the residual path before an SP addition.
// Rank 0's mirror is the canonical source. No arithmetic takes place.
// For the initial implementation, host staging is used for both backends.
inline void scatter_mirrored(const devices_t & devices, const input_t & full,
                            const output_t & shard, counts_t counts, size_t tile_tokens = 32) {
    detail::validate(devices, full, shard, counts, tile_tokens, true);
    const size_t row = detail::multiply(full[0]->ne[0], detail::element_bytes(full[0]->type));

    ggml_backend_synchronize(devices[0]);
    ggml_backend_synchronize(devices[1]);

    int64_t global_start = 0;
    for (int rank = 0; rank < 2; ++rank) {
        for (int64_t done = 0; done < counts[rank]; ) {
            const int64_t n = std::min<int64_t>(counts[rank]-done, static_cast<int64_t>(tile_tokens));
            const size_t bytes = detail::multiply(n, row);
            const size_t src_offset = detail::multiply(global_start+done, row);
            const size_t dst_offset = detail::multiply(done, row);
            std::vector<uint8_t> data(bytes);
            ggml_backend_tensor_get(full[0], data.data(), src_offset, bytes);
            ggml_backend_tensor_set(shard[rank], data.data(), dst_offset, bytes);
            ggml_backend_synchronize(devices[rank]);
            done += n;
        }
        global_start += counts[rank];
    }
}

} // namespace ggml_galaxy_collectives

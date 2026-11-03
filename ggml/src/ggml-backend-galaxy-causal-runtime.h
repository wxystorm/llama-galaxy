#pragma once

// Experimental PC+Phone causal sequence-parallel Attention execution primitive.
//
// This is a correctness-first HOST-STAGED backend operation, NOT yet wired
// into the Qwen3/Llama transformer graph. Unlike a host-only reference,
// it reads local Q/K/V and a canonical historical K/V prefix from real
// ggml backends (including RPC), gathers current K/V onto BOTH backends,
// then computes each rank's local causal Query outputs from the complete
// K/V tensor stored on that rank.
//
// Inputs must already include absolute-position RoPE and have contiguous
// [heads*head_dim, tokens] F32/F16 storage. The historical cache is a
// canonical contiguous prefix on rank 0. This simple contract deliberately
// rejects multi-sequence/paged cache and decode; it does NOT reinterpret
// llama.cpp's physical KV-cache indices as linear positions.
//
// Readback, host softmax, and backend writes are intentionally slow. Runtime
// inference must not use this until the graph can produce local Q and
// cache-correct K/V (including mask and rotary coordinates).

#include "ggml-backend-galaxy-causal-plan.h"
#include "ggml-backend-galaxy-collectives.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ggml_galaxy_causal {

using backend_pair = ggml_galaxy_collectives::devices_t;
using input_pair   = ggml_galaxy_collectives::input_t;
using output_pair  = ggml_galaxy_collectives::output_t;

namespace runtime_detail {

inline void require(bool condition, const char * message) {
    if (!condition) {
        throw std::invalid_argument(std::string("Galaxy causal backend: ") + message);
    }
}

inline int64_t checked_width(int64_t heads, int64_t head_dim) {
    require(heads > 0 && head_dim > 0 &&
            heads <= std::numeric_limits<int64_t>::max()/head_dim,
            "invalid head width");
    return heads*head_dim;
}

inline void validate_tensor(const ggml_tensor * tensor, ggml_type dtype,
        int64_t width, int64_t tokens, const char * role) {
    require(tensor != nullptr && tensor->buffer != nullptr && tensor->data != nullptr,
            "unallocated input/output tensor");
    require(tensor->type == dtype && tensor->ne[0] == width &&
            tensor->ne[1] == tokens && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
            ggml_is_contiguous(tensor),
            "tensor dtype, shape, or strides mismatch");
    const size_t item = ggml_galaxy_collectives::detail::element_bytes(dtype);
    const size_t expected = ggml_galaxy_collectives::detail::multiply(
            tokens, ggml_galaxy_collectives::detail::multiply(width, item));
    require(ggml_nbytes(tensor) == expected, "tensor byte count mismatch");
    (void) role;
}

inline void validate_execution(const backend_pair & devices, const layout & shape,
        const input_pair & local_q, const input_pair & new_k, const input_pair & new_v,
        const ggml_tensor * cached_k, const ggml_tensor * cached_v,
        const output_pair & complete_k, const output_pair & complete_v,
        const output_pair & local_output, size_t tile_tokens) {
    validate(shape);
    require(runtime_prefill_eligible(shape),
            "only multi-token prefill with a nonempty Query shard per rank is supported");
    require(devices[0] != nullptr && devices[1] != nullptr, "two backends required");
    require(tile_tokens >= 1 && tile_tokens <= 1024, "tile_tokens must be 1..1024");

    const int64_t q_width = checked_width(shape.query_heads, shape.head_dim);
    const int64_t kv_width = checked_width(shape.kv_heads, shape.head_dim);
    const int64_t n_kv = total_kv(shape);
    require(local_q[0] && local_q[1], "local Query buffers required");
    const ggml_type type = local_q[0]->type;
    require(type == GGML_TYPE_F32 || type == GGML_TYPE_F16,
            "only contiguous F32/F16 are supported");

    std::vector<const ggml_tensor *> inputs;
    std::vector<const ggml_tensor *> outputs;
    for (int rank = 0; rank < 2; ++rank) {
        const auto query = rank_queries(shape, rank);
        validate_tensor(local_q[rank], type, q_width, query.count, "local Query");
        validate_tensor(new_k[rank], type, kv_width, query.count, "local new Key");
        validate_tensor(new_v[rank], type, kv_width, query.count, "local new Value");
        validate_tensor(complete_k[rank], type, kv_width, n_kv, "complete Key");
        validate_tensor(complete_v[rank], type, kv_width, n_kv, "complete Value");
        validate_tensor(local_output[rank], type, q_width, query.count, "local output");
        for (const auto * ptr : {local_q[rank], new_k[rank], new_v[rank]}) {
            inputs.push_back(ptr);
        }
        for (const auto * ptr : {complete_k[rank], complete_v[rank], local_output[rank]}) {
            outputs.push_back(ptr);
        }
    }

    if (shape.cached_tokens > 0) {
        validate_tensor(cached_k, type, kv_width, shape.cached_tokens, "canonical cached Key");
        validate_tensor(cached_v, type, kv_width, shape.cached_tokens, "canonical cached Value");
        inputs.push_back(cached_k);
        inputs.push_back(cached_v);
    } else {
        require(cached_k == nullptr && cached_v == nullptr,
                "zero-history case must not include cached tensors");
    }

    // Avoid output clobbering, including cross-kind K/V aliasing. Two
    // distinct backend tensors must never share a writable destination.
    for (size_t i = 0; i < outputs.size(); ++i) {
        for (size_t j = i+1; j < outputs.size(); ++j) {
            require(outputs[i] != outputs[j], "writable output tensors alias");
        }
        for (const auto * input : inputs) {
            require(outputs[i] != input, "input/output alias is unsupported");
        }
    }
}

inline void copy_shard_to_both(const backend_pair & devices,
        const ggml_tensor * src, const output_pair & dst,
        int64_t source_tokens, int64_t absolute_dst_start,
        size_t row_bytes, size_t tile_tokens) {
    const int64_t max_tile = std::min<int64_t>(source_tokens, static_cast<int64_t>(tile_tokens));
    if (max_tile == 0) {
        return;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(max_tile)*row_bytes);
    for (int64_t offset = 0; offset < source_tokens; ) {
        const int64_t n = std::min<int64_t>(source_tokens-offset,
                static_cast<int64_t>(tile_tokens));
        const size_t length = static_cast<size_t>(n)*row_bytes;
        ggml_backend_tensor_get(src, bytes.data(),
                static_cast<size_t>(offset)*row_bytes, length);
        for (int rank = 0; rank < 2; ++rank) {
            ggml_backend_tensor_set(dst[rank], bytes.data(),
                    static_cast<size_t>(absolute_dst_start+offset)*row_bytes, length);
            // The staging buffer may be reused only after the destination
            // consumes this tile, including an asynchronous CUDA backend.
            ggml_backend_synchronize(devices[rank]);
        }
        offset += n;
    }
}

inline std::vector<uint8_t> read_complete(const ggml_tensor * tensor) {
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    return bytes;
}

inline double at(const std::vector<uint8_t> & values,
        ggml_type dtype, size_t idx) {
    const size_t item = ggml_galaxy_collectives::detail::element_bytes(dtype);
    return static_cast<double>(ggml_galaxy_collectives::detail::read(
            dtype, values.data() + idx*item));
}

} // namespace runtime_detail

// Gather both K and V, including canonical cached history. "complete_*"
// is a complete *logical* KV tensor on each backend after this call.
// New tokens are source-sharded, historical tokens are read from rank0.
// This is a correctness fallback, not a fast KV-cache replication protocol.
inline void gather_kv_with_history(const backend_pair & devices, const layout & shape,
        const input_pair & new_k, const input_pair & new_v,
        const ggml_tensor * cached_k, const ggml_tensor * cached_v,
        const output_pair & complete_k, const output_pair & complete_v,
        size_t tile_tokens = 32) {
    validate(shape);
    runtime_detail::require(devices[0] && devices[1], "two backends required");
    runtime_detail::require(tile_tokens > 0 && tile_tokens <= 1024,
            "tile_tokens must be 1..1024");
    runtime_detail::require(new_k[0] && new_k[1] && new_v[0] && new_v[1],
            "both ranks must provide new K/V");
    const ggml_type type = new_k[0]->type;
    runtime_detail::require(type == GGML_TYPE_F16 || type == GGML_TYPE_F32,
            "only F16/F32 are supported");
    const int64_t width = runtime_detail::checked_width(shape.kv_heads, shape.head_dim);
    for (int rank = 0; rank < 2; ++rank) {
        const auto interval_ = rank_queries(shape, rank);
        runtime_detail::validate_tensor(new_k[rank], type, width, interval_.count, "new Key");
        runtime_detail::validate_tensor(new_v[rank], type, width, interval_.count, "new Value");
        runtime_detail::validate_tensor(complete_k[rank], type, width, total_kv(shape), "full Key");
        runtime_detail::validate_tensor(complete_v[rank], type, width, total_kv(shape), "full Value");
        for (auto * output : {complete_k[rank], complete_v[rank]}) {
            for (const auto * input : {new_k[0], new_k[1], new_v[0], new_v[1],
                    cached_k, cached_v}) {
                runtime_detail::require(output != input, "K/V output aliases input");
            }
        }
    }
    runtime_detail::require(complete_k[0] != complete_v[0] &&
            complete_k[1] != complete_v[1] &&
            complete_k[0] != complete_k[1] &&
            complete_v[0] != complete_v[1] &&
            complete_k[0] != complete_v[1] &&
            complete_k[1] != complete_v[0], "K/V outputs alias");
    if (shape.cached_tokens > 0) {
        runtime_detail::validate_tensor(cached_k, type, width,
                shape.cached_tokens, "cached Key");
        runtime_detail::validate_tensor(cached_v, type, width,
                shape.cached_tokens, "cached Value");
    } else {
        runtime_detail::require(cached_k == nullptr && cached_v == nullptr,
                "zero-history case must have no cache tensors");
    }

    ggml_backend_synchronize(devices[0]);
    ggml_backend_synchronize(devices[1]);
    const size_t row = ggml_galaxy_collectives::detail::multiply(
            width, ggml_galaxy_collectives::detail::element_bytes(type));
    if (shape.cached_tokens) {
        runtime_detail::copy_shard_to_both(devices, cached_k, complete_k,
                shape.cached_tokens, 0, row, tile_tokens);
        runtime_detail::copy_shard_to_both(devices, cached_v, complete_v,
                shape.cached_tokens, 0, row, tile_tokens);
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto interval_ = rank_queries(shape, rank);
        const int64_t begin = shape.cached_tokens + interval_.start;
        runtime_detail::copy_shard_to_both(devices, new_k[rank], complete_k,
                interval_.count, begin, row, tile_tokens);
        runtime_detail::copy_shard_to_both(devices, new_v[rank], complete_v,
                interval_.count, begin, row, tile_tokens);
    }
}

// Execute local Query shards with the gathered full global K/V now resident
// on EACH destination backend. Host softmax is deliberately unoptimized;
// this is a CPU computation reference over actual device buffers.
inline void host_staged_attention(const backend_pair & devices, const layout & shape,
        const input_pair & local_q, const input_pair & new_k, const input_pair & new_v,
        const ggml_tensor * cached_k, const ggml_tensor * cached_v,
        const output_pair & complete_k, const output_pair & complete_v,
        const output_pair & local_output, size_t tile_tokens = 32) {
    runtime_detail::validate_execution(devices, shape, local_q, new_k, new_v,
            cached_k, cached_v, complete_k, complete_v, local_output, tile_tokens);
    gather_kv_with_history(devices, shape, new_k, new_v,
            cached_k, cached_v, complete_k, complete_v, tile_tokens);

    const int64_t q_width = runtime_detail::checked_width(shape.query_heads, shape.head_dim);
    const int64_t kv_width = runtime_detail::checked_width(shape.kv_heads, shape.head_dim);
    const int64_t n_kv = total_kv(shape);
    const ggml_type dtype = local_q[0]->type;
    const size_t item = ggml_galaxy_collectives::detail::element_bytes(dtype);
    const double scale = 1.0/std::sqrt(static_cast<double>(shape.head_dim));

    for (int rank = 0; rank < 2; ++rank) {
        ggml_backend_synchronize(devices[rank]);
        const auto queries = runtime_detail::read_complete(local_q[rank]);
        const auto all_keys = runtime_detail::read_complete(complete_k[rank]);
        const auto all_values = runtime_detail::read_complete(complete_v[rank]);
        const auto interval_ = rank_queries(shape, rank);
        std::vector<uint8_t> out(static_cast<size_t>(interval_.count*q_width)*item);
        std::vector<double> scores(static_cast<size_t>(n_kv));

        for (int64_t t = 0; t < interval_.count; ++t) {
            const int64_t absolute_pos = absolute_query_position(shape, rank, t);
            for (int64_t h = 0; h < shape.query_heads; ++h) {
                const int64_t kvh = kv_head_for_query(shape, h);
                double max_score = -std::numeric_limits<double>::infinity();
                for (int64_t k = 0; k < n_kv; ++k) {
                    // Query/K projections are already globally RoPE-rotated.
                    double score = -std::numeric_limits<double>::infinity();
                    if (visible(shape, absolute_pos, k)) {
                        score = 0.0;
                        for (int64_t d = 0; d < shape.head_dim; ++d) {
                            const size_t qi = static_cast<size_t>(t*q_width+h*shape.head_dim+d);
                            const size_t ki = static_cast<size_t>(k*kv_width+kvh*shape.head_dim+d);
                            score += runtime_detail::at(queries, dtype, qi) *
                                     runtime_detail::at(all_keys, dtype, ki);
                        }
                        score *= scale;
                        max_score = std::max(max_score, score);
                    }
                    scores[static_cast<size_t>(k)] = score;
                }
                runtime_detail::require(std::isfinite(max_score),
                        "Query has no visible causal key");
                double denominator = 0.0;
                for (double score : scores) {
                    if (std::isfinite(score)) {
                        denominator += std::exp(score-max_score);
                    }
                }
                for (int64_t d = 0; d < shape.head_dim; ++d) {
                    double value = 0.0;
                    for (int64_t k = 0; k < n_kv; ++k) {
                        const double score = scores[static_cast<size_t>(k)];
                        if (std::isfinite(score)) {
                            const size_t vi = static_cast<size_t>(k*kv_width+kvh*shape.head_dim+d);
                            value += std::exp(score-max_score)/denominator *
                                     runtime_detail::at(all_values, dtype, vi);
                        }
                    }
                    const size_t oi = static_cast<size_t>(t*q_width+h*shape.head_dim+d);
                    ggml_galaxy_collectives::detail::write(
                            dtype, out.data()+oi*item, static_cast<float>(value));
                }
            }
        }
        ggml_backend_tensor_set(local_output[rank], out.data(), 0, out.size());
        ggml_backend_synchronize(devices[rank]);
    }
}

} // namespace ggml_galaxy_causal

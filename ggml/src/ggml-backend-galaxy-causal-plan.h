#pragma once

// Causal sequence-parallel attention ownership contract for Galaxy-LM.
//
// This is a CPU-testable description of query/KV positions, not a runtime
// operator. Do not route Qwen3 inference to SP Attention until the Meta
// graph correctly connects full K/V, positions, masks and the KV cache.
//
// Q ownership: contiguous local query tokens; rank 0 precedes rank 1.
// KV visibility: each rank must see all prior cached tokens AND the current
// full sequence of K/V, even when it computes only local Q tokens.
// GQA mapping: query head q reads KV head q / (n_q_heads / n_kv_heads).
// Causal condition: key_absolute_position <= query_absolute_position.

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ggml_galaxy_causal {

struct layout {
    int64_t cached_tokens  = 0;   // historical KV entries already present
    int64_t new_tokens     = 0;   // current prefill chunk
    int64_t rank0_queries  = 0;   // rank 1 owns the remainder
    int64_t query_heads    = 0;
    int64_t kv_heads       = 0;
    int64_t head_dim       = 0;
    int64_t sequences      = 1;   // Phase 1 supports one logical sequence
};

struct interval {
    int64_t start = 0;  // index relative to the new-token chunk
    int64_t count = 0;
};

inline void require(bool ok, const char * message) {
    if (!ok) {
        throw std::invalid_argument(message);
    }
}

inline void validate(const layout & l) {
    require(l.cached_tokens >= 0, "Galaxy causal SP: cached_tokens must be >= 0");
    require(l.new_tokens > 0 && l.new_tokens <= std::numeric_limits<int64_t>::max() - l.cached_tokens,
            "Galaxy causal SP: invalid new_tokens or KV extent overflow");
    require(l.rank0_queries >= 0 && l.rank0_queries <= l.new_tokens,
            "Galaxy causal SP: rank0 query count out of range");
    require(l.sequences == 1, "Galaxy causal SP: multi-sequence cache layout unsupported");
    require(l.query_heads > 0 && l.kv_heads > 0 && l.head_dim > 0,
            "Galaxy causal SP: positive query_heads/kv_heads/head_dim required");
    require(l.query_heads % l.kv_heads == 0,
            "Galaxy causal SP: query_heads must be divisible by kv_heads");
}

inline int64_t total_kv(const layout & l) {
    validate(l);
    return l.cached_tokens + l.new_tokens;
}

inline interval rank_queries(const layout & l, int rank) {
    validate(l);
    require(rank == 0 || rank == 1, "Galaxy causal SP: rank must be 0 or 1");
    return rank == 0 ? interval{0, l.rank0_queries}
                     : interval{l.rank0_queries, l.new_tokens - l.rank0_queries};
}

inline int64_t absolute_query_position(const layout & l, int rank, int64_t local_q) {
    const interval interval_ = rank_queries(l, rank);
    require(local_q >= 0 && local_q < interval_.count,
            "Galaxy causal SP: local query index is out of range");
    return l.cached_tokens + interval_.start + local_q;
}

inline int64_t kv_head_for_query(const layout & l, int64_t query_head) {
    validate(l);
    require(query_head >= 0 && query_head < l.query_heads,
            "Galaxy causal SP: query head index is out of range");
    return query_head / (l.query_heads / l.kv_heads);
}

inline bool visible(const layout & l, int64_t query_absolute_pos, int64_t key_absolute_pos) {
    const int64_t n = total_kv(l);
    require(query_absolute_pos >= l.cached_tokens && query_absolute_pos < n,
            "Galaxy causal SP: query position out of current chunk");
    require(key_absolute_pos >= 0 && key_absolute_pos < n,
            "Galaxy causal SP: key position out of KV cache extent");
    return key_absolute_pos <= query_absolute_pos;
}

// Existing runtime falls back to tensor-parallel Attention for one-token
// decode. A local-query, globally visible-KV SP kernel will need an explicit
// cache-ownership protocol before this policy can be used in the live graph.
inline bool runtime_prefill_eligible(const layout & l) {
    validate(l);
    return l.new_tokens > 1 && l.rank0_queries > 0 && l.rank0_queries < l.new_tokens;
}

} // namespace ggml_galaxy_causal

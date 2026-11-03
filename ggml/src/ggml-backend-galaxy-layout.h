#pragma once

// Galaxy TP <-> SP activation layout exchange (two ranks).
//
// TP: rank i owns a contiguous range of COMPLETE heads, for ALL tokens.
// SP: rank i owns a contiguous range of tokens, with ALL heads.
// The in-memory flattened order is [head_dim * heads, tokens], where each
// token's feature row holds contiguous heads. Either 2D [features,tokens] or
// contiguous 3D [head_dim,heads,tokens] GGML tensors may be used.
//
// This is a correctness-first, tiled host staging collective. It never
// performs arithmetic, never converts F16/F32 and never invokes
// ggml_backend_tensor_memset (which RPC does not implement). In particular,
// a rank's TP K/V head group must not be mistaken for a local SP token shard.
// Caller owns head/group alignment, cache ordering and mask semantics.

#include "ggml-backend-galaxy-collectives.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace ggml_galaxy_layout {

using ggml_galaxy_collectives::devices_t;
using ggml_galaxy_collectives::input_t;
using ggml_galaxy_collectives::output_t;
using ggml_galaxy_collectives::counts_t;
namespace base = ggml_galaxy_collectives::detail;

struct exchange {
    counts_t head_counts;     // TP ownership in units of HEADS
    counts_t token_counts;    // SP ownership in units of TOKENS
    int64_t head_dim = 0;
};

inline void validate(const exchange & l, size_t tile_tokens) {
    base::demand(l.head_counts[0] > 0 && l.head_counts[1] > 0,
                 "TP->SP requires nonempty head ownership on both ranks");
    base::demand(l.token_counts[0] > 0 && l.token_counts[1] > 0,
                 "TP->SP requires nonempty token ownership on both ranks");
    base::demand(l.head_dim > 0, "TP->SP requires positive head_dim");
    base::demand(l.head_counts[0] <= INT64_MAX - l.head_counts[1] &&
                 l.token_counts[0] <= INT64_MAX - l.token_counts[1],
                 "TP->SP head/token extent overflow");
    base::demand((l.head_counts[0] + l.head_counts[1]) <= INT64_MAX/l.head_dim,
                 "TP->SP flattened feature width overflow");
    base::demand(tile_tokens > 0 && tile_tokens <= static_cast<size_t>(INT64_MAX),
                 "TP->SP tile_tokens must be positive");
}

inline int64_t total_heads(const exchange & l) {
    validate(l, 1);
    return l.head_counts[0] + l.head_counts[1];
}

inline int64_t total_tokens(const exchange & l) {
    validate(l, 1);
    return l.token_counts[0] + l.token_counts[1];
}

inline void check_tensor(const ggml_tensor * tensor, ggml_type type,
                         int64_t heads, int64_t dim, int64_t tokens,
                         const char * role) {
    base::demand(tensor != nullptr && tensor->buffer && tensor->data,
                 std::string(role) + ": missing backend allocation");
    base::demand(tensor->type == type && ggml_is_contiguous(tensor),
                 std::string(role) + ": expected contiguous F16/F32 tensor");
    const int64_t width = heads*dim; // validated to fit int64 by caller
    const bool flat = tensor->ne[0] == width && tensor->ne[1] == tokens &&
                      tensor->ne[2] == 1 && tensor->ne[3] == 1;
    const bool headed = tensor->ne[0] == dim && tensor->ne[1] == heads &&
                        tensor->ne[2] == tokens && tensor->ne[3] == 1;
    base::demand(flat || headed,
                 std::string(role) + ": requires [head_features,tokens] or [dim,heads,tokens]");
    base::demand(ggml_nbytes(tensor) ==
                 base::multiply(tokens, base::multiply(width, base::element_bytes(type))),
                 std::string(role) + ": unexpected padded/non-dense byte extent");
}

inline void check(const devices_t & backends, const input_t & source,
                  const output_t & dest, const exchange & l,
                  size_t tile_tokens, bool tp_to_sp) {
    validate(l, tile_tokens);
    base::demand(backends[0] && backends[1], "TP<->SP requires two backends");
    base::demand(source[0] && source[1] && dest[0] && dest[1],
                 "TP<->SP requires all 4 tensors");
    const auto type = source[0]->type;
    (void) base::element_bytes(type);
    const int64_t heads = total_heads(l), tokens = total_tokens(l);
    for (int rank = 0; rank < 2; ++rank) {
        if (tp_to_sp) {
            check_tensor(source[rank], type, l.head_counts[rank], l.head_dim, tokens, "TP source");
            check_tensor(dest[rank], type, heads, l.head_dim, l.token_counts[rank], "SP destination");
        } else {
            check_tensor(source[rank], type, heads, l.head_dim, l.token_counts[rank], "SP source");
            check_tensor(dest[rank], type, l.head_counts[rank], l.head_dim, tokens, "TP destination");
        }
        for (int d = 0; d < 2; ++d) {
            base::demand(source[rank] != dest[d], "TP<->SP input/output alias unsupported");
        }
    }
}

// Complete heads on two TP ranks -> complete features for each rank's
// respective token interval. All 4 logical TP->SP quadrants are copied.
// The inputs contain ALL tokens, so each device contributes to BOTH
// destinations; the output has no replicated token rows.
inline void head_to_token(const devices_t & backends,
                          const input_t & source_tp,
                          const output_t & dest_sp,
                          const exchange & l, size_t tile_tokens = 32) {
    check(backends, source_tp, dest_sp, l, tile_tokens, true);
    ggml_galaxy_collectives::detail::measure perf;
    const size_t item = base::element_bytes(source_tp[0]->type);
    const size_t rows[2] = {
        base::multiply(l.head_counts[0]*l.head_dim, item),
        base::multiply(l.head_counts[1]*l.head_dim, item),
    };
    const size_t full_row = rows[0] + rows[1];
    const int64_t capacity = std::min<int64_t>(
            std::max(l.token_counts[0],l.token_counts[1]),
            static_cast<int64_t>(tile_tokens));
    std::vector<uint8_t> rank0(base::multiply(capacity,rows[0]));
    std::vector<uint8_t> rank1(base::multiply(capacity,rows[1]));
    std::vector<uint8_t> packed(base::multiply(capacity,full_row));
    perf.sync(backends[0]);
    perf.sync(backends[1]);

    int64_t global_begin = 0;
    for (int dst = 0; dst < 2; ++dst) {
        for (int64_t done = 0; done < l.token_counts[dst]; ) {
            const int64_t n = std::min<int64_t>(l.token_counts[dst]-done,
                                               static_cast<int64_t>(tile_tokens));
            const int64_t start = global_begin + done;
            perf.read(source_tp[0], rank0.data(), base::multiply(start,rows[0]),
                      base::multiply(n,rows[0]));
            perf.read(source_tp[1], rank1.data(), base::multiply(start,rows[1]),
                      base::multiply(n,rows[1]));
            const auto began = ggml_galaxy_profile::clock::now();
            for (int64_t t = 0; t < n; ++t) {
                uint8_t * row = packed.data() + static_cast<size_t>(t)*full_row;
                std::memcpy(row,rank0.data()+static_cast<size_t>(t)*rows[0],rows[0]);
                std::memcpy(row+rows[0],rank1.data()+static_cast<size_t>(t)*rows[1],rows[1]);
            }
            if (perf.on) perf.host_ms += ggml_galaxy_profile::ms(began,ggml_galaxy_profile::clock::now());
            perf.write(dest_sp[dst],packed.data(),base::multiply(done,full_row),
                       base::multiply(n,full_row));
            perf.sync(backends[dst]);
            done += n;
        }
        global_begin += l.token_counts[dst];
    }
    perf.finish("HEAD_TO_TOKEN",dest_sp[0],total_tokens(l),tile_tokens);
}

// Complete features on SP token ranks -> original TP head ownership on
// each rank for ALL tokens. A token tile is read once, then the two
// non-overlapping head slices are dispatched to their TP destinations.
// Useful before an existing head-sharded W_o output projection.
inline void token_to_head(const devices_t & backends,
                          const input_t & source_sp,
                          const output_t & dest_tp,
                          const exchange & l, size_t tile_tokens = 32) {
    check(backends, source_sp, dest_tp, l, tile_tokens, false);
    ggml_galaxy_collectives::detail::measure perf;
    const size_t item = base::element_bytes(source_sp[0]->type);
    const size_t rows[2] = {
        base::multiply(l.head_counts[0]*l.head_dim, item),
        base::multiply(l.head_counts[1]*l.head_dim, item),
    };
    const size_t full_row = rows[0] + rows[1];
    const int64_t capacity = std::min<int64_t>(
            std::max(l.token_counts[0],l.token_counts[1]),
            static_cast<int64_t>(tile_tokens));
    std::vector<uint8_t> packed(base::multiply(capacity,full_row));
    std::vector<uint8_t> head0(base::multiply(capacity,rows[0]));
    std::vector<uint8_t> head1(base::multiply(capacity,rows[1]));
    perf.sync(backends[0]);
    perf.sync(backends[1]);

    int64_t global_begin = 0;
    for (int src = 0; src < 2; ++src) {
        for (int64_t done = 0; done < l.token_counts[src]; ) {
            const int64_t n = std::min<int64_t>(l.token_counts[src]-done,
                                               static_cast<int64_t>(tile_tokens));
            perf.read(source_sp[src],packed.data(),base::multiply(done,full_row),
                      base::multiply(n,full_row));
            const auto began = ggml_galaxy_profile::clock::now();
            for (int64_t t = 0; t < n; ++t) {
                const uint8_t * row = packed.data() + static_cast<size_t>(t)*full_row;
                std::memcpy(head0.data()+static_cast<size_t>(t)*rows[0],row,rows[0]);
                std::memcpy(head1.data()+static_cast<size_t>(t)*rows[1],row+rows[0],rows[1]);
            }
            if (perf.on) perf.host_ms += ggml_galaxy_profile::ms(began,ggml_galaxy_profile::clock::now());
            const int64_t offset = global_begin+done;
            perf.write(dest_tp[0],head0.data(),base::multiply(offset,rows[0]),
                       base::multiply(n,rows[0]));
            perf.sync(backends[0]);
            perf.write(dest_tp[1],head1.data(),base::multiply(offset,rows[1]),
                       base::multiply(n,rows[1]));
            perf.sync(backends[1]);
            done += n;
        }
        global_begin += l.token_counts[src];
    }
    perf.finish("TOKEN_TO_HEAD",dest_tp[0],total_tokens(l),tile_tokens);
}

} // namespace ggml_galaxy_layout

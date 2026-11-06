#pragma once

// Galaxy TP/SP collectives integrated with the Meta backend. Tiled Host
// Staging remains the correctness baseline. Increasing tile size reduces
// CPU/CUDA/RPC calls without changing arithmetic or output ordering.
//
// GGML contiguous 2-D layout: [hidden, tokens], rank 0 first in sequence.
// Each rank provides its own backend-allocated tensors.

#include "ggml-backend.h"
#include "ggml-backend-galaxy-profile.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <memory>
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

// Timers are populated only when LLAMA_GALAXY_PROFILE_DIR is set.
// Network latency is INCLUDED in backend tensor_get/set for RPC; separate
// phone-side compute timings require the optional RPC server profiler.
struct measure {
    bool on = ggml_galaxy_profile::enabled();
    ggml_galaxy_profile::instant begin = ggml_galaxy_profile::clock::now();
    double read_ms = 0;
    double write_ms = 0;
    double host_ms = 0;
    double sync_ms = 0;
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;
    uint64_t read_calls = 0;
    uint64_t write_calls = 0;
    uint64_t sync_calls = 0;

    void sync(ggml_backend_t backend) {
        const auto t = ggml_galaxy_profile::clock::now();
        ggml_backend_synchronize(backend);
        if (on) {
            sync_ms += ggml_galaxy_profile::ms(t, ggml_galaxy_profile::clock::now());
            sync_calls++;
        }
    }
    void read(const ggml_tensor * tensor, void * buffer, size_t offset, size_t bytes) {
        const auto t = ggml_galaxy_profile::clock::now();
        ggml_backend_tensor_get(tensor, buffer, offset, bytes);
        if (on) {
            read_ms += ggml_galaxy_profile::ms(t, ggml_galaxy_profile::clock::now());
            read_bytes += bytes;
            read_calls++;
        }
    }
    void write(ggml_tensor * tensor, const void * buffer, size_t offset, size_t bytes) {
        const auto t = ggml_galaxy_profile::clock::now();
        ggml_backend_tensor_set(tensor, buffer, offset, bytes);
        if (on) {
            write_ms += ggml_galaxy_profile::ms(t, ggml_galaxy_profile::clock::now());
            write_bytes += bytes;
            write_calls++;
        }
    }
    void finish(const char * kind, const ggml_tensor * tensor, int64_t tokens,
                size_t tile_tokens, bool rs_fused_read = false) const {
        if (!on) return;
        ggml_galaxy_profile::communication(kind, tensor ? tensor->name : "", tokens,
                read_bytes, write_bytes, read_ms, host_ms, write_ms,
                sync_ms, ggml_galaxy_profile::ms(begin, ggml_galaxy_profile::clock::now()),
                read_calls, write_calls, sync_calls, tile_tokens, rs_fused_read);
    }
};

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

// Original Galaxy-LM implements a two-rank ReduceScatter with each rank
// receiving ONLY its own peer contribution and doing the ADD locally.
// This implementation uses existing ggml RPC graph-compute for the local
// ADD/COPY, with get/set transporting the peer slice through the PC as a
// byte relay. It does not require new RPC commands or emulate Gloo P2P.
// HOST remains the baseline and the default. Only standard Galaxy/Galaxy2
// opt into RANK_LOCAL; the native SP/causal Hybrid path is unchanged.
enum class transport { host, rank_local };

inline transport parse_transport(const char * raw) {
    if (!raw || !raw[0] || std::strcmp(raw,"HOST")==0) return transport::host;
    if (std::strcmp(raw,"RANK_LOCAL")==0) return transport::rank_local;
    throw std::invalid_argument(
            "LLAMA_GALAXY_COLLECTIVE_TRANSPORT must be HOST or RANK_LOCAL");
}
inline transport configured_transport() {
    return parse_transport(std::getenv("LLAMA_GALAXY_COLLECTIVE_TRANSPORT"));
}

namespace detail {

// Scope-local leaf: the data and backend buffer are BORROWED from an existing
// real physical tensor. Do not use the graph producer as a source: it would
// re-evaluate the original Transformer subgraph during the collective.
inline ggml_tensor * borrowed_slice(ggml_context * ctx,
                                    const ggml_tensor * physical,
                                    int64_t tokens, size_t offset) {
    ggml_tensor * alias=ggml_new_tensor_2d(ctx,physical->type,physical->ne[0],tokens);
    demand(alias!=nullptr && physical->buffer && physical->data,
           "rank-local slice has no physical buffer");
    demand(offset <= ggml_nbytes(physical) &&
           ggml_nbytes(alias) <= ggml_nbytes(physical)-offset,
           "rank-local slice outside original allocation");
    alias->buffer=physical->buffer;
    alias->data=(char *)physical->data+offset;
    return alias;
}

struct transient_ctx {
    ggml_context * ctx=nullptr;
    ggml_backend_buffer_t scratch=nullptr;
    transient_ctx() {
        const ggml_init_params params={1024*1024,nullptr,true};
        ctx=ggml_init(params);
        demand(ctx!=nullptr,"cannot initialize rank-local collective graph");
    }
    ~transient_ctx() {
        if(scratch) ggml_backend_buffer_free(scratch);
        if(ctx) ggml_free(ctx);
    }
    transient_ctx(const transient_ctx &)=delete;
    transient_ctx & operator=(const transient_ctx &)=delete;
    void execute(ggml_backend_t backend, ggml_tensor * terminal) {
        ggml_cgraph * graph=ggml_new_graph_custom(ctx,16,false);
        demand(graph!=nullptr,"cannot create rank-local collective graph");
        ggml_build_forward_expand(graph,terminal);
        const ggml_status st=ggml_backend_graph_compute(backend,graph);
        demand(st==GGML_STATUS_SUCCESS,"rank-local device graph failed");
        ggml_backend_synchronize(backend);
    }
};

// The local slice has already been calculated by the TP producer.
// Receive only the opposite rank's contribution into a temporary tensor
// ON this rank, and compute local + received on that rank's backend.
struct rank_local_sum {
    transient_ctx lifetime;
    ggml_tensor * peer=nullptr;
    ggml_tensor * output=nullptr;
    ggml_backend_t backend=nullptr;

    rank_local_sum(ggml_backend_t device, const ggml_tensor * local_full,
                   ggml_tensor * destination, int64_t tokens, size_t offset)
                   : backend(device) {
        ggml_context * ctx=lifetime.ctx;
        auto * owned=borrowed_slice(ctx,local_full,tokens,offset);
        peer=ggml_new_tensor_2d(ctx,local_full->type,local_full->ne[0],tokens);
        auto * dest=borrowed_slice(ctx,destination,tokens,0);
        auto * sum=ggml_add(ctx,owned,peer);
        output=ggml_cpy(ctx,sum,dest);
        // Allocates peer and sum, but does NOT allocate/overwrite either
        // borrowed producer nor the original output physical tensor.
        lifetime.scratch=ggml_backend_alloc_ctx_tensors(ctx,backend);
        demand(lifetime.scratch!=nullptr,
               "rank-local reduction cannot allocate backend scratch");
        demand(output->buffer && output->data && peer->buffer && peer->data,
               "rank-local reduction tensors are not backend allocated");
    }
    void compute() { lifetime.execute(backend,output); }
};

// Device-local COPY for AG's own shard and mirrored residual scatter.
// COPY is executed by CUDA/RPC/CPU on the same backend; no backend GET/SET
// through the PC host is involved for the local portion.
inline void device_copy_slice(ggml_backend_t backend,
                              const ggml_tensor * source, size_t source_offset,
                              ggml_tensor * destination, size_t destination_offset,
                              int64_t tokens) {
    transient_ctx scope;
    ggml_context * ctx=scope.ctx;
    auto * src=borrowed_slice(ctx,source,tokens,source_offset);
    auto * dst=borrowed_slice(ctx,destination,tokens,destination_offset);
    auto * copy=ggml_cpy(ctx,src,dst);
    const ggml_status view_status=ggml_backend_view_init(copy);
    demand(view_status==GGML_STATUS_SUCCESS,
           "device-local copy output view could not be initialized");
    scope.execute(backend,copy);
}

inline void report_rank_local(const char * kind, int64_t tokens,
                              size_t cross_bytes, size_t local_bytes) {
    std::fprintf(stderr,
            "[GALAXY_COLLECTIVE_ROUTE] kind=%s mode=RANK_LOCAL tokens=%" PRId64
            " cross_payload_bytes=%zu local_device_bytes=%zu"
            " transport=RPC_RELAY compute=DEVICE_GRAPH status=PASS\n",
            kind,tokens,cross_bytes,local_bytes);
    std::fflush(stderr);
}

} // namespace detail

// Default 32 matches the previous implementation; larger tiles reduce RPC
// request count. Limit to 1024 tokens to bound temporary Host allocations.
inline size_t parse_tile_tokens(const char * raw) {
    if (raw == nullptr || raw[0] == '\0') {
        return 32;
    }
    size_t value = 0;
    for (const char * p = raw; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            throw std::invalid_argument("LLAMA_GALAXY_COLLECTIVE_TILE_TOKENS must be 1..1024");
        }
        const size_t digit = static_cast<size_t>(*p - '0');
        if (value > (size_t(1024) - digit)/10) {
            throw std::invalid_argument("LLAMA_GALAXY_COLLECTIVE_TILE_TOKENS exceeds 1024");
        }
        value = value*10 + digit;
    }
    if (value == 0 || value > 1024) {
        throw std::invalid_argument("LLAMA_GALAXY_COLLECTIVE_TILE_TOKENS must be 1..1024");
    }
    return value;
}

inline size_t configured_tile_tokens() {
    return parse_tile_tokens(std::getenv("LLAMA_GALAXY_COLLECTIVE_TILE_TOKENS"));
}

// Optional 2-read ReduceScatter strategy. Keep the original per-slice
// tensor_get path as the default for A/B and memory-constrained runs.
inline bool parse_rs_fused_read(const char * raw) {
    if (raw == nullptr || raw[0] == '\0' || std::strcmp(raw, "0") == 0) return false;
    if (std::strcmp(raw, "1") == 0) return true;
    throw std::invalid_argument("LLAMA_GALAXY_RS_FUSED_READ must be 0 or 1");
}

inline bool configured_rs_fused_read() {
    return parse_rs_fused_read(std::getenv("LLAMA_GALAXY_RS_FUSED_READ"));
}

// Maximum size of EACH full partial input. Fused mode holds two sources
// (32 MiB in total at this limit), plus at most one output tile.
// Larger activations transparently use the legacy tiled read path.
inline constexpr size_t rs_fused_max_input_bytes = size_t(16) * 1024 * 1024;

inline bool rs_fused_read_eligible(counts_t counts, size_t row_bytes) {
    const int64_t total = detail::total_tokens(counts);
    return row_bytes > 0 &&
           static_cast<uint64_t>(total) <= rs_fused_max_input_bytes / row_bytes;
}

// Input rank tensors each contain a partial contribution for ALL tokens.
// Output rank tensors contain only the reduced tokens assigned to that rank.
// This implementation creates only each destination's reduced slice.
// Optional fused input read transfers each rank's full Partial once into
// bounded PC Host memory; reductions and output writes preserve token order.
inline void reduce_scatter(const devices_t & devices, const input_t & partial,
                           const output_t & shard, counts_t counts, size_t tile_tokens = 32,
                           bool request_fused_read = false,
                           transport route = configured_transport()) {
    detail::validate(devices, partial, shard, counts, tile_tokens, true);
    if (route == transport::rank_local) {
        detail::demand(!request_fused_read,
                "RANK_LOCAL ReduceScatter cannot combine with HOST RS fused read");
        detail::measure measure;
        const size_t row=detail::multiply(partial[0]->ne[0],
                                         detail::element_bytes(partial[0]->type));
        const int64_t total=detail::total_tokens(counts);
        const int64_t capacity=std::min<int64_t>(std::max(counts[0],counts[1]),
                                                static_cast<int64_t>(tile_tokens));
        std::vector<uint8_t> relay(detail::multiply(capacity,row));
        measure.sync(devices[0]);
        measure.sync(devices[1]);
        int64_t start=0;
        for(int rank=0;rank<2;++rank) {
            // One ring exchange step for two ranks: receive the opposite
            // partial contribution for THIS rank's token interval.
            detail::rank_local_sum reducer(devices[rank],partial[rank],
                                            shard[rank],counts[rank],
                                            detail::multiply(start,row));
            for(int64_t done=0;done<counts[rank];) {
                const int64_t n=std::min<int64_t>(counts[rank]-done,
                                                  static_cast<int64_t>(tile_tokens));
                const size_t bytes=detail::multiply(n,row);
                measure.read(partial[1-rank],relay.data(),
                             detail::multiply(start+done,row),bytes);
                measure.write(reducer.peer,relay.data(),
                              detail::multiply(done,row),bytes);
                measure.sync(devices[rank]);
                done+=n;
            }
            reducer.compute();
            start+=counts[rank];
        }
        measure.finish("RS",shard[0],total,tile_tokens);
        detail::report_rank_local("RS",total,
                detail::multiply(total,row),detail::multiply(total,row));
        return;
    }
    detail::measure measure;
    const ggml_type type = partial[0]->type;
    const size_t item = detail::element_bytes(type);
    const size_t row = detail::multiply(partial[0]->ne[0], item);

    measure.sync(devices[0]);
    measure.sync(devices[1]);

    // One reusable Host allocation per collective, capped at tile_tokens.
    // backend_tensor_set is synchronous; retain the completion fence before
    // reusing these buffers to preserve compatibility with other backends.
    const int64_t capacity_tokens = std::min<int64_t>(
            std::max(counts[0], counts[1]), static_cast<int64_t>(tile_tokens));
    const size_t capacity_bytes = detail::multiply(capacity_tokens, row);
    const bool fused_read = request_fused_read && rs_fused_read_eligible(counts, row);
    const size_t source_bytes = fused_read
            ? detail::multiply(detail::total_tokens(counts), row)
            : capacity_bytes;
    std::vector<uint8_t> a(source_bytes), b(source_bytes), reduced(capacity_bytes);
    if (fused_read) {
        // The per-rank source tensors each contain the same token count,
        // but different TP partial contributions. Reading them once avoids
        // a fresh RPC GET for each destination rank's local sequence.
        measure.read(partial[0], a.data(), 0, source_bytes);
        measure.read(partial[1], b.data(), 0, source_bytes);
    }

    int64_t global_start = 0;
    for (int rank = 0; rank < 2; ++rank) {
        for (int64_t done = 0; done < counts[rank]; ) {
            const int64_t n = std::min<int64_t>(counts[rank]-done, static_cast<int64_t>(tile_tokens));
            const size_t bytes = detail::multiply(n, row);
            const size_t src_offset = detail::multiply(global_start+done, row);
            const size_t dst_offset = detail::multiply(done, row);
            if (!fused_read) {
                measure.read(partial[0], a.data(), src_offset, bytes);
                measure.read(partial[1], b.data(), src_offset, bytes);
            }
            const uint8_t * chunk_a = a.data() + (fused_read ? src_offset : 0);
            const uint8_t * chunk_b = b.data() + (fused_read ? src_offset : 0);
            const auto host_start = ggml_galaxy_profile::clock::now();
            for (size_t k = 0; k < bytes/item; ++k) {
                detail::write(type, reduced.data()+k*item,
                        detail::read(type, chunk_a+k*item) +
                        detail::read(type, chunk_b+k*item));
            }
            if (measure.on) {
                measure.host_ms += ggml_galaxy_profile::ms(host_start, ggml_galaxy_profile::clock::now());
            }
            measure.write(shard[rank], reduced.data(), dst_offset, bytes);
            measure.sync(devices[rank]);
            done += n;
        }
        global_start += counts[rank];
    }
    measure.finish("RS", shard[0], counts[0] + counts[1], tile_tokens, fused_read);
}

// Input rank tensors each hold their own reduced sequence interval.
// Output rank tensors each receive the full original sequence in rank order.
// Unequal splits are handled without padding.
inline void all_gather(const devices_t & devices, const input_t & shard,
                       const output_t & replicated, counts_t counts, size_t tile_tokens = 32,
                       transport route = configured_transport()) {
    detail::validate(devices, shard, replicated, counts, tile_tokens, false);
    if (route == transport::rank_local) {
        detail::measure measure;
        const size_t row=detail::multiply(shard[0]->ne[0],
                                         detail::element_bytes(shard[0]->type));
        const int64_t total=detail::total_tokens(counts);
        const int64_t capacity=std::min<int64_t>(std::max(counts[0],counts[1]),
                                                static_cast<int64_t>(tile_tokens));
        std::vector<uint8_t> relay(detail::multiply(capacity,row));
        measure.sync(devices[0]);
        measure.sync(devices[1]);
        int64_t start=0;
        for(int rank=0;rank<2;++rank) {
            // Each rank already owns its part: materialize it LOCALLY and
            // relay a single copy to the other rank, as in AllGather.
            detail::device_copy_slice(devices[rank],shard[rank],0,
                                      replicated[rank],detail::multiply(start,row),
                                      counts[rank]);
            const int peer=1-rank;
            for(int64_t done=0;done<counts[rank];) {
                const int64_t n=std::min<int64_t>(counts[rank]-done,
                                                  static_cast<int64_t>(tile_tokens));
                const size_t bytes=detail::multiply(n,row);
                measure.read(shard[rank],relay.data(),detail::multiply(done,row),bytes);
                measure.write(replicated[peer],relay.data(),
                              detail::multiply(start+done,row),bytes);
                measure.sync(devices[peer]);
                done+=n;
            }
            start+=counts[rank];
        }
        measure.finish("AG",replicated[0],total,tile_tokens);
        detail::report_rank_local("AG",total,
                detail::multiply(total,row),detail::multiply(total,row));
        return;
    }
    detail::measure measure;
    const size_t row = detail::multiply(shard[0]->ne[0], detail::element_bytes(shard[0]->type));

    measure.sync(devices[0]);
    measure.sync(devices[1]);

    const int64_t capacity_tokens = std::min<int64_t>(
            std::max(counts[0], counts[1]), static_cast<int64_t>(tile_tokens));
    std::vector<uint8_t> data(detail::multiply(capacity_tokens, row));

    int64_t global_start = 0;
    for (int rank = 0; rank < 2; ++rank) {
        for (int64_t done = 0; done < counts[rank]; ) {
            const int64_t n = std::min<int64_t>(counts[rank]-done, static_cast<int64_t>(tile_tokens));
            const size_t bytes = detail::multiply(n, row);
            const size_t src_offset = detail::multiply(done, row);
            const size_t dst_offset = detail::multiply(global_start+done, row);
            measure.read(shard[rank], data.data(), src_offset, bytes);
            for (int dest = 0; dest < 2; ++dest) {
                measure.write(replicated[dest], data.data(), dst_offset, bytes);
                // The host buffer is reused after the backend write completes.
                measure.sync(devices[dest]);
            }
            done += n;
        }
        global_start += counts[rank];
    }
    measure.finish("AG", replicated[0], counts[0] + counts[1], tile_tokens);
}

// Source Galaxy-LM SPBertAttention computes Q locally on a sequence
// shard but explicitly gathers K and V across the complete sequence
// before the QK/softmax/V attention. These two gathers are independent
// but must use exactly the same token ownership (no implicit padding).
// For Qwen3, K/V are typically 3D; callers must provide contiguous 2D
// [head_features, local_tokens] tensors and restore head dimensions after
// gathering. This primitive does not by itself enable Qwen3 SP Attention:
// causal mask, KV-cache ownership and rank-local Q shape remain separate.
inline void all_gather_attention_kv(const devices_t & devices,
        const input_t & local_keys, const input_t & local_values,
        const output_t & complete_keys, const output_t & complete_values,
        counts_t counts, size_t tile_tokens = 32,
        transport route = configured_transport()) {
    if (local_keys[0] == local_values[0] || local_keys[1] == local_values[1] ||
            complete_keys[0] == complete_values[0] || complete_keys[1] == complete_values[1]) {
        throw std::invalid_argument("SP Attention K/V require distinct input and output tensor storage");
    }
    all_gather(devices, local_keys, complete_keys, counts, tile_tokens, route);
    all_gather(devices, local_values, complete_values, counts, tile_tokens, route);
}

// Copy the sequence interval owned by each rank out of an already mirrored
// full activation. This is needed for the residual path before an SP addition.
// Rank 0's mirror is the canonical source. No arithmetic takes place.
// For the initial implementation, host staging is used for both backends.
inline void scatter_mirrored(const devices_t & devices, const input_t & full,
                            const output_t & shard, counts_t counts, size_t tile_tokens = 32,
                            transport route = configured_transport()) {
    detail::validate(devices, full, shard, counts, tile_tokens, true);
    if (route == transport::rank_local) {
        detail::measure measure;
        const size_t row=detail::multiply(full[0]->ne[0],
                                         detail::element_bytes(full[0]->type));
        const int64_t total=detail::total_tokens(counts);
        measure.sync(devices[0]);
        measure.sync(devices[1]);
        int64_t start=0;
        for(int rank=0;rank<2;++rank) {
            // Strict precondition: both full activations are MIRRORED.
            // Never re-fetch from PC when Phone already owns its mirror.
            detail::device_copy_slice(devices[rank],full[rank],
                                      detail::multiply(start,row),
                                      shard[rank],0,counts[rank]);
            start+=counts[rank];
        }
        measure.finish("RESIDUAL_SCATTER",shard[0],total,tile_tokens);
        detail::report_rank_local("RESIDUAL_SCATTER",total,0,
                                  detail::multiply(total,row));
        return;
    }
    detail::measure measure;
    const size_t row = detail::multiply(full[0]->ne[0], detail::element_bytes(full[0]->type));

    measure.sync(devices[0]);
    measure.sync(devices[1]);

    const int64_t capacity_tokens = std::min<int64_t>(
            std::max(counts[0], counts[1]), static_cast<int64_t>(tile_tokens));
    std::vector<uint8_t> data(detail::multiply(capacity_tokens, row));

    int64_t global_start = 0;
    for (int rank = 0; rank < 2; ++rank) {
        for (int64_t done = 0; done < counts[rank]; ) {
            const int64_t n = std::min<int64_t>(counts[rank]-done, static_cast<int64_t>(tile_tokens));
            const size_t bytes = detail::multiply(n, row);
            const size_t src_offset = detail::multiply(global_start+done, row);
            const size_t dst_offset = detail::multiply(done, row);
            measure.read(full[0], data.data(), src_offset, bytes);
            measure.write(shard[rank], data.data(), dst_offset, bytes);
            measure.sync(devices[rank]);
            done += n;
        }
        global_start += counts[rank];
    }
    measure.finish("RESIDUAL_SCATTER", shard[0], counts[0] + counts[1], tile_tokens);
}

} // namespace ggml_galaxy_collectives

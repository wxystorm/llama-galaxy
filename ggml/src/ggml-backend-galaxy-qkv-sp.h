#pragma once

// Experimental first-prefill SP QKV projector. Unlike the Hybrid TP->SP
// attention bridge, each rank evaluates full Wq/Wk/Wv on ONLY its own token
// interval and applies the model's actual RoPE op_params at absolute positions.
// Either borrow model-load-resident full GGUF QKV matrices on each backend,
// or reconstruct the full matrices from TP shards in the staged fallback.
// Neither mode approximates or dequantizes GGUF weights on the host.
// The existing TP KV cache path is intentionally left untouched in v1.

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ggml_galaxy_qkv_sp {

inline void check(bool ok, const char * reason) {
    if (!ok) throw std::invalid_argument(std::string("Galaxy native SP QKV: ") + reason);
}

struct weight {
    ggml_type type = GGML_TYPE_F32;
    int64_t in = 0;
    int64_t out = 0;
    std::vector<uint8_t> bytes;  // populated only by legacy staged mode

    void validate_shape() const {
        check(in > 0 && out > 0 && in % ggml_blck_size(type) == 0,
              "weight shape / quantization block misaligned");
    }
    void validate() const {
        validate_shape();
        check(bytes.size() == ggml_row_size(type, in) * size_t(out),
              "full mirrored weight has incorrect byte extent");
    }
};

struct source {
    weight wq;
    weight wk;
    weight wv;
    // These physical model-owned tensors remain valid for the model buffer
    // lifetime. The projection graphs borrow them; they never free them.
    // Rank 0=PC, rank 1=Phone; per rank: 0=Wq,1=Wk,2=Wv.
    bool resident_weights = false;
    std::array<std::array<ggml_tensor *,3>,2> physical_weights = {};
    int64_t hidden = 0;
    int64_t tokens = 0;
    int64_t head_dim = 0;
    int64_t q_heads = 0;
    int64_t kv_heads = 0;
    std::vector<uint8_t> normalized;      // full F32 [hidden,tokens]
    std::vector<int32_t> positions;       // original absolute RoPE positions
    std::vector<uint8_t> rope_factors;    // optional exact model F32 factors
    std::array<int32_t, GGML_MAX_OP_PARAMS/sizeof(int32_t)> q_rope_params = {};
    std::array<int32_t, GGML_MAX_OP_PARAMS/sizeof(int32_t)> k_rope_params = {};
};

struct projected {
    // Each rank holds ALL heads for its own contiguous token interval.
    // Row order [head_dim,heads,local_tokens], type F32.
    std::array<std::vector<uint8_t>, 2> q;
    std::array<std::vector<uint8_t>, 2> k;
    std::array<std::vector<uint8_t>, 2> v;
};

inline void validate(const source & s) {
    if (s.resident_weights) {
        s.wq.validate_shape();
        s.wk.validate_shape();
        s.wv.validate_shape();
        const std::array<const weight *,3> shapes={&s.wq,&s.wk,&s.wv};
        for (int rank=0;rank<2;++rank) {
            for (int kind=0;kind<3;++kind) {
                const ggml_tensor * t=s.physical_weights[rank][kind];
                check(t != nullptr && t->buffer != nullptr && t->op == GGML_OP_NONE &&
                      t->view_src == nullptr && t->type == shapes[kind]->type &&
                      t->ne[0] == shapes[kind]->in &&
                      t->ne[1] == shapes[kind]->out &&
                      t->ne[2] == 1 && t->ne[3] == 1 &&
                      ggml_is_contiguous(t),
                      "model-owned resident QKV tensor missing or has wrong shape/type");
            }
        }
    } else {
        s.wq.validate();
        s.wk.validate();
        s.wv.validate();
    }
    check(s.tokens >= 8 && s.tokens <= 512 && s.hidden > 0 &&
          (s.head_dim == 64 || s.head_dim == 128) &&
          s.q_heads > 0 && s.kv_heads > 0 && s.q_heads % s.kv_heads == 0,
          "only dense first-prefill Llama QKV with valid GQA is supported");
    check(s.wq.in == s.hidden && s.wk.in == s.hidden && s.wv.in == s.hidden &&
          s.wq.out == s.q_heads*s.head_dim &&
          s.wk.out == s.kv_heads*s.head_dim &&
          s.wv.out == s.kv_heads*s.head_dim,
          "full Q/K/V projection dimensions mismatch");
    check(s.normalized.size() == size_t(s.hidden)*size_t(s.tokens)*sizeof(float) &&
          s.positions.size() == size_t(s.tokens),
          "normalized hidden state / positions must cover the entire interval");
    check(s.rope_factors.empty() ||
          (s.rope_factors.size() == size_t(s.head_dim/2)*sizeof(float)),
          "unsupported RoPE factor shape");
    for (int64_t i = 0; i < s.tokens; ++i) {
        check(s.positions[size_t(i)] == i,
              "only contiguous zero-based first Prefill positions are supported");
    }
}

// Verify that a *real* Llama ggml_set_rows writer reads from this
// post-projection K/V producer, following only layout-preserving views.
// Important: current ggml_set_rows stores the VALUE in src[0], row
// INDICES in src[1], and the cache DESTINATION in src[2]. These differ
// from the public ggml_set_rows(ctx,dst,values,indices) argument order.
inline bool cache_set_rows_from(const ggml_tensor * writer,
                                const ggml_tensor * producer) {
    if (!writer || !producer || writer->op!=GGML_OP_SET_ROWS ||
            !writer->src[0] || !writer->src[1] || !writer->src[2]) return false;
    const auto * index=writer->src[1];
    if (index->type!=GGML_TYPE_I64 && index->type!=GGML_TYPE_I32) return false;
    if (writer->src[0]->ne[1]!=index->ne[0] ||
            writer->src[2]->type!=writer->type) return false;
    const ggml_tensor * input=writer->src[0];
    for(int depth=0;input && depth<12;++depth) {
        if(input==producer) return true;
        if(input->op!=GGML_OP_VIEW && input->op!=GGML_OP_RESHAPE &&
                input->op!=GGML_OP_CONT && input->op!=GGML_OP_PERMUTE)
            return false;
        input=input->src[0];
    }
    return false;
}

// Materialize the F32 TP-head view expected by the existing Llama KV
// cache writers. Native SP ranks own disjoint token ranges, but *each*
 // native rank computes every K/V head. The original TP ranks instead own
// disjoint head ranges and need the whole token interval on each rank.
// This is a host bridge for the first-prefill experiment, not a new cache.
inline std::array<std::vector<uint8_t>,2> token_to_head_f32(
        const std::array<std::vector<uint8_t>,2> & token_shards,
        int64_t tokens, int64_t head_dim,
        const std::array<int64_t,2> & heads) {
    check(tokens > 0 && head_dim > 0 && heads[0]>0 && heads[1]>0,
          "invalid native-to-TP KV geometry");
    const int64_t full_heads=heads[0]+heads[1];
    const std::array<int64_t,2> counts={(tokens+1)/2,tokens/2};
    check(counts[1]>0, "empty SP rank token partition");
    const size_t bytes_per_head=size_t(head_dim)*sizeof(float);
    const size_t full_token_bytes=size_t(full_heads)*bytes_per_head;
    std::array<std::vector<uint8_t>,2> tp;
    for (int rank=0;rank<2;++rank) {
        check(token_shards[rank].size() == size_t(counts[rank])*full_token_bytes,
              "SP local K/V tensor byte extent is incorrect");
        tp[rank].resize(size_t(tokens)*size_t(heads[rank])*bytes_per_head);
    }
    int64_t token_begin=0;
    for (int src_rank=0;src_rank<2;++src_rank) {
        for (int64_t t=0;t<counts[src_rank];++t) {
            const auto * source=token_shards[src_rank].data()+
                    size_t(t)*full_token_bytes;
            int64_t head_begin=0;
            for (int dst_rank=0;dst_rank<2;++dst_rank) {
                const size_t size=size_t(heads[dst_rank])*bytes_per_head;
                auto * dest=tp[dst_rank].data()+
                        size_t(token_begin+t)*size;
                std::memcpy(dest,source+size_t(head_begin)*bytes_per_head,size);
                head_begin+=heads[dst_rank];
            }
        }
        token_begin+=counts[src_rank];
    }
    check(token_begin==tokens,"SP/TP K/V token reconstruction incomplete");
    return tp;
}

// The live Meta graph does not promise to enumerate model-weight leaves.
// Instead, follow each *actual* Q/K/V producer from the pinned stage back
// through its known native reshape/RoPE path to GGML_OP_MUL_MAT. Guard both
// the original RMSNorm input and the exact named static GGUF weight, so a
// wrong layer or an unsupported fused/bias/LoRA path cannot be accepted.
inline const ggml_tensor * projection_weight(const ggml_tensor * stage,
                                               const ggml_tensor * normalized,
                                               const std::string & expected_name) {
    check(stage && normalized, "missing stage or normalized projection input");
    const ggml_tensor * node=stage;
    for (int depth=0;depth<12 && node; ++depth) {
        if (node->op==GGML_OP_MUL_MAT) {
            const ggml_tensor * weights=node->src[0];
            if (!weights || node->src[1]!=normalized ||
                    weights->op!=GGML_OP_NONE || weights->view_src!=nullptr ||
                    std::strcmp(weights->name,expected_name.c_str())!=0) {
                throw std::invalid_argument("Galaxy native SP QKV: unexpected MatMul inputs for "+
                        expected_name + " (weight=" +
                        (weights ? std::string(weights->name) : std::string("NULL")) +
                        ", norm_match=" + (node->src[1]==normalized ? "1" : "0") + ")");
            }
            return weights;
        }
        // The dense Llama Q/K path is ROPE -> RESHAPE -> MUL_MAT,
        // while V is RESHAPE -> MUL_MAT. Only harmless view/contiguous
        // wrappers are permitted; e.g. an ADD bias is not silently lost.
        if (node->op!=GGML_OP_ROPE && node->op!=GGML_OP_RESHAPE &&
                node->op!=GGML_OP_VIEW && node->op!=GGML_OP_CONT) {
            throw std::invalid_argument("Galaxy native SP QKV: unsupported projection ancestor "+
                    std::string(ggml_op_name(node->op)) + " while locating " + expected_name);
        }
        node=node->src[0];
    }
    throw std::invalid_argument("Galaxy native SP QKV: no MatMul ancestor for "+expected_name);
}

// A rank graph owns its OWN full QKV matrix copies, but operates on only
// its local tokens. No TP->SP HeadToToken collective appears in this graph.
struct rank_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t storage = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * inp = nullptr;
    ggml_tensor * pos = nullptr;
    ggml_tensor * factors = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    int64_t begin = 0;
    int64_t local = 0;

    rank_graph(ggml_backend_t backend, const source & s, int rank) {
        const ggml_init_params params = {size_t(4)*1024*1024, nullptr, true};
        ctx = ggml_init(params);
        check(ctx != nullptr, "ggml context allocation failed");
        const int64_t first = (s.tokens+1)/2;
        begin = rank == 0 ? 0 : first;
        local = rank == 0 ? first : s.tokens-first;
        check(local > 0, "empty local token interval");

        // Resident mode uses GGML model-weight tensors already allocated
        // by the Meta model loader on THIS physical backend. The graph
        // allocates only activations, RoPE positions and output tensors.
        wq = s.resident_weights ? s.physical_weights[rank][0] :
                ggml_new_tensor_2d(ctx,s.wq.type,s.wq.in,s.wq.out);
        wk = s.resident_weights ? s.physical_weights[rank][1] :
                ggml_new_tensor_2d(ctx,s.wk.type,s.wk.in,s.wk.out);
        wv = s.resident_weights ? s.physical_weights[rank][2] :
                ggml_new_tensor_2d(ctx,s.wv.type,s.wv.in,s.wv.out);
        inp = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,s.hidden,local);
        pos = ggml_new_tensor_1d(ctx,GGML_TYPE_I32,local);
        if (!s.rope_factors.empty()) {
            factors = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,s.head_dim/2);
        }

        ggml_tensor * qmm = ggml_mul_mat(ctx,wq,inp);
        ggml_tensor * kmm = ggml_mul_mat(ctx,wk,inp);
        ggml_tensor * vmm = ggml_mul_mat(ctx,wv,inp);
        ggml_tensor * q3 = ggml_reshape_3d(ctx,qmm,s.head_dim,s.q_heads,local);
        ggml_tensor * k3 = ggml_reshape_3d(ctx,kmm,s.head_dim,s.kv_heads,local);

        // The constructor's placeholder RoPE values are replaced with the
        // exact GGML op_params from the model's Qcur/Kcur RoPE nodes, which
        // include rope_type, n_rot, original context and scaling factors.
        q = ggml_rope_ext(ctx,q3,pos,factors,s.head_dim,0,0,
                          10000.0f,1.0f,0.0f,1.0f,32.0f,1.0f);
        k = ggml_rope_ext(ctx,k3,pos,factors,s.head_dim,0,0,
                          10000.0f,1.0f,0.0f,1.0f,32.0f,1.0f);
        std::memcpy(q->op_params,s.q_rope_params.data(),sizeof(q->op_params));
        std::memcpy(k->op_params,s.k_rope_params.data(),sizeof(k->op_params));
        v = ggml_reshape_3d(ctx,vmm,s.head_dim,s.kv_heads,local);
        ggml_set_input(inp);
        ggml_set_input(pos);
        ggml_set_output(q);
        ggml_set_output(k);
        ggml_set_output(v);
        graph = ggml_new_graph_custom(ctx,128,false);
        ggml_build_forward_expand(graph,q);
        ggml_build_forward_expand(graph,k);
        ggml_build_forward_expand(graph,v);
        check(ggml_backend_dev_supports_op(ggml_backend_get_device(backend),q) &&
              ggml_backend_dev_supports_op(ggml_backend_get_device(backend),k),
              "device does not support native RoPE projection");
        storage = ggml_backend_alloc_ctx_tensors(ctx,backend);
        check(storage != nullptr, "backend QKV tensor allocation failed");
    }

    ~rank_graph() {
        if (storage) ggml_backend_buffer_free(storage);
        if (ctx) ggml_free(ctx);
    }
    rank_graph(const rank_graph &) = delete;
    rank_graph & operator=(const rank_graph &) = delete;

    void upload(ggml_backend_t backend, const source & s) {
        if (!s.resident_weights) {
            ggml_backend_tensor_set(wq,s.wq.bytes.data(),0,s.wq.bytes.size());
            ggml_backend_tensor_set(wk,s.wk.bytes.data(),0,s.wk.bytes.size());
            ggml_backend_tensor_set(wv,s.wv.bytes.data(),0,s.wv.bytes.size());
        }
        const size_t row = size_t(s.hidden)*sizeof(float);
        ggml_backend_tensor_set(inp,s.normalized.data()+size_t(begin)*row,0,size_t(local)*row);
        ggml_backend_tensor_set(pos,s.positions.data()+begin,0,size_t(local)*sizeof(int32_t));
        if (factors) {
            ggml_backend_tensor_set(factors,s.rope_factors.data(),0,s.rope_factors.size());
        }
        ggml_backend_synchronize(backend);
    }
};

inline projected run(const std::array<ggml_backend_t,2> & devices, const source & s) {
    validate(s);
    check(devices[0] && devices[1], "two physical backends are required");
    rank_graph pc(devices[0],s,0);
    rank_graph phone(devices[1],s,1);
    rank_graph * ranks[2] = {&pc,&phone};
    for (int rank=0;rank<2;++rank) ranks[rank]->upload(devices[rank],s);

    for (int rank=0;rank<2;++rank) {
        const ggml_status status=ggml_backend_graph_compute_async(devices[rank],ranks[rank]->graph);
        check(status == GGML_STATUS_SUCCESS, "native MatMul/RoPE dispatch failed");
    }
    projected out;
    for (int rank=0;rank<2;++rank) {
        // RPC backend synchronize is not sufficient as an execution fence:
        // tensor_get through the same socket forces command-order completion.
        ggml_backend_synchronize(devices[rank]);
        for (const auto pair : std::array<std::pair<ggml_tensor *,std::vector<uint8_t> *>,3>{{
                {ranks[rank]->q,&out.q[rank]},
                {ranks[rank]->k,&out.k[rank]},
                {ranks[rank]->v,&out.v[rank]}}}) {
            pair.second->resize(ggml_nbytes(pair.first));
            ggml_backend_tensor_get(pair.first,pair.second->data(),0,pair.second->size());
        }
    }
    return out;
}

} // namespace ggml_galaxy_qkv_sp

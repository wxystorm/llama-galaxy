#pragma once

// Opt-in live causal SP Attention shadow for Meta's *real* Llama TP graph.
//
// IMPORTANT: the caller must supply producer-time copies of rank-local
// post-RoPE Q/K and projected V and the pre-Wo TP Attention output. No GLG1
// files, artificial source projections or model weight mirrors are used.
//
// In SHADOW mode the actual model output is NEVER overwritten. This runtime
// checks the entire TP->SP->device Flash->TP path and fails closed when the
// numerical error is too large. First-prefill, 2 physical ranks, dense
// Llama with contiguous QKV heads, single sequence and F16 K/V only.
//
// GPU/RPC collectives intentionally use correctness-first host staging. The
// per-backend FLASH_ATTN_EXT kernels are native to CUDA and Phone OpenCL.

#include "ggml-backend-galaxy-layout.h"
#include "ggml-backend-galaxy-collectives.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ggml_galaxy_causal_live {

using ggml_galaxy_layout::exchange;
using ggml_galaxy_collectives::devices_t;
using ggml_galaxy_collectives::input_t;
using ggml_galaxy_collectives::output_t;
using ggml_galaxy_collectives::counts_t;

struct head_shards {
    // Rank-local, producer-time bytes; caller owns their lifetime.
    std::array<const std::vector<uint8_t> *, 2> bytes = {nullptr, nullptr};
    std::array<int64_t, 2> heads = {0, 0};
    ggml_type type = GGML_TYPE_F32;
};

struct request {
    head_shards q;
    head_shards k;
    head_shards v;
    head_shards reference;
    int64_t tokens = 0;
    int64_t head_dim = 0;
    int layer = -1;
    size_t tile_tokens = 32;
    double max_rmse = 0.001;
};

struct measurement {
    double rmse = 0;
    double max_abs = 0;
    double cosine = 0;
    double first_token_rmse = 0;
    int64_t worst_token = 0;
    int64_t worst_channel = 0;
    double head_to_token_ms = 0;
    double kv_gather_ms = 0;
    double native_phase_ms = 0;
    double token_to_head_ms = 0;
};

inline void require(bool ok, const std::string & explanation) {
    if (!ok) throw std::invalid_argument("Galaxy live causal SP: "+explanation);
}

inline int64_t total_heads(const head_shards & s) {
    require(s.heads[0] > 0 && s.heads[1] > 0 &&
            s.heads[0] <= INT64_MAX-s.heads[1], "nonempty head count required");
    return s.heads[0]+s.heads[1];
}

inline void validate_shards(const head_shards & s, int64_t tokens, int64_t dim,
                            const char * stage) {
    const int64_t heads = total_heads(s);
    require(tokens > 0 && dim > 0 && heads <= INT64_MAX/dim,
            std::string(stage)+": invalid dimensions");
    require(s.type == GGML_TYPE_F32, std::string(stage)+": F32 required in live shadow");
    for (int rank=0;rank<2;++rank) {
        require(s.bytes[rank] != nullptr, std::string(stage)+": missing source");
        const auto width = s.heads[rank]*dim;
        require(width <= INT64_MAX/tokens &&
                s.bytes[rank]->size() == static_cast<size_t>(width*tokens)*sizeof(float),
                std::string(stage)+": rank source bytes do not match head/token shape");
    }
}

inline void validate(const request & r) {
    require(r.layer >= 0 && r.tokens >= 8 && r.tokens <= 512 &&
            r.head_dim > 0 && r.head_dim <= 256,
            "first-prefill live shadow supports 8..512 tokens, head_dim<=256");
    require(r.tile_tokens > 0 && r.tile_tokens <= 1024,
            "collective tile must be 1..1024");
    require(std::isfinite(r.max_rmse) && r.max_rmse >= 0,
            "RMSE tolerance must be nonnegative and finite");
    validate_shards(r.q,r.tokens,r.head_dim,"Q");
    validate_shards(r.k,r.tokens,r.head_dim,"K");
    validate_shards(r.v,r.tokens,r.head_dim,"V");
    validate_shards(r.reference,r.tokens,r.head_dim,"pre-Wo Attention");
    const int64_t q=total_heads(r.q),kv=total_heads(r.k);
    require(kv == total_heads(r.v) && q == total_heads(r.reference),
            "head counts disagree across Q/K/V/Attention");
    require(q%kv == 0 && r.q.heads[0]% (q/kv) == 0 &&
            r.q.heads[0]/(q/kv) == r.k.heads[0] &&
            r.k.heads == r.v.heads && r.q.heads == r.reference.heads,
            "TP splits must be aligned to complete GQA groups");
    require(q*r.head_dim <= INT32_MAX/r.tokens &&
            kv*r.head_dim <= INT32_MAX/r.tokens,
            "GLG1-sized activation extent overflow");
}

inline float read_f32(const std::vector<uint8_t> & data, size_t i) {
    float result;
    std::memcpy(&result,data.data()+i*sizeof(float),sizeof(result));
    return result;
}

inline std::vector<ggml_fp16_t> convert_f16(const std::vector<uint8_t> & data) {
    require(data.size()%sizeof(float)==0, "F16 conversion requires F32 source");
    const size_t n=data.size()/sizeof(float);
    std::vector<ggml_fp16_t> half(n);
    for (size_t i=0;i<n;++i) {
        const float x=read_f32(data,i);
        require(std::isfinite(x),"nonfinite GGUF activation");
        half[i]=ggml_fp32_to_fp16(x);
    }
    return half;
}

struct native_rank {
    ggml_context * ctx=nullptr;
    ggml_backend_buffer_t storage=nullptr;
    ggml_cgraph * graph=nullptr;

    ggml_tensor * tp_q=nullptr;
    ggml_tensor * tp_k=nullptr;
    ggml_tensor * tp_v=nullptr;
    ggml_tensor * sp_q=nullptr;
    ggml_tensor * sp_k=nullptr;
    ggml_tensor * sp_v=nullptr;
    ggml_tensor * full_k=nullptr;
    ggml_tensor * full_v=nullptr;
    ggml_tensor * k_storage=nullptr;
    ggml_tensor * v_storage=nullptr;
    ggml_tensor * mask=nullptr;
    ggml_tensor * sp_output=nullptr;
    ggml_tensor * tp_output=nullptr;

    int64_t rank_queries=0;
    int64_t begin=0;
    int64_t kv_padded=0;

    native_rank(ggml_backend_t backend, const request & r, int rank) {
        const ggml_init_params p = {size_t(4)*1024*1024, nullptr, true};
        ctx=ggml_init(p);
        require(ctx!=nullptr,"failed to initialize live Flash context");
        const int64_t q_heads=total_heads(r.q);
        const int64_t kv_heads=total_heads(r.k);
        const int64_t qw=q_heads*r.head_dim, kvw=kv_heads*r.head_dim;
        const int64_t local=(r.tokens+1)/2;
        begin=rank==0?0:local;
        rank_queries=rank==0?local:r.tokens-local;
        kv_padded=GGML_PAD(r.tokens,256);

        tp_q=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,r.q.heads[rank]*r.head_dim,r.tokens);
        tp_k=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,r.k.heads[rank]*r.head_dim,r.tokens);
        tp_v=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,r.v.heads[rank]*r.head_dim,r.tokens);
        tp_output=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,r.q.heads[rank]*r.head_dim,r.tokens);

        sp_q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,r.head_dim,q_heads,rank_queries);
        sp_k=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,kvw,rank_queries);
        sp_v=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,kvw,rank_queries);

        k_storage=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,kvw,kv_padded);
        v_storage=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,kvw,kv_padded);
        const size_t row=static_cast<size_t>(kvw)*sizeof(ggml_fp16_t);
        full_k=ggml_view_2d(ctx,k_storage,kvw,r.tokens,row,0);
        full_v=ggml_view_2d(ctx,v_storage,kvw,r.tokens,row,0);

        mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,kv_padded,GGML_PAD(rank_queries,32));
        ggml_set_input(sp_q);
        ggml_set_input(k_storage);
        ggml_set_input(v_storage);
        ggml_set_input(mask);

        ggml_tensor * q=ggml_permute(ctx,sp_q,0,2,1,3);
        ggml_tensor * k=ggml_reshape_3d(ctx,k_storage,r.head_dim,kv_heads,kv_padded);
        k=ggml_permute(ctx,k,0,2,1,3);
        ggml_tensor * v=ggml_reshape_3d(ctx,v_storage,r.head_dim,kv_heads,kv_padded);
        v=ggml_permute(ctx,v,0,2,1,3);
        ggml_tensor * output=ggml_flash_attn_ext(
                ctx,q,k,v,mask,1.0f/std::sqrt(float(r.head_dim)),0.0f,0.0f);
        ggml_flash_attn_ext_set_prec(output,GGML_PREC_F32);
        sp_output=ggml_reshape_2d(ctx,output,qw,rank_queries);
        ggml_set_output(sp_output);
        graph=ggml_new_graph_custom(ctx,128,false);
        ggml_build_forward_expand(graph,sp_output);

        storage=ggml_backend_alloc_ctx_tensors(ctx,backend);
        require(storage!=nullptr,"failed to allocate live Flash backend tensors");
    }

    void initialize(ggml_backend_t backend, const request & r, int rank) {
        const std::vector<ggml_fp16_t> zero(static_cast<size_t>(ggml_nelements(k_storage)),0);
        ggml_backend_tensor_set(k_storage,zero.data(),0,ggml_nbytes(k_storage));
        ggml_backend_tensor_set(v_storage,zero.data(),0,ggml_nbytes(v_storage));
        ggml_backend_tensor_set(tp_q,r.q.bytes[rank]->data(),0,ggml_nbytes(tp_q));
        const auto keys=convert_f16(*r.k.bytes[rank]);
        const auto vals=convert_f16(*r.v.bytes[rank]);
        ggml_backend_tensor_set(tp_k,keys.data(),0,ggml_nbytes(tp_k));
        ggml_backend_tensor_set(tp_v,vals.data(),0,ggml_nbytes(tp_v));

        const int64_t padded_rows=GGML_PAD(rank_queries,32);
        std::vector<ggml_fp16_t> causal(static_cast<size_t>(kv_padded*padded_rows),
                            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t i=0;i<rank_queries;++i) {
            const int64_t absolute=begin+i;
            for (int64_t k=0;k<=absolute;++k) {
                causal[static_cast<size_t>(i*kv_padded+k)]=ggml_fp32_to_fp16(0.0f);
            }
        }
        ggml_backend_tensor_set(mask,causal.data(),0,ggml_nbytes(mask));
        ggml_backend_synchronize(backend); // RPC/driver host buffer lifetime
    }

    ~native_rank() {
        if (storage) ggml_backend_buffer_free(storage);
        if (ctx) ggml_free(ctx);
    }
    native_rank(const native_rank &)=delete;
    native_rank & operator=(const native_rank &)=delete;
};

inline double ms(ggml_galaxy_profile::instant start) {
    return ggml_galaxy_profile::ms(start,ggml_galaxy_profile::clock::now());
}

inline double first_token_error(const request & r) {
    // In a fresh causal prefill query 0 has only value 0 visible. Unlike a
    // graph-level cached-token check this is a strong numerical guard against
    // silently validating continuation or a different attention mask.
    const int64_t q_total=total_heads(r.q),kv_total=total_heads(r.k);
    const int64_t group=q_total/kv_total;
    double sum_sq=0;
    uint64_t count=0;
    for (int rank=0;rank<2;++rank) {
        const auto & v=*r.v.bytes[rank];
        const auto & reference=*r.reference.bytes[rank];
        for (int64_t h=0;h<r.q.heads[rank];++h) {
            for (int64_t d=0;d<r.head_dim;++d) {
                const size_t voff=static_cast<size_t>((h/group)*r.head_dim+d);
                const size_t ooff=static_cast<size_t>(h*r.head_dim+d);
                const double delta=double(read_f32(v,voff))-double(read_f32(reference,ooff));
                sum_sq+=delta*delta;
                count++;
            }
        }
    }
    return std::sqrt(sum_sq/double(count));
}

inline measurement shadow(const devices_t & devices, const request & r) {
    validate(r);
    require(devices[0] && devices[1],"two native backend handles required");
    measurement m;
    m.first_token_rmse=first_token_error(r);
    require(m.first_token_rmse <= 0.005,
            "not a supported fresh causal first prefill (token0 != V0)");

    native_rank pc(devices[0],r,0);
    native_rank phone(devices[1],r,1);
    native_rank * ranks[2]={&pc,&phone};
    for(int rank=0;rank<2;++rank)ranks[rank]->initialize(devices[rank],r,rank);

    const counts_t tokens={pc.rank_queries,phone.rank_queries};
    const exchange q_layout={r.q.heads,tokens,r.head_dim};
    const exchange kv_layout={r.k.heads,tokens,r.head_dim};

    auto start=ggml_galaxy_profile::clock::now();
    ggml_galaxy_layout::head_to_token(
            devices,{pc.tp_q,phone.tp_q},{pc.sp_q,phone.sp_q},q_layout,r.tile_tokens);
    ggml_galaxy_layout::head_to_token(
            devices,{pc.tp_k,phone.tp_k},{pc.sp_k,phone.sp_k},kv_layout,r.tile_tokens);
    ggml_galaxy_layout::head_to_token(
            devices,{pc.tp_v,phone.tp_v},{pc.sp_v,phone.sp_v},kv_layout,r.tile_tokens);
    m.head_to_token_ms=ms(start);

    start=ggml_galaxy_profile::clock::now();
    ggml_galaxy_collectives::all_gather_attention_kv(
            devices,{pc.sp_k,phone.sp_k},{pc.sp_v,phone.sp_v},
            {pc.full_k,phone.full_k},{pc.full_v,phone.full_v},
            tokens,r.tile_tokens);
    m.kv_gather_ms=ms(start);

    start=ggml_galaxy_profile::clock::now();
    for (int rank=0;rank<2;++rank) {
        const ggml_status status=ggml_backend_graph_compute_async(
                devices[rank],ranks[rank]->graph);
        require(status==GGML_STATUS_SUCCESS,"native Flash graph execution failed");
    }
    for (int rank=0;rank<2;++rank) {
        ggml_backend_synchronize(devices[rank]);
    }
    m.native_phase_ms=ms(start);

    start=ggml_galaxy_profile::clock::now();
    ggml_galaxy_layout::token_to_head(
            devices,{pc.sp_output,phone.sp_output},
            {pc.tp_output,phone.tp_output},q_layout,r.tile_tokens);
    m.token_to_head_ms=ms(start);

    double norm_ref=0,norm_out=0,dot=0,sq=0;
    uint64_t seen=0;
    const int64_t q_heads=total_heads(r.q);
    for (int rank=0;rank<2;++rank) {
        ggml_backend_synchronize(devices[rank]);
        std::vector<uint8_t> actual(ggml_nbytes(ranks[rank]->tp_output));
        ggml_backend_tensor_get(ranks[rank]->tp_output,actual.data(),0,actual.size());
        const auto & expected=*r.reference.bytes[rank];
        for (int64_t t=0;t<r.tokens;++t) {
            for (int64_t f=0;f<r.q.heads[rank]*r.head_dim;++f) {
                const size_t ix=static_cast<size_t>(t*r.q.heads[rank]*r.head_dim+f);
                const double ref=read_f32(expected,ix);
                const double val=read_f32(actual,ix);
                require(std::isfinite(val),"non-finite native Flash result");
                const double delta=ref-val;
                sq+=delta*delta;
                dot+=ref*val;
                norm_ref+=ref*ref;
                norm_out+=val*val;
                if (std::abs(delta)>m.max_abs) {
                    m.max_abs=std::abs(delta);
                    m.worst_token=t;
                    m.worst_channel=(rank==0?0:r.q.heads[0]*r.head_dim)+f;
                }
                seen++;
            }
        }
    }
    require(seen==static_cast<uint64_t>(r.tokens*q_heads*r.head_dim),
            "incomplete native Flash output");
    m.rmse=std::sqrt(sq/double(seen));
    m.cosine=dot/std::sqrt(norm_ref*norm_out);
    require(m.rmse<=r.max_rmse,
            "native Flash vs production Attention RMSE exceeds requested threshold");
    return m;
}

} // namespace ggml_galaxy_causal_live

// CPU + CPU executable smoke for the exact native L0 SP Q/K/V projector.
// Uses real GGML MatMul and RoPE nodes with independently owned local token
// intervals. The production path supplies GGUF quantized weights instead of
// these deterministic F32 identity fixtures.
#include "../../ggml/src/ggml-backend-galaxy-qkv-sp.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ggml_galaxy_qkv_sp;

static void expect(bool yes, const char * what) {
    if (!yes) throw std::runtime_error(what);
}
static float from_bytes(const std::vector<uint8_t> & bytes, size_t i) {
    float result=0.0f;
    std::memcpy(&result,bytes.data()+i*sizeof(float),sizeof(float));
    return result;
}
static weight identity_weight(int64_t hidden, int64_t out) {
    weight result;
    result.type=GGML_TYPE_F32;
    result.in=hidden;
    result.out=out;
    std::vector<float> floats(size_t(hidden)*size_t(out),0.0f);
    for(int64_t j=0;j<out;++j) floats[size_t(j*hidden+j)]=1.0f;
    result.bytes.resize(floats.size()*sizeof(float));
    std::memcpy(result.bytes.data(),floats.data(),result.bytes.size());
    return result;
}

static void check_geometry(const projected & r, const source & s) {
    const int64_t first=(s.tokens+1)/2;
    for(int rank=0;rank<2;++rank) {
        const int64_t local=rank==0?first:s.tokens-first;
        const int64_t begin=rank==0?0:first;
        expect(r.q[rank].size() == size_t(local*s.q_heads*s.head_dim)*sizeof(float),
               "wrong Q rank-local output shape");
        expect(r.k[rank].size() == size_t(local*s.kv_heads*s.head_dim)*sizeof(float),
               "wrong K rank-local output shape");
        expect(r.v[rank].size() == size_t(local*s.kv_heads*s.head_dim)*sizeof(float),
               "wrong V rank-local output shape");
        for(int64_t t=0;t<local;++t) {
            const int64_t global=begin+t;
            for(int64_t c=0;c<s.kv_heads*s.head_dim;++c) {
                const float expected=from_bytes(s.normalized,size_t(global*s.hidden+c));
                const float observed=from_bytes(r.v[rank],
                        size_t(t*s.kv_heads*s.head_dim+c));
                expect(std::abs(expected-observed)<1e-4f,
                       "rank-local V MatMul does not match identity GGUF projection");
            }
        }
    }
    // Position zero must leave Q/K unchanged under original RoPE semantics.
    for (int64_t c=0;c<s.head_dim;++c) {
        const float x=from_bytes(s.normalized,size_t(c));
        expect(std::abs(from_bytes(r.q[0],size_t(c))-x)<1e-4f,
               "rank0 Q MatMul/RoPE token0 mismatch");
        expect(std::abs(from_bytes(r.k[0],size_t(c))-x)<1e-4f,
               "rank0 K MatMul/RoPE token0 mismatch");
    }
}

int main() {
    try {
        ggml_backend_load_all();
        ggml_backend_t pc=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        ggml_backend_t phone=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        expect(pc && phone, "CPU backends unavailable");

        // Copy genuine ggml_rope_ext op_params from a graph, just as the
        // production Meta path does from the real model's Qcur/Kcur.
        const ggml_init_params params={size_t(1024)*1024,nullptr,true};
        ggml_context * ctx=ggml_init(params);
        expect(ctx!=nullptr,"fixture context missing");
        ggml_tensor * pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,8);
        // Simulate Llama's real Q/K/V producer topology: a static named
        // weight, mirrored norm, MatMul, Reshape, then Q/K RoPE. The weights
        // are deliberately NOT inserted into a cgraph->leafs list.
        ggml_tensor * normalized=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,128,8);
        ggml_tensor * wq=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,128,128);
        ggml_tensor * wk=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,128,64);
        ggml_tensor * wv=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,128,64);
        ggml_set_name(wq,"blk.0.attn_q.weight");
        ggml_set_name(wk,"blk.0.attn_k.weight");
        ggml_set_name(wv,"blk.0.attn_v.weight");
        ggml_tensor * src_q=ggml_reshape_3d(ctx,ggml_mul_mat(ctx,wq,normalized),64,2,8);
        ggml_tensor * src_k=ggml_reshape_3d(ctx,ggml_mul_mat(ctx,wk,normalized),64,1,8);
        ggml_tensor * src_v=ggml_reshape_3d(ctx,ggml_mul_mat(ctx,wv,normalized),64,1,8);
        ggml_tensor * model_q=ggml_rope_ext(ctx,src_q,pos,nullptr,64,
                          GGML_ROPE_TYPE_NORMAL,8192,10000.f,1.f,0.f,1.f,32.f,1.f);
        ggml_tensor * model_k=ggml_rope_ext(ctx,src_k,pos,nullptr,64,
                          GGML_ROPE_TYPE_NORMAL,8192,10000.f,1.f,0.f,1.f,32.f,1.f);
        expect(projection_weight(model_q,normalized,"blk.0.attn_q.weight")==wq,
               "Q weight must be found by producer ancestry");
        expect(projection_weight(model_k,normalized,"blk.0.attn_k.weight")==wk,
               "K weight must be found by producer ancestry");
        expect(projection_weight(src_v,normalized,"blk.0.attn_v.weight")==wv,
               "V weight must be found by producer ancestry");
        bool wrong_weight_rejected=false;
        try {
            (void) projection_weight(model_q,normalized,"blk.1.attn_q.weight");
        } catch (const std::invalid_argument &) {
            wrong_weight_rejected=true;
        }
        expect(wrong_weight_rejected,"cross-layer QKV weights must be rejected");
        bool wrong_normalized_rejected=false;
        try {
            (void) projection_weight(model_q,src_q,"blk.0.attn_q.weight");
        } catch (const std::invalid_argument &) {
            wrong_normalized_rejected=true;
        }
        expect(wrong_normalized_rejected,"unrelated MatMul activation must be rejected");

        // Regression for real ggml_set_rows source ordering. Llama calls
        // set_rows(cache, data, idx), but the GGML op stores src[0]=data,
        // src[1]=indices, src[2]=cache. The prior bridge incorrectly
        // followed src[1] and rejected both cache writers.
        ggml_tensor * cache_k=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,64,16);
        ggml_tensor * cache_v=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,64,16);
        ggml_tensor * indices=ggml_new_tensor_1d(ctx,GGML_TYPE_I64,8);
        ggml_tensor * k_values=ggml_view_2d(ctx,model_k,64,8,model_k->nb[2],0);
        ggml_tensor * v_values=ggml_view_2d(ctx,src_v,64,8,src_v->nb[2],0);
        ggml_tensor * store_k=ggml_set_rows(ctx,cache_k,k_values,indices);
        ggml_tensor * store_v=ggml_set_rows(ctx,cache_v,v_values,indices);
        expect(store_k->op==GGML_OP_SET_ROWS &&
               store_k->src[0]==k_values &&
               store_k->src[1]==indices &&
               store_k->src[2]==cache_k,
               "ggml_set_rows(src0=data,src1=indices,src2=cache) changed");
        expect(store_v->src[0]==v_values &&
               store_v->src[1]==indices &&
               store_v->src[2]==cache_v,
               "V SET_ROWS source order changed");
        expect(cache_set_rows_from(store_k,model_k),
               "real K-cache writer must trace through VIEW to post-RoPE Kcur");
        expect(cache_set_rows_from(store_v,src_v),
               "real V-cache writer must trace through VIEW to Vcur");
        expect(!cache_set_rows_from(store_k,src_v) &&
               !cache_set_rows_from(store_v,model_k) &&
               !cache_set_rows_from(indices,model_k),
               "K/V cache source identity must be rejected when mismatched");

        source s;
        s.hidden=128;
        s.tokens=8;
        s.head_dim=64;
        s.q_heads=2;
        s.kv_heads=1;
        s.wq=identity_weight(s.hidden,128);
        s.wk=identity_weight(s.hidden,64);
        s.wv=identity_weight(s.hidden,64);
        s.normalized.resize(size_t(s.hidden*s.tokens)*sizeof(float));
        for(int64_t t=0;t<s.tokens;++t)
            for(int64_t c=0;c<s.hidden;++c) {
                const float value=float(1 + 3*t + c)*0.0078125f;
                std::memcpy(s.normalized.data()+size_t(t*s.hidden+c)*sizeof(float),
                            &value,sizeof(float));
            }
        for(int32_t t=0;t<s.tokens;++t) s.positions.push_back(t);
        std::memcpy(s.q_rope_params.data(),model_q->op_params,sizeof(model_q->op_params));
        std::memcpy(s.k_rope_params.data(),model_k->op_params,sizeof(model_k->op_params));
        validate(s);
        const projected out=run({pc,phone},s);
        check_geometry(out,s);

        // The resident path uses model-buffer-owned QKV tensors, supplied
        // from DIFFERENT GGML contexts than each invocation's activation
        // graph. Verify that no ephemeral weight allocation or upload is
        // required and that the external MatMul leaves execute correctly.
        source resident=s;
        resident.resident_weights=true;
        resident.wq.bytes.clear();
        resident.wk.bytes.clear();
        resident.wv.bytes.clear();
        std::array<ggml_context *,2> model_ctxs{};
        std::array<ggml_backend_buffer_t,2> model_bufs{};
        const std::array<ggml_backend_t,2> devices={pc,phone};
        const std::array<const weight *,3> weights={&s.wq,&s.wk,&s.wv};
        for (int rank=0;rank<2;++rank) {
            const ggml_init_params mp={size_t(32)*1024,nullptr,true};
            model_ctxs[rank]=ggml_init(mp);
            expect(model_ctxs[rank]!=nullptr,"model weight GGML context failed");
            for(int kind=0;kind<3;++kind) {
                const weight & w=*weights[kind];
                resident.physical_weights[rank][kind]=ggml_new_tensor_2d(
                        model_ctxs[rank],w.type,w.in,w.out);
            }
            model_bufs[rank]=ggml_backend_alloc_ctx_tensors(
                    model_ctxs[rank],devices[rank]);
            expect(model_bufs[rank]!=nullptr,"model weight backend buffer failed");
            for(int kind=0;kind<3;++kind) {
                ggml_backend_tensor_set(resident.physical_weights[rank][kind],
                        weights[kind]->bytes.data(),0,weights[kind]->bytes.size());
            }
        }
        validate(resident);
        const projected native_resident=run(devices,resident);
        check_geometry(native_resident,resident);
        for(int rank=0;rank<2;++rank) {
            expect(native_resident.q[rank]==out.q[rank] &&
                   native_resident.k[rank]==out.k[rank] &&
                   native_resident.v[rank]==out.v[rank],
                   "model-resident QKV differs from legacy per-invocation upload");
        }
        source missing=resident;
        missing.physical_weights[1][0]=nullptr;
        bool rejected_missing=false;
        try { validate(missing); }
        catch (const std::invalid_argument &) { rejected_missing=true; }
        expect(rejected_missing,"resident projector must reject missing backend weight");
        for(int rank=1;rank>=0;--rank) {
            ggml_backend_buffer_free(model_bufs[rank]);
            ggml_free(model_ctxs[rank]);
        }

        // Verify a nonuniform 2+1 KV head layout and odd 9-token FIFO.
        // Each SP rank sees all heads but owns only its token interval.
        // The original TP cache rank must receive its head subset for
        // ALL tokens, preserving exact F32 bytes without transposition.
        const int64_t kv_tokens=9, kv_dim=64;
        const std::array<int64_t,2> kv_heads={2,1};
        const int64_t total_kv_heads=kv_heads[0]+kv_heads[1];
        std::array<std::vector<uint8_t>,2> sp_kv_fixture;
        std::vector<float> expected_kv(size_t(kv_tokens*kv_dim*total_kv_heads));
        for(int64_t t=0;t<kv_tokens;++t) {
            const int rank=t<(kv_tokens+1)/2?0:1;
            const int64_t local_t=t-(rank==0?0:(kv_tokens+1)/2);
            const size_t row_size=size_t(kv_dim*total_kv_heads*sizeof(float));
            if (sp_kv_fixture[rank].size() < size_t(local_t+1)*row_size)
                sp_kv_fixture[rank].resize(size_t(local_t+1)*row_size);
            for(int64_t h=0;h<total_kv_heads;++h) {
                for(int64_t d=0;d<kv_dim;++d) {
                    const float value=float(t*10000+h*100+d)*0.000125f;
                    const size_t src_idx=size_t((h*kv_dim+d)*sizeof(float));
                    const size_t full_idx=size_t((t*total_kv_heads+h)*kv_dim+d);
                    expected_kv[full_idx]=value;
                    std::memcpy(sp_kv_fixture[rank].data()+size_t(local_t)*row_size+src_idx,
                                &value,sizeof(float));
                }
            }
        }
        const auto tp_kv=token_to_head_f32(sp_kv_fixture,kv_tokens,kv_dim,kv_heads);
        for (int rank=0;rank<2;++rank) {
            expect(tp_kv[rank].size()==size_t(kv_tokens*kv_heads[rank]*kv_dim)*sizeof(float),
                   "wrong TP KV cache head-shard byte length");
            const int64_t head_start=rank==0?0:kv_heads[0];
            for(int64_t t=0;t<kv_tokens;++t)
                for(int64_t h=0;h<kv_heads[rank];++h)
                    for(int64_t d=0;d<kv_dim;++d) {
                        const float original=expected_kv[size_t((t*total_kv_heads+
                                head_start+h)*kv_dim+d)];
                        const float actual=from_bytes(tp_kv[rank],size_t((t*kv_heads[rank]+h)*kv_dim+d));
                        expect(original==actual,"SP K/V to TP cache byte layout mismatch");
                    }
        }
        bool invalid_kv_partition_rejected=false;
        auto invalid_kv=sp_kv_fixture;
        invalid_kv[1].pop_back();
        try {(void)token_to_head_f32(invalid_kv,kv_tokens,kv_dim,kv_heads);}
        catch (const std::invalid_argument &) {invalid_kv_partition_rejected=true;}
        expect(invalid_kv_partition_rejected,"malformed SP K/V split must be rejected");

        source bad=s;
        bad.positions[4]=0;
        bool rejected=false;
        try { validate(bad); } catch (const std::invalid_argument &) { rejected=true; }
        expect(rejected,"noncontiguous positional ownership must be rejected");
        ggml_free(ctx);
        ggml_backend_free(phone);
        ggml_backend_free(pc);
        std::cout << "GALAXY_NATIVE_SP_QKV_SELFTEST PASS full_weights=1 local_tokens=4+4"
                     " matmul=1 rope=1 direct_sp=1 resident_backend_weights=PASS"
                     " unchanged_outputs=PASS kv_token_to_tp_head=PASS"
                     " kv_set_rows_source_order=PASS" << std::endl;
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "GALAXY_NATIVE_SP_QKV_SELFTEST FAIL " << e.what() << std::endl;
        return 1;
    }
}

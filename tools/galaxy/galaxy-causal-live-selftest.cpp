// Isolated runtime smoke of the SAME live shadow helper called by Meta.
// Q = K = 0, hence causal Attention is the average of the visible V tokens.
// This verifies native device Flash, arbitrary TP GQA head ownership,
// cross-rank token exchange and result comparison without GGUF files.

#include "../../ggml/src/ggml-backend-galaxy-causal-live.h"
#include "ggml-backend.h"

#if defined(GALAXY_CAUSAL_LIVE_HAVE_RPC)
#include "ggml-rpc.h"
#endif

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ggml_galaxy_causal_live;

namespace {

void test(bool value,const std::string & reason) {
    if (!value) throw std::runtime_error(reason);
}
void expect_invalid(const std::function<void()> & fn,const char * label) {
    bool raised=false;
    try { fn(); } catch(const std::invalid_argument &) { raised=true; }
    test(raised,std::string("failed to reject ")+label);
}

struct fixtures {
    request r;
    std::vector<uint8_t> q[2],k[2],v[2],reference[2];

    static void push(std::vector<uint8_t> & bytes,float x) {
        const size_t offset=bytes.size();
        bytes.resize(offset+sizeof(float));
        std::memcpy(bytes.data()+offset,&x,sizeof(x));
    }

    fixtures() {
        r.tokens=17;
        r.head_dim=16;
        r.layer=0;
        r.tile_tokens=5;
        r.max_rmse=0.001;
        r.q.heads={6,6};
        r.k.heads={2,2};
        r.v.heads={2,2};
        r.reference.heads={6,6};
        const int64_t group=3;
        for(int rank=0;rank<2;++rank) {
            const int64_t head_offset=rank==0?0:r.k.heads[0];
            for(int64_t t=0;t<r.tokens;++t) {
                for(int64_t h=0;h<r.q.heads[rank];++h) {
                    for(int64_t d=0;d<r.head_dim;++d) push(q[rank],0.0f);
                }
                for(int64_t h=0;h<r.k.heads[rank];++h) {
                    for(int64_t d=0;d<r.head_dim;++d) push(k[rank],0.0f);
                }
                for(int64_t h=0;h<r.v.heads[rank];++h) {
                    // F16 exact values, different heads and tokens.
                    const float value=float(t+1+2*(head_offset+h))*0.0625f;
                    for(int64_t d=0;d<r.head_dim;++d) push(v[rank],value);
                }
                for(int64_t h=0;h<r.q.heads[rank];++h) {
                    const int64_t global_kv=head_offset+h/group;
                    const float uniform_average=
                            float(t+2+4*global_kv)*0.03125f;
                    // mean_{j=0..t}(j+1+2*kv)*1/16
                    for(int64_t d=0;d<r.head_dim;++d) {
                        push(reference[rank],uniform_average);
                    }
                }
            }
            r.q.bytes[rank]=&q[rank];
            r.k.bytes[rank]=&k[rank];
            r.v.bytes[rank]=&v[rank];
            r.reference.bytes[rank]=&reference[rank];
        }
    }
};

void contract_tests() {
    fixtures fx;
    validate(fx.r);
    test(first_token_error(fx.r)==0.0,"token-zero causal identity");
    request invalid=fx.r;
    invalid.tokens=2;
    expect_invalid([&] { validate(invalid); },"warmup-only tokens");
    invalid=fx.r;
    invalid.q.heads={5,7};
    expect_invalid([&] { validate(invalid); },"GQA group split");
    invalid=fx.r;
    invalid.k.bytes[1]=nullptr;
    expect_invalid([&] { validate(invalid); },"missing remote tensor");
    invalid=fx.r;
    invalid.v.type=GGML_TYPE_F16;
    expect_invalid([&] { validate(invalid); },"wrong source type");
    invalid=fx.r;
    invalid.tile_tokens=0;
    expect_invalid([&] { validate(invalid); },"zero collective tile");

    // Deliberately corrupt the reference at token0. The shadow must never
    // accept it as a first causal prefill (strict numerical contract).
    const float wrong=123.0f;
    std::memcpy(fx.reference[0].data(),&wrong,sizeof(wrong));
    test(first_token_error(fx.r)>1.0,"first-token bad reference detection");
    std::cout << "[GALAXY_CAUSAL_LIVE_TEST] part=CONTRACTS status=PASS\n";
}

void run_device_test(const devices_t & devices) {
    fixtures fx;
    const measurement m=shadow(devices,fx.r);
    test(m.rmse<=fx.r.max_rmse,"native dual-rank attention parity");
    test(m.first_token_rmse<1e-8,"token0 V identity");
    std::cout << "[GALAXY_CAUSAL_LIVE_TEST] tokens=" << fx.r.tokens
              << " tp_heads=6+6 kv_heads=2+2"
              << " rmse=" << m.rmse << " max_abs=" << m.max_abs
              << " cosine=" << m.cosine
              << " tp_sp_ms=" << m.head_to_token_ms
              << " kv_gather_ms=" << m.kv_gather_ms
              << " native_phase_ms=" << m.native_phase_ms
              << " sp_tp_ms=" << m.token_to_head_ms
              << " status=PASS\n";
}

} // namespace

int main(int argc,char ** argv) {
    try {
        std::string pc_name,rpc;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--pc" && i+1<argc) pc_name=argv[++i];
            else if(arg=="--rpc" && i+1<argc) rpc=argv[++i];
            else throw std::invalid_argument("usage: galaxy-causal-live-selftest [--pc CUDA0] [--rpc IP:50052]");
        }
        contract_tests();
        ggml_backend_load_all();
        ggml_backend_t pc=pc_name.empty()?
             ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr):
             ggml_backend_init_by_name(pc_name.c_str(),nullptr);
        test(pc!=nullptr,"PC backend unavailable");
        ggml_backend_t phone=nullptr;
#if defined(GALAXY_CAUSAL_LIVE_HAVE_RPC)
        if(!rpc.empty()) phone=ggml_backend_rpc_init(rpc.c_str(),0);
#else
        test(rpc.empty(),"RPC support not enabled");
#endif
        if(rpc.empty()) phone=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        if(!phone) {
            ggml_backend_free(pc);
            throw std::runtime_error("second backend unavailable");
        }
        try {
            run_device_test({pc,phone});
        } catch (...) {
            ggml_backend_free(phone);
            ggml_backend_free(pc);
            throw;
        }
        ggml_backend_free(phone);
        ggml_backend_free(pc);
        std::cout << "GALAXY_CAUSAL_LIVE_SELFTEST PASS mode="
                  << (rpc.empty()?"CPU_CPU":"PC_RPC") << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "GALAXY_CAUSAL_LIVE_SELFTEST FAIL " << e.what() << "\n";
        return 1;
    }
}

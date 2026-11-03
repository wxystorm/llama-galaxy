// Byte-exact TP-head <-> SP-token exchange tests (CPU+CPU / CUDA+Phone).
// No model graph changes and no GPU-native custom kernel required.

#include "../../ggml/src/ggml-backend-galaxy-layout.h"

#if defined(GALAXY_LAYOUT_HAVE_RPC)
#include "ggml-rpc.h"
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ggml_galaxy_layout;

namespace {

void demand(bool ok, const std::string & text) {
    if (!ok) throw std::runtime_error(text);
}
void expect_reject(const std::function<void()> & fn, const char * label) {
    bool hit = false;
    try { fn(); } catch (const std::invalid_argument &) { hit = true; }
    demand(hit, std::string("did not reject ") + label);
}

struct rank_buffers {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tp_in = nullptr;
    ggml_tensor * sp_out = nullptr;
    ggml_tensor * tp_back = nullptr;
    rank_buffers(ggml_backend_t backend, const exchange & l,
                 ggml_type type, int rank, bool as_3d) {
        const ggml_init_params params = {size_t(1024)*1024,nullptr,true};
        ctx = ggml_init(params);
        demand(ctx != nullptr, "context init failed");
        const int64_t features = total_heads(l)*l.head_dim;
        const int64_t total = total_tokens(l);
        if (as_3d) {
            tp_in = ggml_new_tensor_3d(ctx,type,l.head_dim,l.head_counts[rank],total);
            sp_out = ggml_new_tensor_3d(ctx,type,l.head_dim,total_heads(l),l.token_counts[rank]);
            tp_back = ggml_new_tensor_3d(ctx,type,l.head_dim,l.head_counts[rank],total);
        } else {
            tp_in = ggml_new_tensor_2d(ctx,type,l.head_dim*l.head_counts[rank],total);
            sp_out = ggml_new_tensor_2d(ctx,type,features,l.token_counts[rank]);
            tp_back = ggml_new_tensor_2d(ctx,type,l.head_dim*l.head_counts[rank],total);
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx,backend);
        demand(buffer != nullptr,"backend allocation failed");
    }
    ~rank_buffers() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
    rank_buffers(const rank_buffers &) = delete;
    rank_buffers & operator=(const rank_buffers &) = delete;
};

std::vector<uint8_t> get(const ggml_tensor * tensor) {
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor,bytes.data(),0,bytes.size());
    return bytes;
}

void run_case(const devices_t & dev, ggml_type type,
              exchange l, size_t tile, bool as_3d) {
    rank_buffers a(dev[0],l,type,0,as_3d);
    rank_buffers b(dev[1],l,type,1,as_3d);
    rank_buffers * r[2]={&a,&b};
    const int64_t tokens=total_tokens(l), heads=total_heads(l);
    const size_t elem=ggml_galaxy_collectives::detail::element_bytes(type);
    const size_t width=static_cast<size_t>(heads*l.head_dim);
    const size_t row_bytes=width*elem;

    std::vector<uint8_t> canonical(static_cast<size_t>(tokens)*row_bytes);
    for (int64_t t=0;t<tokens;++t) {
        for (int64_t h=0;h<heads;++h) {
            for (int64_t d=0;d<l.head_dim;++d) {
                // Exactly representable in F16; all comparisons are byte-exact.
                const float value=float(((t*31+h*17+d*3)%199)-99)*0.125f;
                const size_t i=static_cast<size_t>(t)*width+
                        static_cast<size_t>(h*l.head_dim+d);
                ggml_galaxy_collectives::detail::write(type,canonical.data()+i*elem,value);
            }
        }
    }
    std::vector<uint8_t> expected_tp[2];
    std::vector<uint8_t> expected_sp[2];

    int64_t head_start=0,token_start=0;
    for (int rank=0;rank<2;++rank) {
        const size_t local_width=static_cast<size_t>(l.head_counts[rank]*l.head_dim);
        expected_tp[rank].resize(static_cast<size_t>(tokens)*local_width*elem);
        for (int64_t t=0;t<tokens;++t) {
            std::memcpy(expected_tp[rank].data()+static_cast<size_t>(t)*local_width*elem,
                        canonical.data()+static_cast<size_t>(t)*row_bytes+
                                static_cast<size_t>(head_start*l.head_dim)*elem,
                        local_width*elem);
        }
        ggml_backend_tensor_set(r[rank]->tp_in,expected_tp[rank].data(),
                                0,expected_tp[rank].size());
        expected_sp[rank].resize(static_cast<size_t>(l.token_counts[rank])*row_bytes);
        std::memcpy(expected_sp[rank].data(),
                    canonical.data()+static_cast<size_t>(token_start)*row_bytes,
                    expected_sp[rank].size());
        head_start+=l.head_counts[rank];
        token_start+=l.token_counts[rank];
    }

    const input_t tp={a.tp_in,b.tp_in};
    const output_t sp={a.sp_out,b.sp_out};
    head_to_token(dev,tp,sp,l,tile);
    for(int rank=0;rank<2;++rank) {
        demand(get(r[rank]->sp_out)==expected_sp[rank],
                "TP->SP byte mismatch rank="+std::to_string(rank));
    }

    const input_t sp_in={a.sp_out,b.sp_out};
    const output_t tp_dest={a.tp_back,b.tp_back};
    token_to_head(dev,sp_in,tp_dest,l,tile);
    for(int rank=0;rank<2;++rank) {
        demand(get(r[rank]->tp_back)==expected_tp[rank],
                "SP->TP roundtrip mismatch rank="+std::to_string(rank));
    }

    // Reject wrong shape/axis ownership even if byte counts happen to match.
    exchange wrong=l;
    wrong.head_counts[0] += 1;
    expect_reject([&] { head_to_token(dev,tp,sp,wrong,tile); },
                  "wrong TP head ownership");
    expect_reject([&] { token_to_head(dev,sp_in,tp_dest,l,0); },
                  "zero tile");
    expect_reject([&] {
        const output_t alias={a.tp_in,b.sp_out};
        head_to_token(dev,tp,alias,l,tile);
    }, "input/output alias");

    std::cout << "[GALAXY_LAYOUT] type=" << ggml_type_name(type)
              << " heads=" << l.head_counts[0] << "+" << l.head_counts[1]
              << " tokens=" << l.token_counts[0] << "+" << l.token_counts[1]
              << " dim=" << l.head_dim << " tile=" << tile
              << " format=" << (as_3d ? "3D" : "2D")
              << " forward=BYTE_EXACT reverse=BYTE_EXACT status=PASS\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string pc_name,rpc_endpoint;
        for(int i=1;i<argc;++i) {
            const std::string key=argv[i];
            if (key=="--pc" && i+1<argc) pc_name=argv[++i];
            else if(key=="--rpc" && i+1<argc) rpc_endpoint=argv[++i];
            else throw std::invalid_argument("usage: galaxy-layout-selftest [--pc CUDA0] [--rpc IP:50052]");
        }
        ggml_backend_load_all();
        ggml_backend_t pc = pc_name.empty()
            ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr)
            : ggml_backend_init_by_name(pc_name.c_str(),nullptr);
        demand(pc!=nullptr,"PC backend unavailable");
        ggml_backend_t phone=nullptr;
#if defined(GALAXY_LAYOUT_HAVE_RPC)
        if(!rpc_endpoint.empty()) phone=ggml_backend_rpc_init(rpc_endpoint.c_str(),0);
#else
        demand(rpc_endpoint.empty(),"build GGML_RPC=ON for RPC layout tests");
#endif
        if(rpc_endpoint.empty()) phone=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        if(!phone) {
            ggml_backend_free(pc);
            throw std::runtime_error("second backend unavailable");
        }
        try {
            const devices_t dev={pc,phone};
            run_case(dev,GGML_TYPE_F32,{{2,4},{3,4},7},1,true);
            run_case(dev,GGML_TYPE_F32,{{2,4},{3,4},7},2,false);
            run_case(dev,GGML_TYPE_F16,{{4,4},{49,49},128},32,true);
            run_case(dev,GGML_TYPE_F16,{{3,5},{7,9},16},17,false);
            run_case(dev,GGML_TYPE_F32,{{12,12},{49,49},128},64,true);
        } catch (...) {
            ggml_backend_free(phone);
            ggml_backend_free(pc);
            throw;
        }
        ggml_backend_free(phone);
        ggml_backend_free(pc);
        std::cout << "GALAXY_LAYOUT_SELFTEST PASS mode="
                  << (rpc_endpoint.empty() ? "LOCAL" : "PC_RPC") << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "GALAXY_LAYOUT_SELFTEST FAIL " << e.what() << "\n";
        return 1;
    }
}

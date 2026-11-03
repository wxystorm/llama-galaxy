#include "../../ggml/src/ggml-backend-galaxy-causal-runtime.h"

#include "ggml-backend.h"

#if defined(GALAXY_CAUSAL_HAVE_RPC)
#include "ggml-rpc.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ggml_galaxy_causal;

static void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

static void expect_reject(const std::function<void()> & fn, const char * label) {
    bool did_reject = false;
    try { fn(); } catch (const std::invalid_argument &) { did_reject = true; }
    check(did_reject, std::string("expected invalid_argument for ") + label);
}

static size_t index(int64_t t, int64_t h, int64_t d,
        int64_t nheads, int64_t width) {
    return static_cast<size_t>((t*nheads+h)*width+d);
}

static void rotate(std::vector<float> & x, int64_t token, int64_t pos,
        int64_t head, int64_t nheads, int64_t dim) {
    for (int64_t d = 0; d+1 < dim; d += 2) {
        const double theta = static_cast<double>(pos)/
                std::pow(10000.0, static_cast<double>(d)/static_cast<double>(dim));
        const double co = std::cos(theta), si = std::sin(theta);
        const size_t i = index(token,head,d,nheads,dim);
        const double a = x[i], b = x[i+1];
        x[i]   = static_cast<float>(a*co - b*si);
        x[i+1] = static_cast<float>(a*si + b*co);
    }
}

static std::vector<uint8_t> pack(ggml_type dtype, const std::vector<float> & x) {
    const size_t item = ggml_galaxy_collectives::detail::element_bytes(dtype);
    std::vector<uint8_t> bytes(x.size()*item);
    for (size_t i = 0; i < x.size(); ++i) {
        ggml_galaxy_collectives::detail::write(dtype, bytes.data()+i*item, x[i]);
    }
    return bytes;
}

static std::vector<float> unpack(ggml_type dtype, const std::vector<uint8_t> & bytes) {
    const size_t item = ggml_galaxy_collectives::detail::element_bytes(dtype);
    check(bytes.size()%item == 0, "invalid source buffer bytes");
    std::vector<float> x(bytes.size()/item);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = ggml_galaxy_collectives::detail::read(dtype, bytes.data()+i*item);
    }
    return x;
}

static std::vector<uint8_t> read(ggml_tensor * t) {
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    return raw;
}

static void write(ggml_tensor * tensor, const std::vector<uint8_t> & bytes) {
    check(bytes.size() == ggml_nbytes(tensor), "source shape mismatch");
    ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
}

static std::vector<uint8_t> slice(const std::vector<uint8_t> & bytes,
        int64_t start, int64_t count, size_t row_bytes) {
    const size_t first = static_cast<size_t>(start)*row_bytes;
    const size_t last = first+static_cast<size_t>(count)*row_bytes;
    check(last <= bytes.size(), "slice out of bounds");
    return std::vector<uint8_t>(bytes.begin()+first, bytes.begin()+last);
}

struct fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr;
    ggml_tensor * cache_k = nullptr, * cache_v = nullptr;
    ggml_tensor * full_k = nullptr, * full_v = nullptr, * out = nullptr;

    fixture(ggml_backend_t backend, const layout & l, int rank, ggml_type type) {
        const auto info = rank_queries(l, rank);
        const ggml_init_params params = {1024*1024, nullptr, true};
        ctx = ggml_init(params);
        check(ctx != nullptr, "ggml_init failed");
        const int64_t qw = l.query_heads*l.head_dim;
        const int64_t kvw = l.kv_heads*l.head_dim;
        q = ggml_new_tensor_2d(ctx, type, qw, info.count);
        k = ggml_new_tensor_2d(ctx, type, kvw, info.count);
        v = ggml_new_tensor_2d(ctx, type, kvw, info.count);
        if (rank == 0 && l.cached_tokens > 0) {
            cache_k = ggml_new_tensor_2d(ctx, type, kvw, l.cached_tokens);
            cache_v = ggml_new_tensor_2d(ctx, type, kvw, l.cached_tokens);
        }
        full_k = ggml_new_tensor_2d(ctx, type, kvw, total_kv(l));
        full_v = ggml_new_tensor_2d(ctx, type, kvw, total_kv(l));
        out = ggml_new_tensor_2d(ctx, type, qw, info.count);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buffer != nullptr, "tensor allocation failed");
    }
    ~fixture() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
    fixture(const fixture &) = delete;
    fixture & operator=(const fixture &) = delete;
};

static std::vector<float> expected_attention(const layout & l,
        const std::vector<float> & q, const std::vector<float> & k,
        const std::vector<float> & v) {
    const int64_t n_kv = total_kv(l);
    const int64_t qw = l.query_heads*l.head_dim;
    const int64_t vw = l.kv_heads*l.head_dim;
    const double scale = 1.0/std::sqrt(double(l.head_dim));
    std::vector<float> out(static_cast<size_t>(l.new_tokens*qw));
    for (int64_t t = 0; t < l.new_tokens; ++t) {
        // Independent golden mask: query at global position cache+t may
        // read keys <= that position. Do not call runtime visible().
        const int64_t last_visible = l.cached_tokens+t;
        for (int64_t h = 0; h < l.query_heads; ++h) {
            const int64_t kvh = h/(l.query_heads/l.kv_heads);
            std::vector<double> scores(static_cast<size_t>(last_visible+1));
            double maximum = -1.0e300;
            for (int64_t key = 0; key <= last_visible; ++key) {
                double dot = 0;
                for (int64_t d = 0; d < l.head_dim; ++d) {
                    dot += double(q[static_cast<size_t>(t*qw+h*l.head_dim+d)]) *
                           double(k[static_cast<size_t>(key*vw+kvh*l.head_dim+d)]);
                }
                const double score = dot*scale;
                scores[static_cast<size_t>(key)] = score;
                maximum = std::max(maximum, score);
            }
            double den = 0.0;
            for (double score : scores) den += std::exp(score-maximum);
            for (int64_t d = 0; d < l.head_dim; ++d) {
                double acc = 0;
                for (int64_t key = 0; key <= last_visible; ++key) {
                    acc += std::exp(scores[static_cast<size_t>(key)]-maximum)/den *
                            double(v[static_cast<size_t>(key*vw+kvh*l.head_dim+d)]);
                }
                out[static_cast<size_t>(t*qw+h*l.head_dim+d)] = static_cast<float>(acc);
            }
        }
    }
    check(out.size() == static_cast<size_t>(l.new_tokens*qw), "golden shape mismatch");
    return out;
}

static void run_case(const backend_pair & devices, layout shape,
        ggml_type type, size_t tile) {
    const int64_t n_kv = total_kv(shape);
    const int64_t qwidth = shape.query_heads*shape.head_dim;
    const int64_t kvwidth = shape.kv_heads*shape.head_dim;

    std::vector<float> q(static_cast<size_t>(shape.new_tokens*qwidth));
    std::vector<float> k(static_cast<size_t>(n_kv*kvwidth));
    std::vector<float> v(k.size());
    for (int64_t t = 0; t < shape.new_tokens; ++t) {
        for (int64_t h = 0; h < shape.query_heads; ++h) {
            for (int64_t d = 0; d < shape.head_dim; ++d) {
                q[index(t,h,d,shape.query_heads,shape.head_dim)] =
                        static_cast<float>(std::sin(double(t*3+h*11+d*5+1)*0.12));
            }
            rotate(q,t,shape.cached_tokens+t,h,shape.query_heads,shape.head_dim);
        }
    }
    for (int64_t t = 0; t < n_kv; ++t) {
        for (int64_t h = 0; h < shape.kv_heads; ++h) {
            for (int64_t d = 0; d < shape.head_dim; ++d) {
                k[index(t,h,d,shape.kv_heads,shape.head_dim)] =
                        static_cast<float>(std::cos(double(t*7+h*3+d*2+1)*0.14));
                v[index(t,h,d,shape.kv_heads,shape.head_dim)] =
                        static_cast<float>(std::sin(double(t*2+h*7+d*13+4)*0.09));
            }
            rotate(k,t,t,h,shape.kv_heads,shape.head_dim);
        }
    }

    // The F16 golden uses the exact values actually sent to devices, so
    // conversion precision cannot be confused with distributed errors.
    const auto qbytes = pack(type,q);
    const auto kbytes = pack(type,k);
    const auto vbytes = pack(type,v);
    const auto expected = expected_attention(shape, unpack(type,qbytes),
            unpack(type,kbytes), unpack(type,vbytes));
    const auto expected_bytes = pack(type,expected);
    const auto expected_rounded = unpack(type,expected_bytes);

    fixture pc(devices[0],shape,0,type);
    fixture phone(devices[1],shape,1,type);
    fixture * fixtures[] = {&pc,&phone};
    const size_t qrow = static_cast<size_t>(qwidth)*
            ggml_galaxy_collectives::detail::element_bytes(type);
    const size_t kvrow = static_cast<size_t>(kvwidth)*
            ggml_galaxy_collectives::detail::element_bytes(type);
    for (int rank = 0; rank < 2; ++rank) {
        const auto split = rank_queries(shape,rank);
        write(fixtures[rank]->q, slice(qbytes,split.start,split.count,qrow));
        write(fixtures[rank]->k, slice(kbytes,shape.cached_tokens+split.start,split.count,kvrow));
        write(fixtures[rank]->v, slice(vbytes,shape.cached_tokens+split.start,split.count,kvrow));
    }
    if (shape.cached_tokens) {
        write(pc.cache_k, slice(kbytes,0,shape.cached_tokens,kvrow));
        write(pc.cache_v, slice(vbytes,0,shape.cached_tokens,kvrow));
    }
    const input_pair local_q = {pc.q,phone.q};
    const input_pair new_k = {pc.k,phone.k};
    const input_pair new_v = {pc.v,phone.v};
    const output_pair full_k = {pc.full_k,phone.full_k};
    const output_pair full_v = {pc.full_v,phone.full_v};
    const output_pair output = {pc.out,phone.out};
    host_staged_attention(devices,shape,local_q,new_k,new_v,
            pc.cache_k,pc.cache_v,full_k,full_v,output,tile);

    for (int rank = 0; rank < 2; ++rank) {
        ggml_backend_synchronize(devices[rank]);
        check(read(fixtures[rank]->full_k) == kbytes,
                "full Key was not reconstructed exactly on rank "+std::to_string(rank));
        check(read(fixtures[rank]->full_v) == vbytes,
                "full Value was not reconstructed exactly on rank "+std::to_string(rank));
    }
    double max_abs = 0.0;
    for (int rank = 0; rank < 2; ++rank) {
        const auto split = rank_queries(shape,rank);
        const auto actual = unpack(type,read(fixtures[rank]->out));
        check(actual.size() == static_cast<size_t>(split.count*qwidth),"result size");
        for (int64_t t = 0; t < split.count; ++t) {
            for (int64_t d = 0; d < qwidth; ++d) {
                const auto i = static_cast<size_t>(t*qwidth+d);
                const auto reference = static_cast<size_t>((split.start+t)*qwidth+d);
                const double err = std::abs(double(actual[i])-double(expected_rounded[reference]));
                max_abs = std::max(max_abs,err);
                check(err <= (type == GGML_TYPE_F16 ? 0.0025 : 0.00001),
                        "local-Query attention vs global reference failed");
            }
        }
    }

    expect_reject([&] { host_staged_attention(devices,shape,local_q,new_k,new_v,
            pc.cache_k,pc.cache_v,full_k,full_v,output,0); }, "zero tile");
    expect_reject([&] { host_staged_attention(devices,shape,local_q,new_k,new_v,
            pc.cache_k,pc.cache_v,full_k,full_v,local_q,1); }, "output aliases input");
    if (shape.cached_tokens) {
        expect_reject([&] { host_staged_attention(devices,shape,local_q,new_k,new_v,
                nullptr,nullptr,full_k,full_v,output,1); }, "missing cache");
    }
    std::cout << "[GALAXY_CAUSAL_BACKEND] cached=" << shape.cached_tokens
              << " new=" << shape.new_tokens << " split=" << shape.rank0_queries
              << "," << shape.new_tokens-shape.rank0_queries << " heads="
              << shape.query_heads << "/" << shape.kv_heads << " type="
              << (type==GGML_TYPE_F32 ? "F32" : "F16")
              << " tile=" << tile << " max_abs=" << max_abs
              << " status=PASS\n";
}

int main(int argc, char ** argv) {
    try {
        std::string endpoint,pc_name;
        for (int i = 1; i < argc; ++i) {
            const std::string opt = argv[i];
            if (opt == "--rpc" && i+1 < argc) endpoint = argv[++i];
            else if (opt == "--pc" && i+1 < argc) pc_name = argv[++i];
            else throw std::invalid_argument(
                    "usage: galaxy-causal-backend-selftest [--pc CUDA0] [--rpc IP:50052]");
        }
        ggml_backend_load_all();
        ggml_backend_t pc = pc_name.empty()
                ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr)
                : ggml_backend_init_by_name(pc_name.c_str(),nullptr);
        check(pc!=nullptr,"PC backend unavailable");
        ggml_backend_t phone = nullptr;
#if defined(GALAXY_CAUSAL_HAVE_RPC)
        if (!endpoint.empty()) phone=ggml_backend_rpc_init(endpoint.c_str(),0);
#else
        check(endpoint.empty(),"rebuild with GGML_RPC=ON for Phone RPC testing");
#endif
        if (endpoint.empty()) {
            phone=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        }
        check(phone!=nullptr,"Phone backend unavailable");
        try {
            const backend_pair devices={pc,phone};
            run_case(devices,{0,9,4,24,8,8},GGML_TYPE_F32,2);
            run_case(devices,{13,11,7,24,8,8},GGML_TYPE_F32,4);
            run_case(devices,{31,17,10,8,8,16},GGML_TYPE_F16,5);
            run_case(devices,{0,2,1,16,8,8},GGML_TYPE_F16,32);
            const layout decode={64,1,1,24,8,8};
            expect_reject([&] { host_staged_attention(devices,decode,
                    {},{},{},nullptr,nullptr,{},{},{},32); }, "decode fallback");
        } catch (...) {
            ggml_backend_free(phone);
            ggml_backend_free(pc);
            throw;
        }
        ggml_backend_free(phone);
        ggml_backend_free(pc);
        std::cout << "GALAXY_CAUSAL_BACKEND_SELFTEST PASS mode="
                  << (endpoint.empty()?"LOCAL":"PC_RPC") << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "GALAXY_CAUSAL_BACKEND_SELFTEST FAIL " << e.what() << "\n";
        return 1;
    }
}

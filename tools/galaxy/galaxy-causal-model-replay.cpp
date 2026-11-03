// Real-model causal SP Attention replay. This does NOT mutate model inference.
//
// The Llama TP graph exports post-RoPE Q/K, projected V, and its own
// pre-Wo Attention output. Repartition Q/K/V by token, gather the real
// values across PC+Phone, execute local-Q/global-KV causal Attention,
// and compare with the original full-token Attention result.
//
// A numerical difference is reported, not silently accepted as parity:
// the original graph may have an F16 KV cache / Flash Attention, whereas
// the correctness bridge explicitly computes F32 host softmax.

#include "../../ggml/src/ggml-backend-galaxy-causal-runtime.h"

#include "ggml-backend.h"
#if defined(GALAXY_CAUSAL_HAVE_RPC)
#include "ggml-rpc.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ggml_galaxy_causal;

namespace {

void require(bool ok, const std::string & description) {
    if (!ok) throw std::runtime_error(description);
}

int64_t positive_integer(const std::string & text, const std::string & key, bool allow_zero=false) {
    require(!text.empty(), key + " requires an integer");
    int64_t result = 0;
    for (char c : text) {
        require(c >= '0' && c <= '9', key + " requires decimal digits only");
        int digit = c - '0';
        require(result <= (std::numeric_limits<int64_t>::max()-digit)/10,
                key + " overflows int64");
        result = result*10 + digit;
    }
    require(result > 0 || (allow_zero && result == 0), key + " requires a positive integer");
    return result;
}

std::vector<float> read_glg1(const std::filesystem::path & path, int64_t expected) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good(), "Cannot open probe: " + path.string());
    const std::streamoff bytes = input.tellg();
    require(expected > 0 && expected <= std::numeric_limits<int32_t>::max(),
            "Probe expected size exceeds GLG1 format");
    require(bytes == 8 + expected*static_cast<int64_t>(sizeof(float)),
            "Probe shape/count mismatch: " + path.string());
    input.seekg(0, std::ios::beg);
    char magic[4];
    int32_t count = 0;
    input.read(magic, 4);
    input.read(reinterpret_cast<char *>(&count), sizeof(count));
    require(input.good() && std::memcmp(magic, "GLG1", 4) == 0 && count == expected,
            "Invalid probe header/count: " + path.string());
    std::vector<float> values(static_cast<size_t>(expected));
    input.read(reinterpret_cast<char *>(values.data()), expected*sizeof(float));
    require(input.good(), "Truncated probe: " + path.string());
    for (float v : values) {
        require(std::isfinite(v), "Non-finite probe element: " + path.string());
    }
    return values;
}

struct buffers {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t allocation = nullptr;
    ggml_tensor * query = nullptr;
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
    ggml_tensor * complete_key = nullptr;
    ggml_tensor * complete_value = nullptr;
    ggml_tensor * output = nullptr;
    buffers(ggml_backend_t backend, const layout & l, int rank) {
        const auto local = rank_queries(l,rank);
        const ggml_init_params params = {1024*1024, nullptr, true};
        ctx = ggml_init(params);
        require(ctx != nullptr, "GGML context creation failed");
        const int64_t qw = l.query_heads*l.head_dim;
        const int64_t kvw = l.kv_heads*l.head_dim;
        query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, qw, local.count);
        key = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kvw, local.count);
        value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kvw, local.count);
        complete_key = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kvw, l.new_tokens);
        complete_value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kvw, l.new_tokens);
        output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, qw, local.count);
        allocation = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(allocation != nullptr, "GGML backend buffer allocation failed");
    }
    ~buffers() {
        if (allocation) ggml_backend_buffer_free(allocation);
        if (ctx) ggml_free(ctx);
    }
    buffers(const buffers &) = delete;
    buffers & operator=(const buffers &) = delete;
};

void upload_tokens(ggml_tensor * tensor, const std::vector<float> & all,
        int64_t global_start, int64_t local_count, int64_t width) {
    require(ggml_nbytes(tensor) == static_cast<size_t>(local_count*width*sizeof(float)),
            "Local probe tensor shape mismatch");
    ggml_backend_tensor_set(tensor, all.data()+global_start*width,
            0, ggml_nbytes(tensor));
}

std::vector<float> read_tensor(ggml_tensor * tensor) {
    require(tensor->type == GGML_TYPE_F32, "F32 tensor required");
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    return values;
}

struct stats {
    double squared = 0.0;
    double maximum = 0.0;
    double norm_ref = 0.0;
    double norm_actual = 0.0;
    double dot = 0.0;
    int64_t values = 0;
    int64_t max_token = 0;
    int64_t max_channel = 0;
    double max_reference = 0.0;
    double max_candidate = 0.0;
    void add(float x, float y, int64_t token, int64_t channel) {
        const double delta = static_cast<double>(x)-static_cast<double>(y);
        squared += delta*delta;
        dot += static_cast<double>(x)*y;
        norm_ref += static_cast<double>(x)*x;
        norm_actual += static_cast<double>(y)*y;
        if (std::abs(delta) > maximum) {
            maximum = std::abs(delta);
            max_token = token;
            max_channel = channel;
            max_reference = x;
            max_candidate = y;
        }
        values++;
    }
    double rmse() const { return std::sqrt(squared/static_cast<double>(values)); }
    double cosine() const { return dot/std::sqrt(norm_ref*norm_actual); }
};

struct settings {
    std::filesystem::path probe_dir;
    std::string pc;
    std::string rpc;
    int64_t layer = 0;
    int64_t tokens = 0;
    int64_t heads = 24;
    int64_t kv_heads = 8;
    int64_t head_dim = 128;
    int64_t rank0 = 0;
    int64_t tile = 32;
};

settings parse(int argc, char ** argv) {
    settings config;
    for (int i = 1; i < argc; ++i) {
        const std::string key=argv[i];
        require(i+1 < argc, "Missing value for " + key);
        const std::string value=argv[++i];
        if (key == "--probe-dir") config.probe_dir=value;
        else if (key == "--pc") config.pc=value;
        else if (key == "--rpc") config.rpc=value;
        else if (key == "--layer") config.layer=positive_integer(value,key,true);
        else if (key == "--tokens") config.tokens=positive_integer(value,key);
        else if (key == "--heads") config.heads=positive_integer(value,key);
        else if (key == "--kv-heads") config.kv_heads=positive_integer(value,key);
        else if (key == "--head-dim") config.head_dim=positive_integer(value,key);
        else if (key == "--rank0") config.rank0=positive_integer(value,key);
        else if (key == "--tile") config.tile=positive_integer(value,key);
        else throw std::invalid_argument("Unknown flag: " + key);
    }
    require(!config.probe_dir.empty(), "--probe-dir is required");
    require(config.tokens >= 2 && config.rank0 > 0 && config.rank0 < config.tokens,
            "Need --tokens >= 2 and --rank0 in 1..tokens-1");
    return config;
}

void run(const settings & c, const backend_pair & devices) {
    const layout l{0,c.tokens,c.rank0,c.heads,c.kv_heads,c.head_dim};
    validate(l);
    const int64_t qw = runtime_detail::checked_width(l.query_heads,l.head_dim);
    const int64_t kvw = runtime_detail::checked_width(l.kv_heads,l.head_dim);
    const auto q = read_glg1(c.probe_dir/("Qcur-"+std::to_string(c.layer)+".f32"), qw*l.new_tokens);
    const auto k = read_glg1(c.probe_dir/("Kcur-"+std::to_string(c.layer)+".f32"), kvw*l.new_tokens);
    const auto v = read_glg1(c.probe_dir/("Vcur-"+std::to_string(c.layer)+".f32"), kvw*l.new_tokens);
    const auto model = read_glg1(c.probe_dir/("kqv_out-"+std::to_string(c.layer)+".f32"), qw*l.new_tokens);

    buffers rank0(devices[0],l,0);
    buffers rank1(devices[1],l,1);
    buffers * ranks[2]={&rank0,&rank1};
    for (int rank = 0; rank < 2; ++rank) {
        const auto part=rank_queries(l,rank);
        upload_tokens(ranks[rank]->query,q,part.start,part.count,qw);
        upload_tokens(ranks[rank]->key,k,part.start,part.count,kvw);
        upload_tokens(ranks[rank]->value,v,part.start,part.count,kvw);
    }
    const input_pair queries={rank0.query,rank1.query};
    const input_pair keys={rank0.key,rank1.key};
    const input_pair values={rank0.value,rank1.value};
    const output_pair all_keys={rank0.complete_key,rank1.complete_key};
    const output_pair all_values={rank0.complete_value,rank1.complete_value};
    const output_pair outputs={rank0.output,rank1.output};

    host_staged_attention(devices,l,queries,keys,values,nullptr,nullptr,
            all_keys,all_values,outputs,static_cast<size_t>(c.tile));
    stats entire;
    for (int rank = 0; rank < 2; ++rank) {
        ggml_backend_synchronize(devices[rank]);
        const auto full_key=read_tensor(ranks[rank]->complete_key);
        const auto full_value=read_tensor(ranks[rank]->complete_value);
        require(full_key == k && full_value == v,
                "Real-model complete K/V does not match the exported GGUF QKV");
        const auto actual=read_tensor(ranks[rank]->output);
        const auto part=rank_queries(l,rank);
        stats per_rank;
        for (int64_t t = 0; t < part.count; ++t) {
            for (int64_t f=0;f<qw;++f) {
                const size_t local_index=static_cast<size_t>(t*qw+f);
                const size_t global_index=static_cast<size_t>((part.start+t)*qw+f);
                per_rank.add(model[global_index],actual[local_index],part.start+t,f);
                entire.add(model[global_index],actual[local_index],part.start+t,f);
            }
        }
        std::cout << "[GALAXY_REAL_REPLAY] rank=" << rank
                  << " rmse=" << per_rank.rmse()
                  << " max_abs=" << per_rank.maximum
                  << " cosine=" << per_rank.cosine()
                  << " status=KV_EXACT_ATTENTION_MEASURED\n";
    }
    std::cout << "[GALAXY_REAL_REPLAY] layer=" << c.layer
              << " tokens=" << c.tokens << " split=" << c.rank0 << ","
              << c.tokens-c.rank0 << " heads=" << c.heads
              << " kv_heads=" << c.kv_heads << " head_dim=" << c.head_dim
              << " rmse=" << entire.rmse()
              << " max_abs=" << entire.maximum
              << " worst_token=" << entire.max_token
              << " worst_channel=" << entire.max_channel
              << " ref=" << entire.max_reference
              << " replay=" << entire.max_candidate
              << " cosine=" << entire.cosine()
              << " transport=PASS parity=MEASURED\n";
}

} // anonymous namespace

int main(int argc, char ** argv) {
    try {
        const settings options=parse(argc,argv);
        ggml_backend_load_all();
        ggml_backend_t pc = options.pc.empty()
                ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr)
                : ggml_backend_init_by_name(options.pc.c_str(),nullptr);
        require(pc!=nullptr, "PC backend unavailable");
        ggml_backend_t second=nullptr;
#if defined(GALAXY_CAUSAL_HAVE_RPC)
        if (!options.rpc.empty()) {
            second=ggml_backend_rpc_init(options.rpc.c_str(),0);
        }
#else
        require(options.rpc.empty(), "Rebuild with GGML_RPC=ON for RPC replay");
#endif
        if (options.rpc.empty()) {
            second=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        }
        if (!second) {
            ggml_backend_free(pc);
            throw std::runtime_error("Second backend unavailable");
        }
        try {
            run(options,{pc,second});
        } catch (...) {
            ggml_backend_free(second);
            ggml_backend_free(pc);
            throw;
        }
        ggml_backend_free(second);
        ggml_backend_free(pc);
        std::cout << "GALAXY_CAUSAL_REAL_REPLAY PASS transport_verified numerical_parity_not_asserted\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "GALAXY_CAUSAL_REAL_REPLAY FAIL: " << e.what() << "\n";
        return 1;
    }
}

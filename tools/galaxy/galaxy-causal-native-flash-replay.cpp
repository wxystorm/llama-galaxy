// Native device-kernel step toward sequence-parallel causal Attention.
//
// Uses real post-RoPE GGUF Q/K/V captures, token-shards Q across PC/Phone,
// AllGathers F16 K/V to BOTH backends and executes each rank's own GGML
// FLASH_ATTN_EXT kernel. This is not the production transformer graph:
// projections are sourced from GLG1 and the first prefill has no KV history.
//
// The old host-softmax replay is kept as an independent numerical baseline.
// Q is F32; K/V are rounded to F16, like the original Flash path.
// Mask rows use *global* Query positions, never rank-local offsets.

#include "ggml.h"
#include "ggml-backend.h"
#include "../../ggml/src/ggml-backend-galaxy-collectives.h"
#if defined(GALAXY_CAUSAL_HAVE_RPC)
#include "ggml-rpc.h"
#endif

#include <algorithm>
#include <chrono>
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

namespace {

using clock_type = std::chrono::steady_clock;
using namespace ggml_galaxy_collectives;

void demand(bool ok, const std::string & reason) {
    if (!ok) throw std::runtime_error(reason);
}

int64_t parse_uint(const std::string & src, const std::string & name, bool zero_ok = false) {
    demand(!src.empty(), "missing value for " + name);
    int64_t value = 0;
    for (char ch : src) {
        demand(ch >= '0' && ch <= '9', name + " requires nonnegative decimal digits");
        const int digit = ch-'0';
        demand(value <= (std::numeric_limits<int64_t>::max()-digit)/10,
                name + " overflows int64");
        value = value*10 + digit;
    }
    demand(value > 0 || (zero_ok && value == 0), name + " must be positive");
    return value;
}

struct options {
    std::filesystem::path probe_dir;
    std::string pc;
    std::string rpc;
    int64_t layer = 0;
    int64_t tokens = 0;
    int64_t rank0 = 0;
    int64_t heads = 24;
    int64_t kv_heads = 8;
    int64_t dim = 128;
    int64_t tile = 32;
    double require_rmse = -1.0;
};

options parse_args(int argc, char ** argv) {
    options c;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        demand(i + 1 < argc, "missing argument for " + key);
        const std::string val = argv[++i];
        if (key == "--probe-dir") c.probe_dir = val;
        else if (key == "--pc") c.pc = val;
        else if (key == "--rpc") c.rpc = val;
        else if (key == "--layer") c.layer = parse_uint(val,key,true);
        else if (key == "--tokens") c.tokens = parse_uint(val,key);
        else if (key == "--rank0") c.rank0 = parse_uint(val,key);
        else if (key == "--heads") c.heads = parse_uint(val,key);
        else if (key == "--kv-heads") c.kv_heads = parse_uint(val,key);
        else if (key == "--head-dim") c.dim = parse_uint(val,key);
        else if (key == "--tile") c.tile = parse_uint(val,key);
        else if (key == "--require-rmse") {
            // The explicit requirement is optional. By default the program
            // reports parity measurements rather than claiming PASS.
            try {
                size_t consumed = 0;
                c.require_rmse = std::stod(val, &consumed);
                demand(consumed == val.size() && std::isfinite(c.require_rmse)
                        && c.require_rmse >= 0, "invalid --require-rmse");
            } catch (const std::exception &) {
                throw std::invalid_argument("invalid --require-rmse");
            }
        } else {
            throw std::invalid_argument("unknown flag " + key);
        }
    }
    demand(!c.probe_dir.empty(), "--probe-dir required");
    demand(c.tokens >= 2 && c.rank0 > 0 && c.rank0 < c.tokens,
            "nonempty Query shard on both devices required");
    demand(c.heads > 0 && c.kv_heads > 0 && c.heads % c.kv_heads == 0,
            "query heads must be divisible by KV heads");
    demand(c.dim > 0 && c.heads <= INT64_MAX/c.dim && c.kv_heads <= INT64_MAX/c.dim,
            "invalid head dimension");
    demand(c.tile >= 1 && c.tile <= 1024, "--tile must be 1..1024");
    demand(c.tokens <= 4096, "native replay is limited to 4096-token prefills");
    demand(c.tokens <= INT32_MAX/(c.heads*c.dim) &&
           c.tokens <= INT32_MAX/(c.kv_heads*c.dim),
           "GLG1 snapshot exceeds 32-bit element count");
    return c;
}

std::vector<float> read_probe(const std::filesystem::path & path, int64_t elements) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    demand(f.good(), "cannot open " + path.string());
    demand(elements > 0 && elements <= INT32_MAX, "invalid probe elements");
    demand(f.tellg() == std::streamoff(8 + elements*int64_t(sizeof(float))),
           "probe file size mismatch " + path.string());
    f.seekg(0);
    char magic[4];
    int32_t n = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    demand(f.good() && std::memcmp(magic, "GLG1", 4) == 0 && n == elements,
           "invalid GLG1 header/count " + path.string());
    std::vector<float> data(static_cast<size_t>(elements));
    f.read(reinterpret_cast<char *>(data.data()), elements*sizeof(float));
    demand(f.good(), "truncated GLG1 " + path.string());
    for (float v : data) demand(std::isfinite(v), "nonfinite GLG1 input " + path.string());
    return data;
}

std::vector<ggml_fp16_t> to_half(const std::vector<float> & input) {
    std::vector<ggml_fp16_t> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = ggml_fp32_to_fp16(input[i]);
    }
    return result;
}

struct comparison {
    double squared = 0;
    double dot = 0;
    double norm_reference = 0;
    double norm_actual = 0;
    double maximum = 0;
    int64_t count = 0, worst_token = 0, worst_channel = 0;

    void add(float expected, float actual, int64_t token, int64_t channel) {
        demand(std::isfinite(actual), "nonfinite native Flash Attention output");
        const double a = expected, b = actual, delta = a-b;
        squared += delta*delta;
        dot += a*b;
        norm_reference += a*a;
        norm_actual += b*b;
        if (std::abs(delta) > maximum) {
            maximum = std::abs(delta);
            worst_token = token;
            worst_channel = channel;
        }
        count++;
    }
    double rmse() const { return std::sqrt(squared/double(count)); }
    double cosine() const {
        return norm_reference > 0 && norm_actual > 0
             ? dot/std::sqrt(norm_reference*norm_actual)
             : std::numeric_limits<double>::quiet_NaN();
    }
};

struct native_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k_local = nullptr;
    ggml_tensor * v_local = nullptr;
    ggml_tensor * k_full = nullptr;  // contiguous 2D view: first actual tokens
    ggml_tensor * v_full = nullptr;
    ggml_tensor * k_storage = nullptr; // padded to match llama.cpp KV cache
    ggml_tensor * v_storage = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * attention = nullptr;
    ggml_tensor * output = nullptr;
    int64_t q_begin = 0;
    int64_t q_count = 0;
    int64_t mask_width = 0;
    int64_t mask_rows = 0;
    int64_t kv_padded = 0;

    native_graph(ggml_backend_t backend, const options & c,
                 int64_t begin, int64_t count) : q_begin(begin), q_count(count) {
        // One separate GGML graph per physical backend, not the Meta wrapper.
        // Input tensors are device-resident and FLASH_ATTN_EXT executes there.
        const ggml_init_params p = {size_t(4)*1024*1024, nullptr, true};
        ctx = ggml_init(p);
        demand(ctx != nullptr, "ggml_init failed");
        const int64_t qw = c.heads*c.dim;
        const int64_t kvw = c.kv_heads*c.dim;

        q       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.dim, c.heads, count);
        k_local = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kvw, count);
        v_local = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kvw, count);
        // Real llama.cpp Flash Attention reads a padded KV-cache length.
        // Gather into a contiguous view of its populated first tokens;
        // the remaining capacity is zero-filled and masked to -infinity.
        kv_padded = GGML_PAD(c.tokens, 256);
        k_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kvw, kv_padded);
        v_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kvw, kv_padded);
        const size_t kv_row_bytes = size_t(kvw)*sizeof(ggml_fp16_t);
        k_full = ggml_view_2d(ctx,k_storage,kvw,c.tokens,kv_row_bytes,0);
        v_full = ggml_view_2d(ctx,v_storage,kvw,c.tokens,kv_row_bytes,0);
        ggml_set_input(q);
        ggml_set_input(k_storage);
        ggml_set_input(v_storage);

        // CUDA/Phone Flash Attention masks are F16, with padded rows.
        mask_width = kv_padded;
        mask_rows  = GGML_PAD(count, 32);
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, mask_width, mask_rows);
        ggml_set_input(mask);

        ggml_tensor * query = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor * key = ggml_reshape_3d(ctx, k_storage, c.dim, c.kv_heads, kv_padded);
        key = ggml_permute(ctx, key, 0, 2, 1, 3);
        ggml_tensor * value = ggml_reshape_3d(ctx, v_storage, c.dim, c.kv_heads, kv_padded);
        value = ggml_permute(ctx, value, 0, 2, 1, 3);
        // For Flash Attention with a non-transposed contiguous V cache,
        // Llama's v_trans is FALSE before permute, so the kernel expects
        // [head_dim, all_key_tokens, kv_heads]. Do not transpose to
        // [all_key_tokens, head_dim, kv_heads] (that is the other cache path).

        attention = ggml_flash_attn_ext(ctx, query, key, value, mask,
                                        1.0f/std::sqrt(float(c.dim)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attention, GGML_PREC_F32);
        output = ggml_reshape_2d(ctx, attention, qw, count);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, output);

        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        demand(buffer != nullptr, "failed to allocate native Flash graph");
        demand(ggml_nbytes(k_local) == size_t(count*kvw*sizeof(ggml_fp16_t)),
                "invalid local K layout");
        demand(ggml_nbytes(output) == size_t(count*qw*sizeof(float)),
                "invalid native output layout");
    }

    void upload(ggml_backend_t backend, const options & c,
                const std::vector<float> & all_q,
                const std::vector<ggml_fp16_t> & all_k,
                const std::vector<ggml_fp16_t> & all_v) const {
        const size_t qw = static_cast<size_t>(c.heads*c.dim);
        const size_t kvw = static_cast<size_t>(c.kv_heads*c.dim);
        // RPC buffers (and some other ggml backends) have no
        // buffer->iface.memset_tensor callback. Backend tensor_set is
        // supported on both CUDA and RPC, so zero padded KV capacity
        // with a host-owned F16 buffer before gathering actual K/V.
        // Keep the host allocation alive through the backend fence.
        const size_t padded_elements = static_cast<size_t>(kv_padded)*kvw;
        std::vector<ggml_fp16_t> zero_padded_kv(padded_elements, ggml_fp16_t(0));
        demand(ggml_nbytes(k_storage) == zero_padded_kv.size()*sizeof(ggml_fp16_t)
                && ggml_nbytes(v_storage) == ggml_nbytes(k_storage),
                "padded K/V allocation mismatch");
        ggml_backend_tensor_set(k_storage, zero_padded_kv.data(), 0, ggml_nbytes(k_storage));
        ggml_backend_tensor_set(v_storage, zero_padded_kv.data(), 0, ggml_nbytes(v_storage));
        ggml_backend_tensor_set(q, all_q.data()+q_begin*qw, 0, ggml_nbytes(q));
        ggml_backend_tensor_set(k_local, all_k.data()+q_begin*kvw,
                                0, ggml_nbytes(k_local));
        ggml_backend_tensor_set(v_local, all_v.data()+q_begin*kvw,
                                0, ggml_nbytes(v_local));

        std::vector<ggml_fp16_t> causal(
                static_cast<size_t>(mask_width*mask_rows),
                ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t local_q = 0; local_q < q_count; ++local_q) {
            const int64_t absolute_q = q_begin + local_q;
            for (int64_t key = 0; key <= absolute_q; ++key) {
                causal[static_cast<size_t>(local_q*mask_width+key)] =
                        ggml_fp32_to_fp16(0.0f);
            }
        }
        ggml_backend_tensor_set(mask, causal.data(), 0, ggml_nbytes(mask));
        ggml_backend_synchronize(backend);
    }

    std::vector<float> read_result(ggml_backend_t backend) const {
        ggml_backend_synchronize(backend);
        std::vector<float> data(static_cast<size_t>(ggml_nelements(output)));
        ggml_backend_tensor_get(output, data.data(), 0, ggml_nbytes(output));
        return data;
    }

    ~native_graph() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
    native_graph(const native_graph &) = delete;
    native_graph & operator=(const native_graph &) = delete;
};

double elapsed_ms(clock_type::time_point begin) {
    return std::chrono::duration<double,std::milli>(clock_type::now()-begin).count();
}

void execute(const options & c, const devices_t & backends) {
    const int64_t qw = c.heads*c.dim, kvw = c.kv_heads*c.dim;
    const auto q = read_probe(c.probe_dir/("Qcur-"+std::to_string(c.layer)+".f32"), c.tokens*qw);
    const auto k = read_probe(c.probe_dir/("Kcur-"+std::to_string(c.layer)+".f32"), c.tokens*kvw);
    const auto v = read_probe(c.probe_dir/("Vcur-"+std::to_string(c.layer)+".f32"), c.tokens*kvw);
    const auto reference = read_probe(c.probe_dir/("kqv_out-"+std::to_string(c.layer)+".f32"),
                                      c.tokens*qw);
    const auto k_half = to_half(k), v_half = to_half(v);

    native_graph pc(backends[0],c,0,c.rank0);
    native_graph phone(backends[1],c,c.rank0,c.tokens-c.rank0);
    native_graph * ranks[2] = {&pc,&phone};
    for (int i = 0; i < 2; ++i) {
        ranks[i]->upload(backends[i],c,q,k_half,v_half);
    }

    const input_t local_k = {pc.k_local,phone.k_local};
    const input_t local_v = {pc.v_local,phone.v_local};
    const output_t full_k = {pc.k_full,phone.k_full};
    const output_t full_v = {pc.v_full,phone.v_full};
    const counts_t counts = {pc.q_count,phone.q_count};

    auto t = clock_type::now();
    all_gather_attention_kv(backends, local_k, local_v, full_k, full_v, counts,
                            static_cast<size_t>(c.tile));
    const double gather_ms = elapsed_ms(t);

    // Exact byte-level verification of all gathered F16 Key/Value buffers.
    // This is an intentional diagnostic readback and NOT an optimized path.
    for (int rank = 0; rank < 2; ++rank) {
        ggml_backend_synchronize(backends[rank]);
        std::vector<ggml_fp16_t> tmp(k_half.size());
        ggml_backend_tensor_get(full_k[rank],tmp.data(),0,ggml_nbytes(full_k[rank]));
        demand(tmp == k_half, "Key AllGather reconstruction mismatch on rank "+std::to_string(rank));
        ggml_backend_tensor_get(full_v[rank],tmp.data(),0,ggml_nbytes(full_v[rank]));
        demand(tmp == v_half, "Value AllGather reconstruction mismatch on rank "+std::to_string(rank));
    }
    const double verify_ms = elapsed_ms(t)-gather_ms;

    comparison total;
    double compute_ms[2] = {0,0};
    for (int rank = 0; rank < 2; ++rank) {
        t = clock_type::now();
        const auto status = ggml_backend_graph_compute(backends[rank], ranks[rank]->graph);
        demand(status == GGML_STATUS_SUCCESS,
                "FLASH_ATTN_EXT graph execution failed on rank "+std::to_string(rank));
        compute_ms[rank] = elapsed_ms(t);
        const auto result = ranks[rank]->read_result(backends[rank]);
        const int64_t local = ranks[rank]->q_count;
        const int64_t begin = ranks[rank]->q_begin;
        demand(result.size() == static_cast<size_t>(qw*local),"native result shape");
        comparison each;
        for (int64_t ti = 0; ti < local; ++ti) {
            for (int64_t d = 0; d < qw; ++d) {
                const size_t src = static_cast<size_t>(ti*qw+d);
                const size_t dest = static_cast<size_t>((begin+ti)*qw+d);
                each.add(reference[dest],result[src],begin+ti,d);
                total.add(reference[dest],result[src],begin+ti,d);
            }
        }
        std::cout << "[GALAXY_CAUSAL_NATIVE] rank=" << rank
                  << " q_range=" << begin << ".." << begin+local-1
                  << " kernel=FLASH_ATTN_EXT"
                  << " compute_ms=" << compute_ms[rank]
                  << " rmse=" << each.rmse()
                  << " max_abs=" << each.maximum
                  << " cosine=" << each.cosine()
                  << " status=MEASURED\n";
    }
    std::cout << "[GALAXY_CAUSAL_NATIVE] layer=" << c.layer
              << " tokens=" << c.tokens
              << " query_split=" << c.rank0 << "," << c.tokens-c.rank0
              << " head_split=FULL_ON_BOTH"
              << " K_V=F16_ALL_GATHER"
              << " mask=GLOBAL_CAUSAL"
              << " gather_ms=" << gather_ms
              << " verify_ms=" << verify_ms
              << " rmse=" << total.rmse()
              << " max_abs=" << total.maximum
              << " cosine=" << total.cosine()
              << " worst_token=" << total.worst_token
              << " worst_channel=" << total.worst_channel
              << " transport=PASS"
              << " parity=" << (c.require_rmse >= 0 ? "THRESHOLD_CHECKED" : "MEASURED")
              << "\n";
    if (c.require_rmse >= 0) {
        demand(total.rmse() <= c.require_rmse,
                "native Flash Attention RMSE exceeds explicit threshold");
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options c = parse_args(argc,argv);
        ggml_backend_load_all();
        ggml_backend_t pc = c.pc.empty()
                ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr)
                : ggml_backend_init_by_name(c.pc.c_str(),nullptr);
        demand(pc!=nullptr, "PC backend unavailable");
        ggml_backend_t phone = nullptr;
#if defined(GALAXY_CAUSAL_HAVE_RPC)
        if (!c.rpc.empty()) phone=ggml_backend_rpc_init(c.rpc.c_str(),0);
#else
        demand(c.rpc.empty(), "GGML_RPC=ON build required for Phone RPC");
#endif
        if (c.rpc.empty()) phone=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        if (phone==nullptr) {
            ggml_backend_free(pc);
            throw std::runtime_error("second backend unavailable");
        }
        try {
            execute(c, {pc,phone});
        } catch (...) {
            ggml_backend_free(phone);
            ggml_backend_free(pc);
            throw;
        }
        ggml_backend_free(phone);
        ggml_backend_free(pc);
        std::cout << "GALAXY_CAUSAL_NATIVE_FLASH_REPLAY PASS kernel_executed=1 "
                  << "production_graph_enabled=0\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "GALAXY_CAUSAL_NATIVE_FLASH_REPLAY FAIL " << e.what() << "\n";
        return 1;
    }
}

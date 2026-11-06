#include "../../ggml/src/ggml-backend-galaxy-collectives.h"

#include "ggml-alloc.h"
#include "ggml-backend-galaxy-markers.h"

#if defined(GALAXY_COLLECTIVES_HAVE_RPC)
#include "ggml-rpc.h"
#endif

#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ggml_galaxy_collectives;

static void require_test(bool condition, const std::string & reason) {
    if (!condition) throw std::runtime_error(reason);
}

static void expect_reject(const std::function<void()> & fn, const char * label) {
    bool caught = false;
    try { fn(); } catch (const std::invalid_argument &) { caught = true; }
    require_test(caught, std::string("expected rejection for ") + label);
}

static std::vector<uint8_t> copy_tensor_bytes(const ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

static void verify_sequence_weighting() {
    require_test(ggml_galaxy::parse_seq_weights(nullptr) == ggml_galaxy::parse_seq_weights("1,1"),
            "default SP sequence weights");
    require_test(ggml_galaxy::configured_sequence_split(2) == ggml_galaxy::split_sequence(2, {1,1}),
            "two-token warmup split");
    require_test(ggml_galaxy::split_sequence(72, {20,12}) ==
            (std::array<int64_t, 2>{45,27}), "72-token 20:12 split");
    require_test(ggml_galaxy::split_sequence(32, {20,12}) ==
            (std::array<int64_t, 2>{20,12}), "source seq_scatter_list 20:12");
    require_test(ggml_galaxy::split_sequence(72, {1,1}) ==
            (std::array<int64_t, 2>{36,36}), "balanced split");
    require_test(ggml_galaxy::split_sequence(2, {20,12}) ==
            (std::array<int64_t, 2>{1,1}), "small warmup");
    require_test(ggml_galaxy::split_sequence(3, {10000,1}) ==
            (std::array<int64_t, 2>{2,1}), "nonzero shard");
    expect_reject([] { (void) ggml_galaxy::parse_seq_weights(""); }, "empty weights");
    expect_reject([] { (void) ggml_galaxy::parse_seq_weights("1,0"); }, "zero weight");
    expect_reject([] { (void) ggml_galaxy::parse_seq_weights("3,2,1"); }, "three ranks");
    expect_reject([] { (void) ggml_galaxy::parse_seq_weights("-1,2"); }, "negative weight");
    expect_reject([] { (void) ggml_galaxy::parse_seq_weights("1,9223372036854775808"); }, "overflow");
    std::cout << "GALAXY_SEQ_WEIGHTS_SELFTEST PASS\n";
}

static void verify_rs_fused_config() {
    require_test(!parse_rs_fused_read(nullptr), "unset RS prefetch must be off");
    require_test(!parse_rs_fused_read("0"), "disabled RS prefetch");
    require_test(parse_rs_fused_read("1"), "enabled RS prefetch");
    require_test(rs_fused_read_eligible({72,73}, 5120*2), "14B 145-token F16 fits fused cap");
    require_test(!rs_fused_read_eligible({1024,1024}, 5120*2), "large RS falls back to tiles");
    expect_reject([] { (void) parse_rs_fused_read("true"); }, "invalid RS prefetch");
    expect_reject([] { (void) parse_rs_fused_read("2"); }, "invalid RS prefetch mode");
    std::cout << "GALAXY_RS_FUSED_CONFIG_SELFTEST PASS\n";
}

static void verify_rank_local_transport_parser() {
    require_test(parse_transport(nullptr)==transport::host,
                 "unset transport must preserve host baseline");
    require_test(parse_transport("")==transport::host,"empty transport selects host");
    require_test(parse_transport("HOST")==transport::host,"explicit HOST");
    require_test(parse_transport("RANK_LOCAL")==transport::rank_local,
                 "explicit two-rank local reduction");
    expect_reject([]{ (void)parse_transport("RING_DIRECT"); },
                  "unsupported direct ring transport");
    expect_reject([]{ (void)parse_transport("rank_local"); },
                  "mis-cased route");
    std::cout << "GALAXY_RANK_LOCAL_PARSER_SELFTEST PASS\n";
}

static void verify_tile_parser() {
    require_test(parse_tile_tokens(nullptr) == 32, "default tile must preserve baseline");
    require_test(parse_tile_tokens("") == 32, "empty tile must preserve baseline");
    require_test(parse_tile_tokens("1") == 1, "min tile");
    require_test(parse_tile_tokens("32") == 32, "legacy tile");
    require_test(parse_tile_tokens("256") == 256, "optimized tile");
    require_test(parse_tile_tokens("1024") == 1024, "max tile");
    expect_reject([] { (void) parse_tile_tokens("0"); }, "zero tile configuration");
    expect_reject([] { (void) parse_tile_tokens("1025"); }, "oversized tile configuration");
    expect_reject([] { (void) parse_tile_tokens("-1"); }, "negative tile configuration");
    expect_reject([] { (void) parse_tile_tokens("32x"); }, "non-numeric tile configuration");
    expect_reject([] { (void) parse_tile_tokens("18446744073709551615"); }, "overflow tile configuration");
    std::cout << "GALAXY_COLLECTIVE_TILE_SELFTEST PASS\n";
}

static void verify_marker_identity_survives_callback_rename() {
    const ggml_init_params params = {1024*1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    require_test(ctx != nullptr, "could not initialize marker test context");
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    // Names alone are not evidence of a layout transition.
    ggml_tensor * plain = ggml_dup(ctx, src);
    ggml_set_name(plain, "GALAXY_AG_FFN_0");
    require_test(ggml_galaxy::get_transition(plain) == ggml_galaxy::sp_transition::none,
                 "ordinary DUP must not become an AllGather based on its name");

    ggml_tensor * allgather = ggml_dup(ctx, src);
    ggml_galaxy::mark_transition(allgather, ggml_galaxy::sp_transition::all_gather);
    ggml_format_name(allgather, "GALAXY_AG_FFN_%d", 0);
    ggml_galaxy::set_stage(allgather, ggml_galaxy::sp_stage::ffn_output_ag, 0);
    ggml_format_name(allgather, "l_out-%d", 0); // matches graph_get_cb()
    require_test(ggml_galaxy::get_transition(allgather) == ggml_galaxy::sp_transition::all_gather,
                 "renaming the layer output must not erase an AllGather marker");
    require_test(ggml_galaxy::get_stage(allgather) == ggml_galaxy::sp_stage::ffn_output_ag &&
                 ggml_galaxy::get_stage_layer(allgather) == 0,
                 "renamed final FFN AllGather must retain stage/layer identity");
    require_test(ggml_galaxy::get_stage(plain) == ggml_galaxy::sp_stage::none,
                 "plain renamed tensor must never masquerade as a staged collective");

    ggml_tensor * rs = ggml_dup(ctx, src);
    ggml_galaxy::mark_transition(rs, ggml_galaxy::sp_transition::reduce_scatter);
    ggml_galaxy::set_stage(rs, ggml_galaxy::sp_stage::attention_rs, 7);
    ggml_set_name(rs, "renamed-rs");
    require_test(ggml_galaxy::get_transition(rs) == ggml_galaxy::sp_transition::reduce_scatter &&
                 ggml_galaxy::get_stage(rs) == ggml_galaxy::sp_stage::attention_rs &&
                 ggml_galaxy::get_stage_layer(rs) == 7,
                 "renaming must preserve attention ReduceScatter identity");
    expect_reject([&] {
        ggml_galaxy::set_stage(rs, ggml_galaxy::sp_stage::ffn_output_ag, 7);
    }, "cannot tag a ReduceScatter as an AllGather");

    ggml_tensor * scatter = ggml_dup(ctx, src);
    ggml_galaxy::mark_transition(scatter, ggml_galaxy::sp_transition::scatter_mirrored);
    ggml_galaxy::set_stage(scatter, ggml_galaxy::sp_stage::residual_scatter, 7);
    ggml_set_name(scatter, "renamed-residual");
    require_test(ggml_galaxy::get_transition(scatter) == ggml_galaxy::sp_transition::scatter_mirrored &&
                 ggml_galaxy::get_stage(scatter) == ggml_galaxy::sp_stage::residual_scatter &&
                 ggml_galaxy::get_stage_layer(scatter) == 7,
                 "renaming must preserve Residual Scatter identity");

    expect_reject([&] {
        ggml_galaxy::mark_transition(src, ggml_galaxy::sp_transition::all_gather);
    }, "cannot tag a non-DUP tensor");
    ggml_free(ctx);
    std::cout << "GALAXY_SP_MARKER_SELFTEST PASS\n";
}

static std::vector<uint8_t> pack(ggml_type type, const std::vector<float> & data) {
    const size_t unit = detail::element_bytes(type);
    std::vector<uint8_t> bytes(data.size()*unit);
    for (size_t i = 0; i < data.size(); ++i) detail::write(type, bytes.data()+i*unit, data[i]);
    return bytes;
}

static std::vector<float> unpack(const ggml_tensor * tensor) {
    const size_t unit = detail::element_bytes(tensor->type);
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    std::vector<float> out(bytes.size()/unit);
    for (size_t i = 0; i < out.size(); ++i) out[i] = detail::read(tensor->type, bytes.data()+i*unit);
    return out;
}

struct tensors {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * partial = nullptr;
    ggml_tensor * slice = nullptr;
    ggml_tensor * complete = nullptr;
    ggml_tensor * value_slice = nullptr;
    ggml_tensor * value_complete = nullptr;

    tensors(ggml_backend_t backend, ggml_type type, int64_t hidden, int64_t total, int64_t local) {
        const ggml_init_params params = {1024*1024, nullptr, true};
        ctx = ggml_init(params);
        require_test(ctx != nullptr, "ggml_init failed");
        partial  = ggml_new_tensor_2d(ctx, type, hidden, total);
        slice    = ggml_new_tensor_2d(ctx, type, hidden, local);
        complete = ggml_new_tensor_2d(ctx, type, hidden, total);
        value_slice = ggml_new_tensor_2d(ctx, type, hidden, local);
        value_complete = ggml_new_tensor_2d(ctx, type, hidden, total);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require_test(buffer != nullptr, "alloc_ctx_tensors failed");
    }

    ~tensors() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
    tensors(const tensors &) = delete;
    tensors & operator=(const tensors &) = delete;
};

static void check_equals(const std::vector<float> & got,
                         const std::vector<float> & expected, const char * stage) {
    require_test(got.size() == expected.size(), std::string(stage) + " size mismatch");
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i]-expected[i]) > 0.001f) {
            throw std::runtime_error(std::string(stage) + " differs at element " + std::to_string(i));
        }
    }
}

static void run_case(devices_t devices, ggml_type type,
                     int64_t hidden, counts_t counts, size_t tile) {
    const int64_t total = counts[0]+counts[1];
    tensors rank0(devices[0], type, hidden, total, counts[0]);
    tensors rank1(devices[1], type, hidden, total, counts[1]);
    std::vector<float> a(static_cast<size_t>(total*hidden));
    std::vector<float> b(a.size()), expected(a.size());
    for (int64_t t = 0; t < total; ++t) {
        for (int64_t h = 0; h < hidden; ++h) {
            const size_t j = static_cast<size_t>(t*hidden+h);
            a[j] = 0.125f * static_cast<float>((t*7+h*3)%29-14);
            b[j] = 0.250f * static_cast<float>((t*11+h*5)%31-15);
            expected[j] = a[j]+b[j];
        }
    }

    const auto a_bytes = pack(type, a), b_bytes = pack(type, b);
    ggml_backend_tensor_set(rank0.partial, a_bytes.data(), 0, a_bytes.size());
    ggml_backend_tensor_set(rank1.partial, b_bytes.data(), 0, b_bytes.size());

    input_t partial = {rank0.partial, rank1.partial};
    output_t slices = {rank0.slice, rank1.slice};
    input_t shards = {rank0.slice, rank1.slice};
    output_t full = {rank0.complete, rank1.complete};
    reduce_scatter(devices, partial, slices, counts, tile, false, transport::host);
    // The fused full-input read must produce byte-for-byte identical
    // F16/F32 output, independently of the existing TP/SP-vs-TP error.
    const auto baseline_rank0 = copy_tensor_bytes(rank0.slice);
    const auto baseline_rank1 = copy_tensor_bytes(rank1.slice);
    reduce_scatter(devices, partial, slices, counts, tile, true, transport::host);
    require_test(copy_tensor_bytes(rank0.slice) == baseline_rank0,
                 "fused RS differs bitwise on rank 0");
    require_test(copy_tensor_bytes(rank1.slice) == baseline_rank1,
                 "fused RS differs bitwise on rank 1");
    // Exact source Ring semantics (two ranks): each rank receives ONLY
    // its peer's partial slice and uses a backend ADD for its own tokens.
    reduce_scatter(devices, partial, slices, counts, tile, false,
                   transport::rank_local);
    require_test(copy_tensor_bytes(rank0.slice) == baseline_rank0,
                 "rank-local Ring RS differs bitwise from host on rank 0");
    require_test(copy_tensor_bytes(rank1.slice) == baseline_rank1,
                 "rank-local Ring RS differs bitwise from host on rank 1");
    expect_reject([&]{ reduce_scatter(devices,partial,slices,counts,tile,
                                     true,transport::rank_local); },
                  "rank-local mode must reject host-only fused read");

    const size_t edge = static_cast<size_t>(counts[0]*hidden);
    check_equals(unpack(rank0.slice),
                 std::vector<float>(expected.begin(), expected.begin()+edge),
                 "RS rank0");
    check_equals(unpack(rank1.slice),
                 std::vector<float>(expected.begin()+edge, expected.end()),
                 "RS rank1");

    all_gather(devices, shards, full, counts, tile, transport::host);
    const auto ag_expected0=copy_tensor_bytes(rank0.complete);
    const auto ag_expected1=copy_tensor_bytes(rank1.complete);
    all_gather(devices, shards, full, counts, tile, transport::rank_local);
    require_test(copy_tensor_bytes(rank0.complete)==ag_expected0,
                 "rank-local AllGather differs bitwise on rank 0");
    require_test(copy_tensor_bytes(rank1.complete)==ag_expected1,
                 "rank-local AllGather differs bitwise on rank 1");
    check_equals(unpack(rank0.complete), expected, "AG rank0");
    check_equals(unpack(rank1.complete), expected, "AG rank1");

    // Mirror the source-defined SP Attention K/V AllGather. Values differ
    // from keys to catch accidental K/V aliasing or out-of-order gathering.
    const std::vector<float> key_full = expected;
    std::vector<float> value_full(key_full.size());
    for (size_t i = 0; i < value_full.size(); ++i) {
        value_full[i] = key_full[i]*0.5f - 0.125f;
    }
    const auto v0 = pack(type, std::vector<float>(value_full.begin(), value_full.begin()+edge));
    const auto v1 = pack(type, std::vector<float>(value_full.begin()+edge, value_full.end()));
    ggml_backend_tensor_set(rank0.value_slice, v0.data(), 0, v0.size());
    ggml_backend_tensor_set(rank1.value_slice, v1.data(), 0, v1.size());
    const input_t kv_keys = {rank0.slice, rank1.slice};
    const input_t kv_values = {rank0.value_slice, rank1.value_slice};
    const output_t gathered_k = {rank0.complete, rank1.complete};
    const output_t gathered_v = {rank0.value_complete, rank1.value_complete};
    all_gather_attention_kv(devices, kv_keys, kv_values, gathered_k, gathered_v,
                            counts, tile, transport::host);
    const auto k_expected0=copy_tensor_bytes(rank0.complete);
    const auto k_expected1=copy_tensor_bytes(rank1.complete);
    const auto v_expected0=copy_tensor_bytes(rank0.value_complete);
    const auto v_expected1=copy_tensor_bytes(rank1.value_complete);
    all_gather_attention_kv(devices, kv_keys, kv_values, gathered_k, gathered_v,
                            counts, tile, transport::rank_local);
    require_test(copy_tensor_bytes(rank0.complete)==k_expected0 &&
                 copy_tensor_bytes(rank1.complete)==k_expected1 &&
                 copy_tensor_bytes(rank0.value_complete)==v_expected0 &&
                 copy_tensor_bytes(rank1.value_complete)==v_expected1,
                 "rank-local SP K/V allgathers differ from host reference");
    check_equals(unpack(rank0.complete), key_full, "SP attention K rank0");
    check_equals(unpack(rank1.complete), key_full, "SP attention K rank1");
    check_equals(unpack(rank0.value_complete), value_full, "SP attention V rank0");
    check_equals(unpack(rank1.value_complete), value_full, "SP attention V rank1");
    expect_reject([&] { all_gather_attention_kv(devices, kv_keys, kv_keys,
            gathered_k, gathered_v, counts, tile); }, "K/V cannot alias");
    std::cout << "[GALAXY_SP_ATT_KV] tokens=" << total << " split="
              << counts[0] << "," << counts[1] << " status=PASS\n";

    // The independently gathered complete activation is the canonical
    // residual input; splitting it must return exactly the old intervals.
    const input_t mirrored = {rank0.complete, rank1.complete};
    scatter_mirrored(devices, mirrored, slices, counts, tile, transport::host);
    const auto scatter_expected0=copy_tensor_bytes(rank0.slice);
    const auto scatter_expected1=copy_tensor_bytes(rank1.slice);
    scatter_mirrored(devices, mirrored, slices, counts, tile, transport::rank_local);
    require_test(copy_tensor_bytes(rank0.slice)==scatter_expected0 &&
                 copy_tensor_bytes(rank1.slice)==scatter_expected1,
                 "rank-local mirrored residual differs from host reference");
    check_equals(unpack(rank0.slice),
                 std::vector<float>(expected.begin(), expected.begin()+edge),
                 "SP residual rank0");
    check_equals(unpack(rank1.slice),
                 std::vector<float>(expected.begin()+edge, expected.end()),
                 "SP residual rank1");

    expect_reject([&] { reduce_scatter(devices, partial, slices, {counts[0]+1,counts[1]}); }, "sum");
    expect_reject([&] { all_gather(devices, shards, full, {0,total}); }, "zero rank");
    expect_reject([&] { reduce_scatter(devices, partial, slices, counts, 0); }, "zero tile");
    expect_reject([&] { all_gather(devices, shards, full, counts, 0); }, "zero gather tile");
    expect_reject([&] { reduce_scatter(devices, partial, {rank0.partial,rank1.slice}, counts); }, "in-place");
    expect_reject([&] { scatter_mirrored(devices, mirrored, slices, counts, 0); }, "zero scatter tile");

    std::cout << "[GALAXY_COLLECTIVES_CASE] transports=HOST,RANK_LOCAL"
              << " bitwise_parity=PASS tokens=" << total
              << " hidden=" << hidden << " seq=" << counts[0] << "," << counts[1]
              << " type=" << (type == GGML_TYPE_F32 ? "F32" : "F16")
              << " tile=" << tile << " PASS\n";
}

int main(int argc, char ** argv) {
    try {
        std::string rpc_endpoint, pc_name;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--rpc" && i+1 < argc) rpc_endpoint = argv[++i];
            else if (flag == "--pc" && i+1 < argc) pc_name = argv[++i];
            else throw std::invalid_argument("usage: galaxy-collectives-selftest [--pc CUDA0] [--rpc IP:50052]");
        }
        verify_marker_identity_survives_callback_rename();
        verify_tile_parser();
        verify_sequence_weighting();
        verify_rs_fused_config();
        verify_rank_local_transport_parser();
        ggml_backend_load_all();
        ggml_backend_t pc = pc_name.empty()
                ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)
                : ggml_backend_init_by_name(pc_name.c_str(), nullptr);
        require_test(pc != nullptr, "PC backend unavailable");
        ggml_backend_t second = nullptr;
#if defined(GALAXY_COLLECTIVES_HAVE_RPC)
        if (!rpc_endpoint.empty()) {
            second = ggml_backend_rpc_init(rpc_endpoint.c_str(), 0);
            require_test(second != nullptr, "phone RPC backend unavailable: cannot substitute local CPU");
        }
#else
        require_test(rpc_endpoint.empty(), "rebuild with GGML_RPC=ON for RPC tests");
#endif
        if (rpc_endpoint.empty()) second = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        require_test(second != nullptr, "second backend unavailable");

        try {
            const devices_t devices = {pc, second};
            run_case(devices, GGML_TYPE_F32,   7, {1,2}, 1);
            run_case(devices, GGML_TYPE_F32,  16, {3,5}, 2);
            run_case(devices, GGML_TYPE_F32,  64, {20,44}, 16);
            run_case(devices, GGML_TYPE_F32, 128, {64,65}, 32);
            run_case(devices, GGML_TYPE_F16,  32, {7,9}, 3);
            run_case(devices, GGML_TYPE_F16,  64, {31,33}, 16);
            // Full hidden dimension, unequal local sequence sizes and a
            // tile larger than either rank's shard: one transfer per slice.
            run_case(devices, GGML_TYPE_F16, 5120, {72,73}, 256);
            run_case(devices, GGML_TYPE_F32, 16, {3,5}, 1024);
        } catch (...) {
            ggml_backend_free(second);
            ggml_backend_free(pc);
            throw;
        }
        ggml_backend_free(second);
        ggml_backend_free(pc);
        std::cout << "GALAXY_COLLECTIVES_SELFTEST PASS mode="
                  << (rpc_endpoint.empty() ? "LOCAL" : "PC_RPC") << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "GALAXY_COLLECTIVES_SELFTEST FAIL: " << e.what() << "\n";
        return 1;
    }
}

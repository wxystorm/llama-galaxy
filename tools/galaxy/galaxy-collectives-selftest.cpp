#include "../../ggml/src/ggml-backend-galaxy-collectives.h"

#include "ggml-alloc.h"

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

    tensors(ggml_backend_t backend, ggml_type type, int64_t hidden, int64_t total, int64_t local) {
        const ggml_init_params params = {1024*1024, nullptr, true};
        ctx = ggml_init(params);
        require_test(ctx != nullptr, "ggml_init failed");
        partial  = ggml_new_tensor_2d(ctx, type, hidden, total);
        slice    = ggml_new_tensor_2d(ctx, type, hidden, local);
        complete = ggml_new_tensor_2d(ctx, type, hidden, total);
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
    reduce_scatter(devices, partial, slices, counts, tile);

    const size_t edge = static_cast<size_t>(counts[0]*hidden);
    check_equals(unpack(rank0.slice),
                 std::vector<float>(expected.begin(), expected.begin()+edge),
                 "RS rank0");
    check_equals(unpack(rank1.slice),
                 std::vector<float>(expected.begin()+edge, expected.end()),
                 "RS rank1");

    all_gather(devices, shards, full, counts, tile);
    check_equals(unpack(rank0.complete), expected, "AG rank0");
    check_equals(unpack(rank1.complete), expected, "AG rank1");

    expect_reject([&] { reduce_scatter(devices, partial, slices, {counts[0]+1,counts[1]}); }, "sum");
    expect_reject([&] { all_gather(devices, shards, full, {0,total}); }, "zero rank");
    expect_reject([&] { reduce_scatter(devices, partial, slices, counts, 0); }, "zero tile");
    expect_reject([&] { all_gather(devices, shards, full, counts, 0); }, "zero gather tile");
    expect_reject([&] { reduce_scatter(devices, partial, {rank0.partial,rank1.slice}, counts); }, "in-place");

    std::cout << "[GALAXY_COLLECTIVES_CASE] tokens=" << total
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
        ggml_backend_load_all();
        ggml_backend_t pc = pc_name.empty()
                ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)
                : ggml_backend_init_by_name(pc_name.c_str(), nullptr);
        require_test(pc != nullptr, "PC backend unavailable");
        ggml_backend_t second = nullptr;
#if defined(GALAXY_COLLECTIVES_HAVE_RPC)
        if (!rpc_endpoint.empty()) second = ggml_backend_rpc_init(rpc_endpoint.c_str(), 0);
#else
        require_test(rpc_endpoint.empty(), "rebuild with GGML_RPC=ON for RPC tests");
#endif
        if (!second) second = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        require_test(second != nullptr, "second backend unavailable");

        try {
            const devices_t devices = {pc, second};
            run_case(devices, GGML_TYPE_F32,   7, {1,2}, 1);
            run_case(devices, GGML_TYPE_F32,  16, {3,5}, 2);
            run_case(devices, GGML_TYPE_F32,  64, {20,44}, 16);
            run_case(devices, GGML_TYPE_F32, 128, {64,65}, 32);
            run_case(devices, GGML_TYPE_F16,  32, {7,9}, 3);
            run_case(devices, GGML_TYPE_F16,  64, {31,33}, 16);
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

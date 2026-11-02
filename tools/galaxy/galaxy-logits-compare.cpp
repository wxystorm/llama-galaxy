// Standalone full-vocabulary logits comparator for Galaxy TP/SP correctness.
// File format: 4-byte "GLG1" magic, signed 32-bit vocab count, then FP32 logits.
// No dependency on ggml: works on the PC even when the phone is disconnected.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct metrics {
    int vocab = 0;
    int ref_top1 = -1;
    int test_top1 = -1;
    int top10_overlap = 0;
    int largest_index = -1;
    int invalid_mismatches = 0;
    double max_abs = 0;
    double mean_abs = 0;
    double rmse = 0;
    double cosine = 0;
};

void require(bool cond, const std::string & what) {
    if (!cond) throw std::runtime_error(what);
}

std::vector<float> read_logits(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    require(in.is_open(), "cannot open " + path);
    const std::streamoff bytes = in.tellg();
    require(bytes >= 8, "truncated header in " + path);
    in.seekg(0);
    char magic[4] {};
    int32_t count = 0;
    in.read(magic, 4);
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
    require(in.good() && std::string(magic, 4) == "GLG1", "invalid magic/header in " + path);
    require(count > 0 && count <= 1000000, "invalid vocabulary size in " + path);
    require(bytes == std::streamoff(8) + std::streamoff(count)*4,
            "incomplete or unexpected logit payload size in " + path);
    std::vector<float> v(size_t(count), 0.f);
    in.read(reinterpret_cast<char *>(v.data()), std::streamsize(count)*4);
    require(in.good(), "failed to read logits from " + path);
    return v;
}

std::vector<int> top_ids(const std::vector<float> & v, size_t top) {
    for (float value : v) {
        require(!std::isnan(value), "NaN logits are not comparable");
    }
    std::vector<int> ids(v.size());
    std::iota(ids.begin(), ids.end(), 0);
    top = std::min(top, ids.size());
    std::partial_sort(ids.begin(), ids.begin()+top, ids.end(), [&](int a, int b) {
        return v[size_t(a)] > v[size_t(b)] ||
                (v[size_t(a)] == v[size_t(b)] && a < b);
    });
    ids.resize(top);
    return ids;
}

metrics compare(const std::vector<float> & ref, const std::vector<float> & test) {
    require(ref.size() == test.size() && !ref.empty(), "vocab length mismatch");
    metrics m {};
    m.vocab = int(ref.size());
    const auto ref_ids = top_ids(ref, 10);
    const auto test_ids = top_ids(test, 10);
    m.ref_top1 = ref_ids[0];
    m.test_top1 = test_ids[0];
    for (int id : ref_ids) {
        if (std::find(test_ids.begin(), test_ids.end(), id) != test_ids.end()) {
            m.top10_overlap++;
        }
    }

    double sum_abs = 0, sum_sq = 0, dot = 0, norm_ref = 0, norm_test = 0;
    size_t n_finite = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float a = ref[i], b = test[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            if (a != b) m.invalid_mismatches++;
            continue;
        }
        const double d = double(a) - double(b);
        const double abs = std::fabs(d);
        if (abs > m.max_abs) {
            m.max_abs = abs;
            m.largest_index = int(i);
        }
        sum_abs += abs;
        sum_sq += d*d;
        dot += double(a)*double(b);
        norm_ref += double(a)*double(a);
        norm_test += double(b)*double(b);
        n_finite++;
    }
    require(n_finite > 0, "there are no finite logits to compare");
    m.mean_abs = sum_abs / double(n_finite);
    m.rmse = std::sqrt(sum_sq / double(n_finite));
    m.cosine = norm_ref > 0 && norm_test > 0
             ? dot / std::sqrt(norm_ref*norm_test)
             : 0;
    return m;
}

double parse_nonnegative(const std::string & v, const char * label) {
    size_t consumed = 0;
    const double x = std::stod(v, &consumed);
    require(consumed == v.size() && std::isfinite(x) && x >= 0,
            std::string(label) + " must be a nonnegative finite number");
    return x;
}

void run_selftest() {
    const std::vector<float> base = {0, 1, 2, 3, 10, -4, -8, 6};
    const metrics identical = compare(base, base);
    require(identical.max_abs == 0 && identical.rmse == 0 &&
            identical.ref_top1 == identical.test_top1, "exact selftest failed");
    auto changed = base;
    changed[0] = 0.5f;
    const metrics small = compare(base, changed);
    require(small.max_abs == 0.5 && small.ref_top1 == small.test_top1 &&
            small.rmse > 0, "perturbation selftest failed");
    changed[4] = -1.f;
    const metrics different = compare(base, changed);
    require(different.ref_top1 != different.test_top1,
            "top-1 mismatch selftest failed");
    std::cout << "GALAXY_LOGITS_COMPARE_SELFTEST PASS\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--selftest") {
            run_selftest();
            return 0;
        }
        std::string ref_path, sp_path, json_path;
        double max_abs_tol = 0.10;
        double rmse_tol = 0.02;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            require(i+1 < argc, "missing value for " + arg);
            if (arg == "--tp") ref_path = argv[++i];
            else if (arg == "--sp") sp_path = argv[++i];
            else if (arg == "--max-abs") max_abs_tol = parse_nonnegative(argv[++i], "--max-abs");
            else if (arg == "--max-rmse") rmse_tol = parse_nonnegative(argv[++i], "--max-rmse");
            else if (arg == "--json") json_path = argv[++i];
            else throw std::invalid_argument("unknown argument: " + arg);
        }
        require(!ref_path.empty() && !sp_path.empty(),
                "usage: galaxy-logits-compare --tp <file> --sp <file> [--max-abs 0.1] [--max-rmse 0.02] [--json report.json]");
        const metrics m = compare(read_logits(ref_path), read_logits(sp_path));
        const bool passed = m.invalid_mismatches == 0 &&
                            m.ref_top1 == m.test_top1 &&
                            m.max_abs <= max_abs_tol && m.rmse <= rmse_tol;
        std::cout << std::setprecision(9)
                  << "[GALAXY_LOGITS_COMPARE] status=" << (passed ? "PASS" : "FAIL")
                  << " vocab=" << m.vocab
                  << " tp_top1=" << m.ref_top1 << " sp_top1=" << m.test_top1
                  << " top10_overlap=" << m.top10_overlap << "/10\n"
                  << "[GALAXY_LOGITS_COMPARE] max_abs=" << m.max_abs
                  << " max_abs_index=" << m.largest_index
                  << " mean_abs=" << m.mean_abs
                  << " rmse=" << m.rmse
                  << " cosine=" << m.cosine
                  << " nonfinite_mismatch=" << m.invalid_mismatches << "\n"
                  << "[GALAXY_LOGITS_COMPARE] limits max_abs=" << max_abs_tol
                  << " rmse=" << rmse_tol << " top1_equal=required\n";
        if (!json_path.empty()) {
            std::ofstream o(json_path, std::ios::trunc);
            require(o.is_open(), "cannot create JSON report: " + json_path);
            o << std::setprecision(12)
              << "{\n"
              << "  \"status\": \"" << (passed ? "PASS" : "FAIL") << "\",\n"
              << "  \"vocab\": " << m.vocab << ",\n"
              << "  \"tp_top1\": " << m.ref_top1 << ",\n"
              << "  \"sp_top1\": " << m.test_top1 << ",\n"
              << "  \"top10_overlap\": " << m.top10_overlap << ",\n"
              << "  \"max_abs\": " << m.max_abs << ",\n"
              << "  \"max_abs_index\": " << m.largest_index << ",\n"
              << "  \"mean_abs\": " << m.mean_abs << ",\n"
              << "  \"rmse\": " << m.rmse << ",\n"
              << "  \"cosine\": " << m.cosine << ",\n"
              << "  \"nonfinite_mismatch\": " << m.invalid_mismatches << ",\n"
              << "  \"max_abs_tolerance\": " << max_abs_tol << ",\n"
              << "  \"rmse_tolerance\": " << rmse_tol << "\n}\n";
            require(o.good(), "cannot write JSON report: " + json_path);
        }
        return passed ? 0 : 2;
    } catch (const std::exception & e) {
        std::cerr << "[GALAXY_LOGITS_COMPARE] ERROR " << e.what() << "\n";
        return 1;
    }
}

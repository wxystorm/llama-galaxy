#include "../../ggml/src/ggml-backend-galaxy-causal-plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Independent numerical reference: NOT a GPU or RPC implementation.
// Compare full causal attention to two contiguous local-Q shards reading
// globally gathered (including cached-prefix) K/V. Uses grouped-query
// heads and absolute-position RoPE, so local offsets cannot be mistaken
// for global causal-mask or rotary positions.

using ggml_galaxy_causal::layout;

static void check(bool cond, const char * reason) {
    if (!cond) {
        throw std::runtime_error(reason);
    }
}

static void rejects(const std::function<void()> & fn) {
    bool caught = false;
    try {
        fn();
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    check(caught, "invalid layout unexpectedly accepted");
}

struct sample {
    layout shape;
    std::vector<float> q; // new_tokens, query_heads, head_dim
    std::vector<float> k; // cached_tokens+new_tokens, kv_heads, head_dim
    std::vector<float> v; // cached_tokens+new_tokens, kv_heads, head_dim
};

static size_t index(int64_t token, int64_t head, int64_t dim,
                    int64_t heads, int64_t width) {
    return static_cast<size_t>((token * heads + head) * width + dim);
}

static void rope(std::vector<float> & data, int64_t token, int64_t position,
                 int64_t head, int64_t heads, int64_t dim) {
    // Use absolute positions for both current Query and cached/new Key.
    // Each adjacent pair receives the same rotary angle.
    for (int64_t d = 0; d+1 < dim; d += 2) {
        const double theta = static_cast<double>(position) /
                std::pow(10000.0, static_cast<double>(d) / static_cast<double>(dim));
        const double co = std::cos(theta), si = std::sin(theta);
        const auto at = index(token, head, d, heads, dim);
        const double a = data[at], b = data[at+1];
        data[at]   = static_cast<float>(a*co - b*si);
        data[at+1] = static_cast<float>(a*si + b*co);
    }
}

static sample make_sample(layout shape) {
    ggml_galaxy_causal::validate(shape);
    sample result;
    result.shape = shape;
    const int64_t n_kv = ggml_galaxy_causal::total_kv(shape);
    result.q.resize(static_cast<size_t>(shape.new_tokens*shape.query_heads*shape.head_dim));
    result.k.resize(static_cast<size_t>(n_kv*shape.kv_heads*shape.head_dim));
    result.v.resize(result.k.size());

    for (int64_t t = 0; t < shape.new_tokens; ++t) {
        for (int64_t h = 0; h < shape.query_heads; ++h) {
            for (int64_t d = 0; d < shape.head_dim; ++d) {
                result.q[index(t,h,d,shape.query_heads,shape.head_dim)] =
                    static_cast<float>(std::sin(double(2*t+11*h+3*d+1)*0.17) +
                                       std::cos(double(t+7*h+5*d+2)*0.09)*0.3);
            }
            rope(result.q, t, shape.cached_tokens+t, h, shape.query_heads, shape.head_dim);
        }
    }
    for (int64_t t = 0; t < n_kv; ++t) {
        for (int64_t h = 0; h < shape.kv_heads; ++h) {
            for (int64_t d = 0; d < shape.head_dim; ++d) {
                result.k[index(t,h,d,shape.kv_heads,shape.head_dim)] =
                    static_cast<float>(std::sin(double(3*t+5*h+7*d+1)*0.11));
                result.v[index(t,h,d,shape.kv_heads,shape.head_dim)] =
                    static_cast<float>(std::cos(double(7*t+3*h+2*d+2)*0.13));
            }
            rope(result.k, t, t, h, shape.kv_heads, shape.head_dim);
        }
    }
    return result;
}

static std::vector<float> attention_one(const sample & data,
                                        int64_t new_query, int64_t query_head,
                                        bool causal = true,
                                        bool use_sp_contract = false) {
    const layout & shape = data.shape;
    const int64_t query_pos = shape.cached_tokens + new_query;
    const int64_t kv_head = ggml_galaxy_causal::kv_head_for_query(shape, query_head);
    const int64_t n_kv = ggml_galaxy_causal::total_kv(shape);
    const double scale = 1.0 / std::sqrt(double(shape.head_dim));
    std::vector<double> scores(static_cast<size_t>(n_kv),
                               -std::numeric_limits<double>::infinity());
    double max_score = -std::numeric_limits<double>::infinity();
    for (int64_t key = 0; key < n_kv; ++key) {
        const bool permitted = !causal || (use_sp_contract
                ? ggml_galaxy_causal::visible(shape, query_pos, key)
                : key <= query_pos);
        if (!permitted) {
            continue;
        }
        double dot = 0.0;
        for (int64_t d = 0; d < shape.head_dim; ++d) {
            dot += double(data.q[index(new_query,query_head,d,shape.query_heads,shape.head_dim)]) *
                   double(data.k[index(key,kv_head,d,shape.kv_heads,shape.head_dim)]);
        }
        const double score = dot*scale;
        scores[static_cast<size_t>(key)] = score;
        max_score = std::max(max_score, score);
    }
    check(std::isfinite(max_score), "query without visible keys");

    double sum_exp = 0.0;
    for (double score : scores) {
        if (std::isfinite(score)) {
            sum_exp += std::exp(score - max_score);
        }
    }
    std::vector<float> out(static_cast<size_t>(shape.head_dim), 0.0f);
    for (int64_t d = 0; d < shape.head_dim; ++d) {
        double value = 0.0;
        for (int64_t key = 0; key < n_kv; ++key) {
            if (std::isfinite(scores[static_cast<size_t>(key)])) {
                const double weight = std::exp(scores[static_cast<size_t>(key)]-max_score)/sum_exp;
                value += weight*double(data.v[index(key,kv_head,d,shape.kv_heads,shape.head_dim)]);
            }
        }
        out[static_cast<size_t>(d)] = static_cast<float>(value);
    }
    return out;
}

static std::vector<float> full_reference(const sample & data) {
    const layout & s = data.shape;
    std::vector<float> out(static_cast<size_t>(s.new_tokens*s.query_heads*s.head_dim));
    for (int64_t t = 0; t < s.new_tokens; ++t) {
        for (int64_t h = 0; h < s.query_heads; ++h) {
            const auto result = attention_one(data, t, h);
            for (int64_t d = 0; d < s.head_dim; ++d) {
                out[index(t,h,d,s.query_heads,s.head_dim)] = result[static_cast<size_t>(d)];
            }
        }
    }
    return out;
}

static std::vector<float> sequence_parallel(const sample & data) {
    const layout & s = data.shape;
    // Model the backend pair of K/V AllGathers, including the historical
    // cache prefix. K/V ownership for new tokens is separate from local Q.
    sample gathered = data;
    std::fill(gathered.k.begin(), gathered.k.end(), 0.0f);
    std::fill(gathered.v.begin(), gathered.v.end(), 0.0f);
    const size_t kv_stride = static_cast<size_t>(s.kv_heads*s.head_dim);
    const size_t history = static_cast<size_t>(s.cached_tokens)*kv_stride;
    std::copy_n(data.k.begin(), history, gathered.k.begin());
    std::copy_n(data.v.begin(), history, gathered.v.begin());
    for (int rank = 0; rank < 2; ++rank) {
        const auto interval_ = ggml_galaxy_causal::rank_queries(s, rank);
        const size_t at = static_cast<size_t>(s.cached_tokens+interval_.start)*kv_stride;
        const size_t count = static_cast<size_t>(interval_.count)*kv_stride;
        std::copy_n(data.k.begin()+at, count, gathered.k.begin()+at);
        std::copy_n(data.v.begin()+at, count, gathered.v.begin()+at);
    }
    check(gathered.k == data.k && gathered.v == data.v,
          "K/V gather did not reconstruct the global cache");

    std::vector<float> out(static_cast<size_t>(s.new_tokens*s.query_heads*s.head_dim));
    for (int rank = 0; rank < 2; ++rank) {
        const auto interval_ = ggml_galaxy_causal::rank_queries(s, rank);
        for (int64_t local = 0; local < interval_.count; ++local) {
            const int64_t pos = ggml_galaxy_causal::absolute_query_position(s, rank, local);
            const int64_t t = pos - s.cached_tokens;
            for (int64_t h = 0; h < s.query_heads; ++h) {
                const auto result = attention_one(gathered, t, h, true, true);
                for (int64_t d = 0; d < s.head_dim; ++d) {
                    out[index(t,h,d,s.query_heads,s.head_dim)] = result[static_cast<size_t>(d)];
                }
            }
        }
    }
    return out;
}

static void run_case(layout shape) {
    const auto data = make_sample(shape);
    const auto reference = full_reference(data);
    const auto sharded = sequence_parallel(data);
    check(reference.size() == sharded.size(), "reference/SP shape mismatch");
    double max_abs = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double diff = std::abs(double(reference[i]) - double(sharded[i]));
        max_abs = std::max(max_abs, diff);
        check(diff <= 1.0e-6, "local Q / global KV numerical parity failed");
    }
    check(ggml_galaxy_causal::visible(shape, shape.cached_tokens, 0),
          "first local query must see oldest KV");
    check(!ggml_galaxy_causal::visible(shape, shape.cached_tokens,
                                       ggml_galaxy_causal::total_kv(shape)-1) ||
          shape.new_tokens == 1, "first query must mask future KV");
    std::cout << "[GALAXY_CAUSAL_SP_REFERENCE] cached=" << shape.cached_tokens
              << " tokens=" << shape.new_tokens
              << " split=" << shape.rank0_queries << ","
              << shape.new_tokens-shape.rank0_queries
              << " q_heads=" << shape.query_heads
              << " kv_heads=" << shape.kv_heads
              << " max_abs=" << max_abs << " status=PASS\n";
}

int main() {
    try {
        run_case({0, 9, 4, 24, 8, 8});
        run_case({13, 11, 7, 24, 8, 8});
        run_case({1, 2, 1, 16, 8, 8});
        run_case({31, 17, 10, 8, 8, 16});
        run_case({64, 1, 1, 24, 8, 8});  // decode contract only; runtime TP
        check(!ggml_galaxy_causal::runtime_prefill_eligible({64,1,1,24,8,8}),
              "single-token decode must not enable SP Attention");
        check(ggml_galaxy_causal::runtime_prefill_eligible({13,11,7,24,8,8}),
              "asymmetric prefill should be eligible");
        check(ggml_galaxy_causal::absolute_query_position({13,11,7,24,8,8}, 1, 0) == 20,
              "rank1 absolute query position must include past and rank0 new tokens");
        check(ggml_galaxy_causal::kv_head_for_query({13,11,7,24,8,8}, 3) == 1,
              "GQA group-head mapping");
        rejects([] { ggml_galaxy_causal::validate({0, 4, 2, 24, 8, 8, 2}); });
        rejects([] { ggml_galaxy_causal::validate({0, 4, 5, 24, 8, 8}); });
        rejects([] { ggml_galaxy_causal::validate({0, 4, 2, 24, 7, 8}); });
        rejects([] { ggml_galaxy_causal::validate({-1, 4, 2, 24, 8, 8}); });
        rejects([] { ggml_galaxy_causal::validate({0, 0, 0, 24, 8, 8}); });
        rejects([] { (void) ggml_galaxy_causal::rank_queries({0,4,2,24,8,8}, 2); });
        rejects([] { (void) ggml_galaxy_causal::absolute_query_position({3,4,2,24,8,8}, 1, 2); });
        rejects([] { (void) ggml_galaxy_causal::kv_head_for_query({0,4,2,24,8,8}, 24); });
        rejects([] { (void) ggml_galaxy_causal::visible({0,4,2,24,8,8}, 4, 0); });
        // Incorrectly unmasking future keys must observably change results.
        const auto causal_sample = make_sample({0,9,4,24,8,8});
        const auto good = attention_one(causal_sample, 0, 0, true);
        const auto incorrect = attention_one(causal_sample, 0, 0, false);
        double diff = 0.0;
        for (size_t i = 0; i < good.size(); ++i) {
            diff = std::max(diff, std::abs(double(good[i])-double(incorrect[i])));
        }
        check(diff > 1.0e-4, "future-key masking negative control not detected");
        std::cout << "GALAXY_CAUSAL_ATTENTION_SELFTEST PASS\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "GALAXY_CAUSAL_ATTENTION_SELFTEST FAIL " << e.what() << "\n";
        return 1;
    }
}

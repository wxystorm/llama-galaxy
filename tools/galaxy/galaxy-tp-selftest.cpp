#include "../../src/llama-galaxy-tp.h"

#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

using llama_galaxy_tp::parse_counts;
using llama_galaxy_tp::validate_counts;

static void require_test(bool ok, const char * name) {
    if (!ok) throw std::runtime_error(name);
}
static void must_throw(const std::function<void()> & fn) {
    bool did_throw = false;
    try { fn(); } catch (const std::invalid_argument &) { did_throw = true; }
    require_test(did_throw, "expected std::invalid_argument");
}

int main() {
    require_test((parse_counts("20,20", "heads", 2) == std::vector<int64_t>{20, 20}), "head split20/20");
    require_test((parse_counts("15,25", "heads", 2) == std::vector<int64_t>{15, 25}), "head split15/25");
    validate_counts({20, 20}, 40, 5, "heads");
    validate_counts({15, 25}, 40, 5, "heads");
    validate_counts({8704, 8704}, 17408, 256, "ffn");
    must_throw([] { parse_counts("", "heads", 2); });
    must_throw([] { parse_counts("20,0", "heads", 2); });
    must_throw([] { parse_counts("20,", "heads", 2); });
    must_throw([] { parse_counts("20,x", "heads", 2); });
    must_throw([] { parse_counts("20", "heads", 2); });
    must_throw([] { parse_counts("9223372036854775808,1", "heads", 2); });
    must_throw([] { validate_counts({16, 24}, 40, 5, "heads"); });
    must_throw([] { validate_counts({20, 20}, 41, 5, "heads"); });
    must_throw([] { validate_counts({8704, 8704}, 17408, 1024, "ffn"); });
    // Dense Qwen3 and Llama 3.2 can reuse the same static rank policy:
    // Llama 3.2 3B: 24 Attention heads, 8 KV heads, FFN=8192.
    validate_counts({12, 12}, 24, 3, "Llama3.2 attention heads");
    validate_counts({4096, 4096}, 8192, 256, "Llama3.2 FFN units");
    must_throw([] { validate_counts({10,14}, 24, 3, "invalid Llama GQA split"); });
    // The original Qwen3 last-token shortcut is retained unless final-layer
    // SP was explicitly requested, so these eligible counts differ by one.
    require_test(llama_galaxy_tp::galaxy2_eligible_ffn_layers(40, false) == 39, "last TP");
    require_test(llama_galaxy_tp::galaxy2_eligible_ffn_layers(40, true) == 40, "full Galaxy2");
    must_throw([] { (void) llama_galaxy_tp::galaxy2_eligible_ffn_layers(0, true); });
    // Galaxy2 full mode and limited-memory hybrid share the same parser.
    require_test(llama_galaxy_tp::galaxy2_parse_ffn_layers(nullptr, 39) == 39, "galaxy2 all");
    require_test(llama_galaxy_tp::galaxy2_parse_ffn_layers("all", 39) == 39, "galaxy2 all2");
    require_test(llama_galaxy_tp::galaxy2_parse_ffn_layers("1", 39) == 1, "galaxy2 one");
    require_test(llama_galaxy_tp::galaxy2_parse_ffn_layers("0", 39) == 0, "galaxy2 zero");
    must_throw([] { llama_galaxy_tp::galaxy2_parse_ffn_layers("", 39); });
    must_throw([] { llama_galaxy_tp::galaxy2_parse_ffn_layers("-1", 39); });
    must_throw([] { llama_galaxy_tp::galaxy2_parse_ffn_layers("40", 39); });
    must_throw([] { llama_galaxy_tp::galaxy2_parse_ffn_layers("18446744073709551615", 39); });
    // Stage and final-layer snapshot selection share the exact same parser.
    require_test(llama_galaxy_tp::layer_selection_contains("0,1", 0), "stage selected 0");
    require_test(llama_galaxy_tp::layer_selection_contains("0,1", 1), "stage selected 1");
    require_test(!llama_galaxy_tp::layer_selection_contains("0,1", 2), "stage excludes 2");
    require_test(!llama_galaxy_tp::layer_selection_contains("10,11", 1), "exact layer token");
    require_test(llama_galaxy_tp::layer_selection_contains("all", 26), "stage select all");
    require_test(!llama_galaxy_tp::layer_selection_contains(nullptr, 0), "stage off");
    require_test(!llama_galaxy_tp::layer_selection_contains("0,1", -1), "negative layer");
    std::cout << "GALAXY_TP_SELFTEST PASS\n";
    return 0;
}

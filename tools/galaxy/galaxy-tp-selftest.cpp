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
    // Arbitrary Galaxy2 layer ownership, shared by model weight placement
    // and graph building, including holes and inclusive ranges.
    const auto layer_set = llama_galaxy_tp::galaxy2_parse_layer_set("0,3,5-7", 27);
    require_test(layer_set.size() == 27 &&
            layer_set[0] && layer_set[3] && layer_set[5] && layer_set[6] &&
            layer_set[7] && !layer_set[1] && !layer_set[4] && !layer_set[8],
            "Galaxy2 noncontiguous FFN mask");
    const auto all_selected = llama_galaxy_tp::galaxy2_parse_layer_set("all", 27);
    const auto none_selected = llama_galaxy_tp::galaxy2_parse_layer_set("none", 27);
    require_test(all_selected.size() == 27 && none_selected.size() == 27,
            "Galaxy2 set dimension");
    for (size_t i = 0; i < 27; ++i) {
        require_test(all_selected[i] == 1 && none_selected[i] == 0,
                "Galaxy2 all/none FFN schedule");
    }
    require_test(llama_galaxy_tp::galaxy2_parse_layer_set("27", 28)[27] == 1,
            "Galaxy2 optional final FFN layer");
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("27", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set(nullptr, 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("0,0", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("1-3,3", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("0,", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set(",0", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("0,,1", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("8-6", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("8-", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("-1", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("1a", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("18446744073709551615", 27); });
    require_test(llama_galaxy_tp::galaxy2_parse_layer_set("none", 0).empty(),
            "no eligible FFN layers allows none");
    must_throw([] { (void) llama_galaxy_tp::galaxy2_parse_layer_set("0", 0); });
    const auto arbitrary = llama_galaxy_tp::galaxy2_resolve_ffn_sp_mask("0,2,5-7", nullptr, 27);
    require_test(arbitrary.size() == 27 && arbitrary[0] && arbitrary[2] &&
            arbitrary[5] && arbitrary[6] && arbitrary[7] && !arbitrary[1] &&
            !arbitrary[3], "Galaxy2 resolve explicit layer set");
    const auto legacy = llama_galaxy_tp::galaxy2_resolve_ffn_sp_mask(nullptr, "2", 27);
    require_test(legacy.size() == 27 && legacy[0] && legacy[1] &&
            !legacy[2], "Galaxy2 legacy leading two FFN layers");
    const auto legacy_all = llama_galaxy_tp::galaxy2_resolve_ffn_sp_mask(nullptr, nullptr, 27);
    require_test(legacy_all.size() == 27 && legacy_all[26],
            "Galaxy2 legacy unset means all eligible");
    must_throw([] { (void) llama_galaxy_tp::galaxy2_resolve_ffn_sp_mask("0", "1", 27); });
    must_throw([] { (void) llama_galaxy_tp::galaxy2_resolve_ffn_sp_mask("27", nullptr, 27); });
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

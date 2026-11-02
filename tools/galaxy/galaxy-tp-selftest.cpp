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
    // Galaxy2 full mode and limited-memory hybrid share the same parser.
    require_test(llama_galaxy_tp::galaxy2_parse_ffn_layers(nullptr, 39) == 39, "galaxy2 all");
    require_test(llama_galaxy_tp::galaxy2_parse_ffn_layers("all", 39) == 39, "galaxy2 all2");
    require_test(llama_galaxy_tp::galaxy2_parse_ffn_layers("1", 39) == 1, "galaxy2 one");
    require_test(llama_galaxy_tp::galaxy2_parse_ffn_layers("0", 39) == 0, "galaxy2 zero");
    must_throw([] { llama_galaxy_tp::galaxy2_parse_ffn_layers("", 39); });
    must_throw([] { llama_galaxy_tp::galaxy2_parse_ffn_layers("-1", 39); });
    must_throw([] { llama_galaxy_tp::galaxy2_parse_ffn_layers("40", 39); });
    must_throw([] { llama_galaxy_tp::galaxy2_parse_ffn_layers("18446744073709551615", 39); });
    std::cout << "GALAXY_TP_SELFTEST PASS\n";
    return 0;
}

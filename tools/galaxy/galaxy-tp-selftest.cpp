#include "../../src/llama-galaxy-tp.h"

#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

using llama_galaxy_tp::parse_counts;
using llama_galaxy_tp::validate_counts;

static void must_throw(const std::function<void()> & fn) {
    bool did_throw = false;
    try { fn(); } catch (const std::invalid_argument &) { did_throw = true; }
    assert(did_throw);
}

int main() {
    assert((parse_counts("20,20", "heads", 2) == std::vector<int64_t>{20, 20}));
    assert((parse_counts("15,25", "heads", 2) == std::vector<int64_t>{15, 25}));
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
    // Galaxy2 only changes FFN placement for selected, non-final layers.
    // Full mode is valid as a graph mode even if the model will not fit in
    // the available hardware memory. Memory is checked at allocation time.
    must_throw([] {
        // Separate parser check via the numeric cap is exercised by the
        // environment-aware integration smoke; do not mutate global env here.
        llama_galaxy_tp::galaxy2_ffn_sp_count(-1);
    });
    std::cout << "GALAXY_TP_SELFTEST PASS\n";
    return 0;
}

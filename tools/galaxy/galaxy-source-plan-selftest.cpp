#include "ggml-backend-galaxy-source-plan.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace ggml_galaxy_source;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
static void invalid(const std::function<void()> & fn) {
    bool caught = false;
    try { fn(); } catch (const std::invalid_argument &) { caught = true; }
    check(caught, "missing expected invalid_argument");
}

int main() {
    try {
        auto tp = plan(method::tp, method::tp, connection::none);
        check(tp.attention_allreduce && tp.ffn_allreduce &&
                !tp.attention_reducescatter && !tp.ffn_reducescatter,
                "source TP should use AllReduce after both blocks");

        auto galaxy = plan(method::tp, method::tp, connection::sp);
        check(galaxy.attention_reducescatter && galaxy.ffn_reducescatter &&
                galaxy.connection1_allgather && galaxy.connection2_allgather &&
                !galaxy.attention_full_weights && !galaxy.ffn_full_weights,
                "source Galaxy scheme-1 TP/SP/TP/SP transitions");

        auto galaxy2 = plan(method::tp, method::sp, connection::sp);
        check(galaxy2.attention_reducescatter && galaxy2.connection2_allgather &&
                galaxy2.ffn_full_weights && !galaxy2.ffn_reducescatter &&
                !galaxy2.connection1_allgather,
                "source Galaxy2 scheme-2 must eliminate FFN-side AG+RS");

        auto sp = plan(method::sp, method::sp, connection::sp);
        check(sp.initial_sequence_scatter && sp.key_allgather &&
                sp.value_allgather && sp.output_allgather &&
                sp.attention_full_weights && sp.ffn_full_weights &&
                !sp.attention_reducescatter && !sp.ffn_reducescatter &&
                !sp.connection1_allgather && !sp.connection2_allgather,
                "source pure SP must gather K/V, not FFN boundaries");

        auto mixed = plan(method::sp, method::tp, connection::sp);
        check(mixed.key_allgather && mixed.value_allgather &&
                mixed.connection1_allgather && mixed.ffn_reducescatter &&
                !mixed.connection2_allgather && mixed.output_allgather,
                "source SP-attention TP-FFN bridge");

        invalid([] { (void) parse_method("NONE"); });
        invalid([] { (void) parse_connection("TP"); });
        invalid([] { (void) plan(method::sp, method::sp, connection::none); });
        invalid([] { (void) plan(method::tp, method::sp, connection::none); });

        std::cout << "GALAXY_SOURCE_PLAN_SELFTEST PASS TP GALAXY GALAXY2 SP MIXED\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "GALAXY_SOURCE_PLAN_SELFTEST FAIL " << e.what() << "\n";
        return 1;
    }
}

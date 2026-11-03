#pragma once

// Galaxy-LM source-defined BERT parallelism choices (model-parallel only).
// This is a pure planning/reference contract, not a model executor.
// Reference: galaxy/models/bert/galaxy_bert_model.py,
// galaxy/models/bert/tp_bert_model.py and sp_bert_model.py.
#include <stdexcept>
#include <string>

namespace ggml_galaxy_source {

enum class method { tp, sp };
enum class connection { none, sp };

inline method parse_method(const std::string & text) {
    if (text == "TP") return method::tp;
    if (text == "SP") return method::sp;
    throw std::invalid_argument("Galaxy source parallel method must be TP or SP");
}

inline connection parse_connection(const std::string & text) {
    if (text == "None") return connection::none;
    if (text == "SP") return connection::sp;
    throw std::invalid_argument("Galaxy source connection mode must be None or SP");
}

struct layer_plan {
    bool initial_sequence_scatter = false;
    bool key_allgather = false;
    bool value_allgather = false;
    bool attention_allreduce = false;
    bool attention_reducescatter = false;
    bool connection1_allgather = false;
    bool ffn_allreduce = false;
    bool ffn_reducescatter = false;
    bool connection2_allgather = false;
    bool output_allgather = false;
    bool attention_full_weights = false;
    bool ffn_full_weights = false;
};

// Match upstream's attention/connection/MLP control flow.
// In source GalaxyBertConnectLayer, a connection SP leaves tokens
// sharded until the next tensor-parallel block needs full input.
// Original SPBertAttention gathers BOTH K and V before local Q attention.
inline layer_plan plan(method attention, method ffn, connection conn) {
    if (conn == connection::none && (attention != method::tp || ffn != method::tp)) {
        throw std::invalid_argument("Galaxy source connection=None requires TP attention and TP FFN");
    }
    layer_plan p;
    p.initial_sequence_scatter = attention == method::sp;
    p.attention_full_weights = attention == method::sp;
    p.ffn_full_weights = ffn == method::sp;
    p.key_allgather = attention == method::sp;
    p.value_allgather = attention == method::sp;
    if (conn == connection::none) {
        p.attention_allreduce = true;
        p.ffn_allreduce = true;
    } else {
        p.attention_reducescatter = attention == method::tp;
        p.connection1_allgather = ffn == method::tp;
        p.ffn_reducescatter = ffn == method::tp;
        p.connection2_allgather = attention == method::tp;
        p.output_allgather = attention == method::sp;
    }
    return p;
}

} // namespace ggml_galaxy_source

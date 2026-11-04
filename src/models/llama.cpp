#include "models.h"
#include "../llama-galaxy-tp.h"
#include "ggml-backend-galaxy-markers.h"

void llama_model_llama::load_arch_hparams(llama_model_loader & ml) {
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (hparams.n_expert == 8) {
        switch (hparams.n_layer()) {
            case 32: type = LLM_TYPE_8x7B; break;
            case 56: type = LLM_TYPE_8x22B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    } else {
        switch (hparams.n_layer()) {
            case 16: type = LLM_TYPE_1B; break; // Llama 3.2 1B
            case 22: type = LLM_TYPE_1B; break;
            case 26: type = LLM_TYPE_3B; break;
            case 28: type = LLM_TYPE_3B; break; // Llama 3.2 3B
            case 30: type = LLM_TYPE_256M; break; // smoldocling 256M
            // granite uses a vocab with len 49152
            case 32: type = n_vocab == 49152 ? LLM_TYPE_3B : (n_vocab < 40000 ? LLM_TYPE_7B : LLM_TYPE_8B); break;
            case 36: type = LLM_TYPE_8B; break; // granite
            case 40: type = LLM_TYPE_13B; break;
            case 48: type = LLM_TYPE_34B; break;
            case 60: type = LLM_TYPE_30B; break;
            case 80: type = hparams.n_head() == hparams.n_head_kv() ? LLM_TYPE_65B : LLM_TYPE_70B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    }
}

void llama_model_llama::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];   //一个层的张量目录

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // optional bias tensors
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }
        else {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }

        if (n_expert == 0) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

            // optional MLP bias
            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

            // For Granite MoE Shared
            if (hparams.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_llama::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph<false>>(*this, params);
}

template <bool embed>
llama_model_llama::graph<embed>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    // Port Galaxy TP/SP and Galaxy2 onto the dense Llama graph without
    // changing RoPE, causal KV cache or the single-token decode path.
    const bool galaxy_sp = !embed && llama_galaxy_tp::sp_enabled() && n_tokens > 1;
    const bool galaxy_sp_final = galaxy_sp && llama_galaxy_tp::sp_include_final_layer();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<embed, llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;

    inp_attn_type * inp_attn = nullptr;
    if constexpr (embed) {
        inp_attn = build_attn_inp_no_cache();
    } else {
        inp_attn = build_attn_inp_kv();
    }
    // 选择不同的模式，embed为true说明是embedding模式，embed为false说明是普通的decoder模式
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    // transformer layers
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;
        // 把当前层的输入保存到结果中，方便后续使用
        // inpl是当前的输入hidden states
        ggml_tensor * inpSA = inpL;
        // inpsa把本层输入另存一份，后续会在ffn中使用
        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
            //归一化
        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);
                //计算Q、K、V，并进行RoPE处理
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
                    //这里和下面进行位置编码
            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
                    // 普通的llama不会启用
            if (hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }
            // Opt-in real-GGUF bridge: preserve the actual post-RoPE
            // Q/K and projected V before the existing cache/Attention path.
            // Llama with Q/K RMSNorm changes the source node naming and is
            // not yet supported by this diagnostic (Llama-3.2-3B has none).
            if (!embed && n_tokens > 1 &&
                    (llama_galaxy_tp::qkv_probe_enabled(il) ||
                     llama_galaxy_tp::causal_live_shadow_enabled(il))) {
                if (hparams.use_kq_norm) {
                    throw std::invalid_argument("Galaxy QKV probe requires Llama without Q/K RMSNorm");
                }
                ggml_set_output(Qcur);
                ggml_set_output(Kcur);
                ggml_set_output(Vcur);
            }
            //计算self-attention的输出
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        const bool galaxy_sp_layer = galaxy_sp && (il + 1 < n_layer || galaxy_sp_final);
        const bool galaxy2_ffn_sp = galaxy_sp_layer &&
                llama_galaxy_tp::galaxy2_ffn_is_sp(il, n_layer);
        if (galaxy_sp_layer) {
            // Attention output is TP_PARTIAL; RS materializes local SP tokens.
            cur = ggml_dup(ctx0, cur);
            ggml_galaxy::mark_transition(cur, ggml_galaxy::sp_transition::reduce_scatter);
            ggml_galaxy::set_stage(cur, ggml_galaxy::sp_stage::attention_rs, il);
            ggml_format_name(cur, "GALAXY_RS_ATT_%d", il);
        }
        if (il == n_layer - 1 && inp_out_ids && !galaxy_sp_final) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * residual_attn = inpSA;
        if (galaxy_sp_layer) {
            // Match Attention RS token ownership for the residual add.
            residual_attn = ggml_dup(ctx0, inpSA);
            ggml_galaxy::mark_transition(residual_attn, ggml_galaxy::sp_transition::scatter_mirrored);
            ggml_galaxy::set_stage(residual_attn, ggml_galaxy::sp_stage::residual_scatter, il);
            ggml_format_name(residual_attn, "GALAXY_SCATTER_RESID_%d", il);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, residual_attn);
        cb(ffn_inp, "ffn_inp", il);
        // Diagnostic only: preserve the comparable post-attention residual
        // activation. TP produces a mirrored tensor; Galaxy uses SP shards.
        // The Meta dump reconstructs the original sequence for comparison.
        const bool galaxy_stage_dump = !embed && n_tokens > 1 &&
                model.layers[il].ffn_gate_inp == nullptr &&
                llama_galaxy_tp::dump_stage_enabled(il);
        if (galaxy_stage_dump) {
            ggml_set_output(ffn_inp);
        }
        // 残差连接
        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
            if (galaxy_stage_dump) {
                ggml_set_output(cur);
            }

            if (galaxy_sp_layer && !galaxy2_ffn_sp) {
                // Galaxy Scheme-1 keeps FFN tensor-parallel.
                // Galaxy2 instead processes local tokens with full FFN weights.
                cur = ggml_dup(ctx0, cur);
                ggml_galaxy::mark_transition(cur, ggml_galaxy::sp_transition::all_gather);
                ggml_galaxy::set_stage(cur, ggml_galaxy::sp_stage::ffn_input_ag, il);
                ggml_format_name(cur, "GALAXY_AG_ATT_%d", il);
            }

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
            // 构建普通FFN
        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cb(cur, "ffn_moe_out", il);
        }
        if (galaxy_sp_layer && !galaxy2_ffn_sp) {
            // FFN TP partial -> SP local tokens for Connection 2.
            cur = ggml_dup(ctx0, cur);
            ggml_galaxy::mark_transition(cur, ggml_galaxy::sp_transition::reduce_scatter);
            ggml_galaxy::set_stage(cur, ggml_galaxy::sp_stage::ffn_output_rs, il);
            ggml_format_name(cur, "GALAXY_RS_FFN_%d", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        if (galaxy_sp_layer) {
            // Next Attention is TP; restore the full hidden sequence.
            cur = ggml_dup(ctx0, cur);
            ggml_galaxy::mark_transition(cur, ggml_galaxy::sp_transition::all_gather);
            ggml_galaxy::set_stage(cur, ggml_galaxy::sp_stage::ffn_output_ag, il);
            ggml_format_name(cur, "GALAXY_AG_FFN_%d", il);
        }
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        if (llama_galaxy_tp::dump_layer_enabled(il) && n_tokens > 1) {
            ggml_set_output(cur);
        }

        // input for next layer
        inpL = cur;
    }
    cur = inpL;
    if (galaxy_sp_final && inp_out_ids) {
        // Full Galaxy source scheme: select output IDs after final AllGather.
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        cb(cur, "galaxy_final_out_ids", -1);
    }

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if constexpr (!embed) {
        // lm_head
        cur = build_lora_mm(model.output, cur, model.output_s);

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
    //创建计算图节点，没有真正的计算，只是把计算图构建出来，后续会在llm_graph_context::compute()中进行计算
}

template struct llama_model_llama::graph<false>;
template struct llama_model_llama::graph<true>;

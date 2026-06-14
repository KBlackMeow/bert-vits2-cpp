#include "bv2_mlx_models.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "bv2_mlx_flow_modules.h"
#include "bv2_mlx_ops.h"

namespace mc = ::mlx::core;

namespace bv2 {
namespace mlx_rt {

// ---------------------------------------------------------------------------
// TextEncoder
// ---------------------------------------------------------------------------

TextEncoderOut text_encoder_apply(const ParamStore& p,
                                  const array& x_ids,
                                  const array& tone_ids,
                                  const array& lang_ids,
                                  const array& bert_zh,
                                  const array& bert_jp,
                                  const array& bert_en,
                                  const array& x_mask,
                                  const array* g,
                                  const MlxConfig& cfg) {
    // Embeddings (sum) -> [B, T_x, hidden]
    array sym_emb  = embedding_apply(p, "enc_p.emb",          x_ids);
    array tone_emb = embedding_apply(p, "enc_p.tone_emb",     tone_ids);
    array lang_emb = embedding_apply(p, "enc_p.language_emb", lang_ids);

    // BERT projections: input shape comes in as [1024, T_x]; we form
    // [1, 1024, T_x] and run conv1d_apply (kernel=1) -> [1, hidden, T_x],
    // then transpose to [B, T_x, hidden] for the additive sum.
    auto bert_to_hidden = [&](const array& bert_T_C, const std::string& key) {
        array t = mc::expand_dims(bert_T_C, 0);                  // [1, 1024, T_x]
        array h = conv1d_apply(p, key, t);                       // [1, hidden, T_x]
        return mc::transpose(h, {0, 2, 1});                      // [1, T_x, hidden]
    };
    array bz = bert_to_hidden(bert_zh, "enc_p.bert_proj");
    array bj = bert_to_hidden(bert_jp, "enc_p.ja_bert_proj");
    array be = bert_to_hidden(bert_en, "enc_p.en_bert_proj");

    array x = mc::add(sym_emb, tone_emb);
    x = mc::add(x, lang_emb);
    x = mc::add(x, bz);
    x = mc::add(x, bj);
    x = mc::add(x, be);
    x = mc::multiply(x, array(std::sqrt(static_cast<float>(cfg.hidden_channels))));

    // [B, T, H] -> [B, H, T]
    x = mc::transpose(x, {0, 2, 1});

    EncoderCfg enc_cfg;
    enc_cfg.hidden_channels = cfg.hidden_channels;
    enc_cfg.filter_channels = cfg.filter_channels;
    enc_cfg.n_heads         = cfg.n_heads;
    enc_cfg.n_layers        = cfg.n_layers;
    enc_cfg.kernel_size     = cfg.kernel_size;
    enc_cfg.window_size     = 4;
    enc_cfg.gin_channels    = cfg.gin_channels;
    enc_cfg.cond_layer_idx  = 2;

    array x_masked = mc::multiply(x, x_mask);
    array enc = encoder_apply(p, "enc_p.encoder", x_masked, x_mask, g, enc_cfg);

    array stats = conv1d_apply(p, "enc_p.proj", enc);            // [B, 2*inter, T_x]
    stats = mc::multiply(stats, x_mask);

    auto sh = stats.shape();
    int B = sh[0], TwoI = sh[1], T_x = sh[2];
    int inter = TwoI / 2;
    array m_p    = mc::slice(stats, {0, 0,     0}, {B, inter,        T_x});
    array logs_p = mc::slice(stats, {0, inter, 0}, {B, 2 * inter,    T_x});
    return {enc, m_p, logs_p};
}

// ---------------------------------------------------------------------------
// DurationPredictor
// ---------------------------------------------------------------------------

array duration_predictor_apply(const ParamStore& p,
                               const std::string& prefix,
                               const array& x_in,
                               const array& x_mask,
                               const array* g,
                               int kernel_size,
                               int filter_channels) {
    (void)filter_channels;
    array x = x_in;
    if (g != nullptr) {
        array g_proj = conv1d_apply(p, prefix + ".cond", *g);   // [B, hidden, 1]
        x = mc::add(x, g_proj);
    }

    Conv1dCfg cfg;
    cfg.padding = kernel_size / 2;

    x = mc::multiply(x, x_mask);
    x = conv1d_apply(p, prefix + ".conv_1", x, cfg);
    x = mc::maximum(x, array(0.0f));
    x = layer_norm_channel_apply(p, prefix + ".norm_1", x);
    // dropout omitted for inference

    x = mc::multiply(x, x_mask);
    x = conv1d_apply(p, prefix + ".conv_2", x, cfg);
    x = mc::maximum(x, array(0.0f));
    x = layer_norm_channel_apply(p, prefix + ".norm_2", x);

    x = mc::multiply(x, x_mask);
    x = conv1d_apply(p, prefix + ".proj", x);                   // [B, 1, T]
    return mc::multiply(x, x_mask);
}

// ---------------------------------------------------------------------------
// StochasticDurationPredictor
// ---------------------------------------------------------------------------

array stochastic_duration_predictor_apply(const ParamStore& p,
                                          const std::string& prefix,
                                          const array& x_in,
                                          const array& x_mask,
                                          const array& z_init,
                                          const array* g,
                                          int filter_channels,
                                          int kernel_size,
                                          int n_flows) {
    array x = conv1d_apply(p, prefix + ".pre", x_in);            // [B, F, T]
    if (g != nullptr) {
        array gp = conv1d_apply(p, prefix + ".cond", *g);
        x = mc::add(x, gp);
    }
    x = dds_conv_apply(p, prefix + ".convs", x, x_mask, /*n_layers*/ 3, kernel_size);
    x = conv1d_apply(p, prefix + ".proj", x);
    x = mc::multiply(x, x_mask);

    // Reverse the `flows` list and drop the v-flow (Python `flows[:-2]+[flows[-1]]`).
    // The list was stored as [ElementwiseAffine, ConvFlow, Flip, ConvFlow, Flip, ...]
    // i.e. positions: 0 = EA, 1..2N = (CF, Flip) pairs.
    //
    // After `reversed(flows)` the order is the reversed full list. Then
    // `[:-2] + [flows[-1]]` drops the original first ConvFlow's pair from
    // the end (a "useless v-flow"), keeping ElementwiseAffine at the very
    // end of the iteration.
    //
    // For our v2.3 SDP (n_flows=4), `flows` has 1 + 2*4 = 9 entries, so the
    // walk below visits indices [8, 7, 6, 5, 4, 3, 2, 0] (skip 1).
    int total = 1 + 2 * n_flows;
    std::vector<int> reversed_indices;
    for (int i = total - 1; i >= 0; --i) reversed_indices.push_back(i);
    if (reversed_indices.size() >= 2) {
        // drop the second-to-last (== original flows[1] == first ConvFlow)
        reversed_indices.erase(reversed_indices.end() - 2);
    }

    // `flows` are operating on z (2 channels) with x as conditioning (`g=x`).
    array z = z_init;  // [B, 2, T_x]
    array g_for_flows = x;  // SDP: pass x as `g` to ConvFlow's DDSConv.
    int num_bins = 10;
    float tail_bound = 5.0f;

    for (int idx : reversed_indices) {
        std::string flow_prefix = prefix + ".flows." + std::to_string(idx);
        if (idx == 0) {
            z = elementwise_affine_inverse(p, flow_prefix, z, x_mask);
        } else if (idx % 2 == 1) {
            // ConvFlow at odd indices.
            z = conv_flow_inverse(p, flow_prefix,
                                  z, x_mask, &g_for_flows,
                                  filter_channels, kernel_size,
                                  /*n_layers*/ 3, num_bins, tail_bound);
        } else {
            // Flip at even indices >= 2.
            z = flip_channels(z);
        }
    }

    // logw = z[:, :1, :]
    auto sh = z.shape();
    int B = sh[0], T = sh[2];
    return mc::slice(z, {0, 0, 0}, {B, 1, T});
}

// ---------------------------------------------------------------------------
// TransformerCouplingBlock (reverse)
// ---------------------------------------------------------------------------

array transformer_coupling_block_inverse(const ParamStore& p,
                                         const std::string& prefix,
                                         const array& x_in,
                                         const array& x_mask,
                                         const array* g,
                                         const MlxConfig& cfg) {
    EncoderCfg enc_cfg;
    enc_cfg.hidden_channels = cfg.hidden_channels;
    enc_cfg.filter_channels = cfg.filter_channels;
    enc_cfg.n_heads         = cfg.n_heads;
    enc_cfg.n_layers        = cfg.n_layers_trans_flow;
    enc_cfg.kernel_size     = 5;     // hard-coded in V230 TransformerCouplingBlock
    enc_cfg.window_size     = 4;
    enc_cfg.gin_channels    = cfg.gin_channels;
    enc_cfg.cond_layer_idx  = 2;

    int half = cfg.inter_channels / 2;
    int total_flows = 2 * cfg.n_flow_layer;  // 8 entries: TCL, Flip, TCL, Flip, ...

    array z = x_in;
    for (int i = total_flows - 1; i >= 0; --i) {
        std::string fp = prefix + ".flows." + std::to_string(i);
        if (i % 2 == 1) {
            // Flip at odd indices.
            z = flip_channels(z);
        } else {
            z = transformer_coupling_layer_inverse(p, fp, z, x_mask, g,
                                                   enc_cfg, half);
        }
    }
    return z;
}

// ---------------------------------------------------------------------------
// Generator (HiFi-GAN)
// ---------------------------------------------------------------------------

array generator_apply(const ParamStore& p,
                      const std::string& prefix,
                      const array& x_in,
                      const array* g,
                      const MlxConfig& cfg) {
    Conv1dCfg pre_cfg;
    pre_cfg.padding = 3;  // (7-1)/2 -> matches PyTorch padding=3
    array x = conv1d_apply(p, prefix + ".conv_pre", x_in, pre_cfg);
    if (g != nullptr) {
        array gp = conv1d_apply(p, prefix + ".cond", *g);
        x = mc::add(x, gp);
    }

    int num_upsamples = static_cast<int>(cfg.upsample_rates.size());
    int num_kernels   = static_cast<int>(cfg.resblock_kernel_sizes.size());

    for (int i = 0; i < num_upsamples; ++i) {
        x = leaky_relu_01(x);
        int u = cfg.upsample_rates[i];
        int k = cfg.upsample_kernel_sizes[i];
        int padding = (k - u) / 2;
        x = conv_transpose1d_apply(p, prefix + ".ups." + std::to_string(i),
                                   x, /*stride*/ u, /*padding*/ padding);

        // Sum of `num_kernels` ResBlock1 outputs, divided by num_kernels.
        // `mlx::core::array` has no default constructor, so accumulate via
        // `std::optional` and unwrap once at the end.
        std::optional<array> xs;
        for (int j = 0; j < num_kernels; ++j) {
            int k_size = cfg.resblock_kernel_sizes[j];
            const auto& dilations = cfg.resblock_dilation_sizes[j];
            std::string rb = prefix + ".resblocks." + std::to_string(i * num_kernels + j);
            // ResBlock1 forward
            array y = x;
            for (size_t d_idx = 0; d_idx < dilations.size(); ++d_idx) {
                int dil = dilations[d_idx];
                int pad = ((k_size - 1) * dil) / 2;
                Conv1dCfg cfg1;
                cfg1.padding  = pad;
                cfg1.dilation = dil;
                array yt = leaky_relu_01(y);
                yt = conv1d_apply(p, rb + ".convs1." + std::to_string(d_idx),
                                  yt, cfg1);
                yt = leaky_relu_01(yt);
                Conv1dCfg cfg2;
                cfg2.padding = (k_size - 1) / 2;
                cfg2.dilation = 1;
                yt = conv1d_apply(p, rb + ".convs2." + std::to_string(d_idx),
                                  yt, cfg2);
                y = mc::add(y, yt);
            }
            xs = xs ? mc::add(*xs, y) : y;
        }
        x = mc::divide(*xs, array(static_cast<float>(num_kernels)));
    }
    x = leaky_relu_01(x);

    // conv_post: kernel 7, padding 3, no bias
    Conv1dCfg post_cfg;
    post_cfg.padding = 3;
    x = conv1d_apply(p, prefix + ".conv_post", x, post_cfg);
    return mc::tanh(x);
}

// ---------------------------------------------------------------------------
// Top-level inference
// ---------------------------------------------------------------------------

namespace {

array make_int_array(const std::vector<int64_t>& v, int batch) {
    int T = static_cast<int>(v.size());
    std::vector<int32_t> data(T);
    for (int i = 0; i < T; ++i) {
        data[i] = static_cast<int32_t>(v[i]);
    }
    return mc::reshape(
        array(data.data(), {T}, mc::int32),
        {batch, T});
}

array make_bert_T_C(const std::vector<float>& v, int Tx) {
    // Caller (`Tensor zeros_bert(phones)` in bv2_tts.cpp) is row-major
    // [T_x, 1024]. conv1d_apply expects PyTorch channels-first layout
    // [B, C_in, T] = [1, 1024, T_x], so build [T_x, 1024] then transpose.
    if (static_cast<int>(v.size()) != Tx * 1024) {
        throw std::runtime_error("bert tensor size mismatch");
    }
    array t = array(v.data(), {Tx, 1024}, mc::float32);
    return mc::transpose(t, {1, 0});
}

} // namespace

array synthesizer_infer(const ParamStore& p,
                        const MlxInferInputs& in,
                        const MlxConfig& cfg) {
    int Tx = static_cast<int>(in.phones.size());
    if (Tx == 0) {
        throw std::runtime_error("phones is empty");
    }

    array x_ids   = make_int_array(in.phones,    1);
    array tone_id = make_int_array(in.tones,     1);
    array lang_id = make_int_array(in.languages, 1);
    array bert_zh = make_bert_T_C(in.bert_zh, Tx);
    array bert_jp = make_bert_T_C(in.bert_jp, Tx);
    array bert_en = make_bert_T_C(in.bert_en, Tx);

    // x_mask = ones [1, 1, T_x]
    array x_mask = mc::ones({1, 1, Tx}, mc::float32);

    // Speaker embedding
    int sid = in.speaker_id;
    array sid_arr = array(static_cast<int32_t>(sid));
    sid_arr = mc::reshape(sid_arr, {1});
    array g = embedding_apply(p, "emb_g", sid_arr);  // [1, gin]
    g = mc::expand_dims(g, -1);                      // [1, gin, 1]

    auto enc_out = text_encoder_apply(p,
                                      x_ids, tone_id, lang_id,
                                      bert_zh, bert_jp, bert_en,
                                      x_mask, &g, cfg);

    // Duration prediction.
    array key = make_key(in.seed);
    array zinput = randn(key, {1, 2, Tx});
    zinput = mc::multiply(zinput, array(in.noise_scale_w));

    array logw_sdp = stochastic_duration_predictor_apply(
        p, "sdp", enc_out.x, x_mask, zinput, &g,
        cfg.sdp_filter_channels, cfg.sdp_kernel_size, cfg.sdp_n_flows);
    array logw_dp  = duration_predictor_apply(
        p, "dp", enc_out.x, x_mask, &g,
        cfg.dp_kernel_size, cfg.dp_filter_channels);

    array logw = mc::add(mc::multiply(logw_sdp, array(in.sdp_ratio)),
                         mc::multiply(logw_dp,  array(1.0f - in.sdp_ratio)));
    array w = mc::multiply(mc::exp(logw), x_mask);
    w = mc::multiply(w, array(in.length_scale));
    array w_ceil = mc::ceil(w);

    // y_lengths = clamp(sum(w_ceil), min=1).long()
    array y_len_f = mc::sum(w_ceil, /*axes*/ {1, 2});      // [B]
    y_len_f = mc::maximum(y_len_f, array(1.0f));
    array y_len_i = mc::astype(y_len_f, mc::int32);

    // Materialise the scalar so we can build a fixed-shape mask.
    mc::eval(y_len_i);
    int Ty = static_cast<int>(y_len_i.data<int32_t>()[0]);
    if (Ty <= 0) Ty = 1;

    array y_mask = sequence_mask_f32(y_len_i, Ty);          // [B, Ty]
    y_mask = mc::expand_dims(y_mask, 1);                    // [B, 1, Ty]

    // attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1) -> [B, 1, Ty, Tx]
    array x_mask_u = mc::expand_dims(x_mask, 2);            // [B, 1, 1, Tx]
    array y_mask_u = mc::expand_dims(y_mask, -1);           // [B, 1, Ty, 1]
    array attn_mask = mc::multiply(x_mask_u, y_mask_u);

    // attn = generate_path(w_ceil, attn_mask) [B, 1, Ty, Tx]
    array attn = generate_path(w_ceil, attn_mask);

    // m_p, logs_p: matmul attn.squeeze(1) [B, Ty, Tx] with m_p.transpose(1,2) [B, Tx, inter]
    // -> [B, Ty, inter] -> transpose(1,2) -> [B, inter, Ty]
    array attn_2d = mc::squeeze(attn, /*axis*/ 1);          // [B, Ty, Tx]
    array m_p_aligned    = mc::matmul(attn_2d, mc::transpose(enc_out.m_p,    {0, 2, 1}));
    array logs_p_aligned = mc::matmul(attn_2d, mc::transpose(enc_out.logs_p, {0, 2, 1}));
    m_p_aligned    = mc::transpose(m_p_aligned,    {0, 2, 1});  // [B, inter, Ty]
    logs_p_aligned = mc::transpose(logs_p_aligned, {0, 2, 1});

    // z_p = m_p + randn * exp(logs_p) * noise_scale
    array eps = randn(key, m_p_aligned.shape());
    array z_p = mc::add(m_p_aligned,
                        mc::multiply(eps,
                                     mc::multiply(mc::exp(logs_p_aligned),
                                                  array(in.noise_scale))));

    if (!cfg.use_transformer_flow) {
        // Fallback: ResidualCouplingBlock (not used in v2.3 default).
        // Not implemented; the converter / config check in load() prevents this.
        throw std::runtime_error("ResidualCouplingBlock not implemented (use_transformer_flow=False)");
    }
    array z = transformer_coupling_block_inverse(p, "flow", z_p, y_mask, &g, cfg);

    array z_in = mc::multiply(z, y_mask);
    array audio = generator_apply(p, "dec", z_in, &g, cfg);   // [1, 1, T_audio]
    return audio;
}

} // namespace mlx_rt
} // namespace bv2

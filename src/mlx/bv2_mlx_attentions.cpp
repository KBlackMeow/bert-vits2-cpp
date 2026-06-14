#include "bv2_mlx_attentions.h"
#include "bv2_mlx_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <mlx/array.h>
#include <mlx/ops.h>

namespace mc = ::mlx::core;

namespace bv2 {
namespace mlx_rt {

// ---------------------------------------------------------------------------
// Relative position embedding helpers
// ---------------------------------------------------------------------------

namespace {

// _get_relative_embeddings(rel_emb, length): returns [n_heads_rel, 2*L-1, k_chan].
//   rel_emb: [n_heads_rel, 2W+1, k_chan]
array get_relative_embeddings(const array& rel_emb, int length, int window_size) {
    int max_rel = 2 * window_size + 1;
    (void)max_rel;
    int pad_length = std::max(length - (window_size + 1), 0);
    int slice_start = std::max((window_size + 1) - length, 0);
    int slice_end = slice_start + 2 * length - 1;

    array padded = rel_emb;
    if (pad_length > 0) {
        padded = mc::pad(rel_emb, /*axes*/ {1},
                         /*low_pad*/ {pad_length}, /*high_pad*/ {pad_length},
                         /*value*/ array(0.0f));
    }
    auto sh = padded.shape();
    return mc::slice(padded,
                     /*starts*/ {0, slice_start, 0},
                     /*stops*/  {(int)sh[0], slice_end, (int)sh[2]});
}

// Convert relative-position scores [B, H, L, 2L-1] to absolute scores
// [B, H, L, L] using the canonical "pad+reshape+slice" trick from
// `attentions._relative_position_to_absolute_position`.
array rel_to_abs(const array& x) {
    auto sh = x.shape();
    if (sh.size() != 4) {
        throw std::runtime_error("rel_to_abs: expected 4D");
    }
    int B = sh[0], H = sh[1], L = sh[2], W = sh[3];  // W = 2L - 1
    if (W != 2 * L - 1) {
        throw std::runtime_error("rel_to_abs: width mismatch");
    }
    // pad last axis: (0, 1)
    array y = mc::pad(x, /*axes*/ {3}, /*low*/ {0}, /*high*/ {1},
                      /*value*/ array(0.0f));                 // [B, H, L, 2L]
    // flatten last two dims
    array flat = mc::reshape(y, {B, H, L * 2 * L});
    // pad to add L-1 trailing zeros
    flat = mc::pad(flat, /*axes*/ {2}, /*low*/ {0}, /*high*/ {L - 1},
                   /*value*/ array(0.0f));                    // [B, H, L*2L + L - 1]
    array reshaped = mc::reshape(flat, {B, H, L + 1, 2 * L - 1});
    // slice [:, :, :L, L-1:]
    return mc::slice(reshaped,
                     /*starts*/ {0, 0, 0, L - 1},
                     /*stops*/ {B, H, L, 2 * L - 1});           // [B, H, L, L]
}

// Inverse of `rel_to_abs`: from abs attention probs [B, H, L, L] to
// relative-position weights [B, H, L, 2L-1].
array abs_to_rel(const array& x) {
    auto sh = x.shape();
    int B = sh[0], H = sh[1], L = sh[2];
    array y = mc::pad(x, /*axes*/ {3}, /*low*/ {0}, /*high*/ {L - 1},
                      /*value*/ array(0.0f));                 // [B, H, L, 2L-1]
    array flat = mc::reshape(y, {B, H, L * L + L * (L - 1)});
    flat = mc::pad(flat, /*axes*/ {2}, /*low*/ {L}, /*high*/ {0},
                   /*value*/ array(0.0f));
    array reshaped = mc::reshape(flat, {B, H, L, 2 * L});
    return mc::slice(reshaped,
                     /*starts*/ {0, 0, 0, 1},
                     /*stops*/ {B, H, L, 2 * L});             // [B, H, L, 2L-1]
}

} // namespace

// ---------------------------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------------------------

array mha_apply(const ParamStore& p,
                const std::string& prefix,
                const array& x,
                const array& attn_mask,
                int n_heads,
                int window_size) {
    array q = conv1d_apply(p, prefix + ".conv_q", x);   // [B, C, T]
    array k = conv1d_apply(p, prefix + ".conv_k", x);
    array v = conv1d_apply(p, prefix + ".conv_v", x);

    auto qsh = q.shape();
    int B = qsh[0], C = qsh[1], T_t = qsh[2];
    int k_chan = C / n_heads;
    int T_s = T_t;

    auto reshape_to_heads = [&](const array& t, int Tt) {
        // [B, C, T] -> [B, n_heads, k_chan, T] -> [B, n_heads, T, k_chan]
        array y = mc::reshape(t, {B, n_heads, k_chan, Tt});
        return mc::transpose(y, {0, 1, 3, 2});
    };
    array Q = reshape_to_heads(q, T_t);
    array K = reshape_to_heads(k, T_s);
    array V = reshape_to_heads(v, T_s);

    float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(k_chan));
    array Q_scaled = mc::multiply(Q, array(inv_sqrt));

    // scores = Q_scaled @ K^T  -> [B, H, T_t, T_s]
    array scores = mc::matmul(Q_scaled, mc::transpose(K, {0, 1, 3, 2}));

    // relative-position bias on keys.
    if (window_size > 0) {
        const array& emb_rel_k = p.at(prefix + ".emb_rel_k"); // [Hr, 2W+1, k_chan]
        array key_rel = get_relative_embeddings(emb_rel_k, T_s, window_size); // [Hr, 2L-1, k_chan]
        // Q [B, H, L, k] * y^T [Hr or 1, k, 2L-1]
        array key_rel_t = mc::transpose(key_rel, {0, 2, 1}); // [Hr, k, 2L-1]
        array key_rel_b = mc::expand_dims(key_rel_t, 0);     // [1, Hr, k, 2L-1]
        array rel_logits = mc::matmul(Q_scaled, key_rel_b);  // [B, H, L, 2L-1]
        scores = mc::add(scores, rel_to_abs(rel_logits));
    }

    if (attn_mask.size() > 0) {
        // scores = scores.masked_fill(mask == 0, -1e4)
        array neg = mc::full(scores.shape(), array(-1e4f), scores.dtype());
        array zero = mc::full(attn_mask.shape(), array(0.0f), attn_mask.dtype());
        array m = mc::equal(attn_mask, zero);
        scores = mc::where(m, neg, scores);
    }

    array p_attn = mc::softmax(scores, /*axes*/ {-1});

    array out = mc::matmul(p_attn, V);                   // [B, H, T_t, k_chan]

    if (window_size > 0) {
        const array& emb_rel_v = p.at(prefix + ".emb_rel_v");
        array val_rel = get_relative_embeddings(emb_rel_v, T_s, window_size);   // [Hr, 2L-1, k]
        array rel_w = abs_to_rel(p_attn);                                       // [B, H, L, 2L-1]
        array val_rel_b = mc::expand_dims(val_rel, 0);                          // [1, Hr, 2L-1, k]
        array contrib = mc::matmul(rel_w, val_rel_b);                           // [B, H, L, k]
        out = mc::add(out, contrib);
    }

    // [B, H, T_t, k_chan] -> [B, H, k_chan, T_t] -> [B, C, T_t]
    out = mc::transpose(out, {0, 1, 3, 2});
    out = mc::reshape(out, {B, C, T_t});

    return conv1d_apply(p, prefix + ".conv_o", out);
}

// ---------------------------------------------------------------------------
// FFN
// ---------------------------------------------------------------------------

array ffn_apply(const ParamStore& p,
                const std::string& prefix,
                const array& x,
                const array& x_mask,
                int kernel_size) {
    array h = mc::multiply(x, x_mask);
    h = pad_same_1d(h, kernel_size);
    h = conv1d_apply(p, prefix + ".conv_1", h);  // [B, F, T]
    h = mc::maximum(h, array(0.0f));             // ReLU (default activation in v2.3)
    h = mc::multiply(h, x_mask);
    h = pad_same_1d(h, kernel_size);
    h = conv1d_apply(p, prefix + ".conv_2", h);  // [B, C, T]
    return mc::multiply(h, x_mask);
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

array encoder_apply(const ParamStore& p,
                    const std::string& prefix,
                    const array& x_in,
                    const array& x_mask,
                    const array* g,
                    const EncoderCfg& cfg) {
    // attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1) -> [B, 1, T, T]
    array attn_mask = mc::multiply(mc::expand_dims(x_mask, 2),
                                   mc::expand_dims(x_mask, -1));

    array x = mc::multiply(x_in, x_mask);
    bool have_g = (g != nullptr);

    for (int i = 0; i < cfg.n_layers; ++i) {
        if (have_g && i == cfg.cond_layer_idx) {
            // g [B, gin, 1]; spk_emb_linear acts on g.transpose(1, 2): [B, 1, gin] -> [B, 1, hidden]
            array g_t = mc::transpose(*g, {0, 2, 1});                     // [B, 1, gin]
            array g_h = linear_apply(p, prefix + ".spk_emb_linear", g_t); // [B, 1, hidden]
            g_h = mc::transpose(g_h, {0, 2, 1});                          // [B, hidden, 1]
            x = mc::add(x, g_h);
            x = mc::multiply(x, x_mask);
        }
        std::string base = prefix + ".attn_layers." + std::to_string(i);
        array y = mha_apply(p, base, x, attn_mask, cfg.n_heads, cfg.window_size);
        x = mc::add(x, y);
        x = layer_norm_channel_apply(p, prefix + ".norm_layers_1." + std::to_string(i), x);

        std::string ff_base = prefix + ".ffn_layers." + std::to_string(i);
        array y2 = ffn_apply(p, ff_base, x, x_mask, cfg.kernel_size);
        x = mc::add(x, y2);
        x = layer_norm_channel_apply(p, prefix + ".norm_layers_2." + std::to_string(i), x);
    }
    return mc::multiply(x, x_mask);
}

} // namespace mlx_rt
} // namespace bv2

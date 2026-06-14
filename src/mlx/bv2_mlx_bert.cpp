#include "bv2_mlx_bert.h"

#include "bv2_mlx_modules.h"
#include "bv2_mlx_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace mc = ::mlx::core;

namespace bv2 {
namespace mlx_rt {

// ============================================================================
// Helpers
// ============================================================================

namespace {

// Wrapper around `mc::fast::layer_norm` that works on the LAST axis (HF/BERT
// convention). Reads `<prefix>.weight` and `<prefix>.bias`.
array layer_norm_last_axis(const ParamStore& p,
                           const std::string& prefix,
                           const array& x,
                           float eps) {
    // Cast to fp32 so layer_norm weight/bias always match activation dtype.
    array w = mc::astype(p.at(prefix + ".weight"), mc::float32);
    array b = mc::astype(p.at(prefix + ".bias"),   mc::float32);
    return mc::fast::layer_norm(x, w, b, eps);
}

// Convert [B, T] attention mask (1=keep, 0=pad) into an additive bias that
// can be added to attention logits before softmax. Returns shape [B, 1, 1, T].
array attention_additive_mask(const array& mask_bt) {
    // bias = (mask - 1) * 1e4  -> 0 where mask=1, -1e4 where mask=0
    array m = mc::astype(mask_bt, mc::float32);
    array bias = mc::multiply(mc::subtract(m, array(1.0f)), array(1e4f));
    bias = mc::expand_dims(bias, 1);   // [B, 1, T]
    bias = mc::expand_dims(bias, 1);   // [B, 1, 1, T]
    return bias;
}

// Reshape [B, T, H] into [B, n_heads, T, head_size].
array split_heads(const array& x, int n_heads) {
    auto sh = x.shape();
    int B = sh[0], T = sh[1], H = sh[2];
    int head_size = H / n_heads;
    array y = mc::reshape(x, {B, T, n_heads, head_size});
    return mc::transpose(y, {0, 2, 1, 3});       // [B, n_heads, T, head_size]
}

// Reshape [B, n_heads, T, head_size] back to [B, T, H].
array merge_heads(const array& x) {
    auto sh = x.shape();
    int B = sh[0], n_heads = sh[1], T = sh[2], head_size = sh[3];
    array y = mc::transpose(x, {0, 2, 1, 3});    // [B, T, n_heads, head_size]
    return mc::reshape(y, {B, T, n_heads * head_size});
}

// ============================================================================
// Embedding sub-modules
// ============================================================================

// BERT (kBert) embeddings:
//   word_emb + position_emb + token_type_emb -> LayerNorm
array bert_embeddings(const ParamStore& p,
                      const array& input_ids,           // [B, T] int
                      const array& token_type_ids,      // [B, T] int (zeros if unused)
                      const BertConfig& cfg) {
    array word_e = embedding_apply(p, "embeddings.word_embeddings", input_ids);

    int T = static_cast<int>(input_ids.shape().back());
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; ++i) pos[i] = i;
    array pos_ids = array(pos.data(), {1, T}, mc::int32);
    array pos_e = embedding_apply(p, "embeddings.position_embeddings", pos_ids);
    word_e = mc::add(word_e, pos_e);

    if (cfg.type_vocab_size > 0 &&
        p.has("embeddings.token_type_embeddings.weight")) {
        array tt_e = embedding_apply(p, "embeddings.token_type_embeddings", token_type_ids);
        word_e = mc::add(word_e, tt_e);
    }

    return layer_norm_last_axis(p, "embeddings.LayerNorm", word_e, cfg.layer_norm_eps);
}

// DeBERTa-v2 embeddings: word_emb (+ position_emb if position_biased_input) -> LayerNorm.
array deberta_embeddings(const ParamStore& p,
                         const array& input_ids,        // [B, T]
                         const BertConfig& cfg) {
    array x = embedding_apply(p, "embeddings.word_embeddings", input_ids);
    if (cfg.position_biased_input &&
        p.has("embeddings.position_embeddings.weight")) {
        int T = static_cast<int>(input_ids.shape().back());
        std::vector<int32_t> pos(T);
        for (int i = 0; i < T; ++i) pos[i] = i;
        array pos_ids = array(pos.data(), {1, T}, mc::int32);
        array pe = embedding_apply(p, "embeddings.position_embeddings", pos_ids);
        x = mc::add(x, pe);
    }
    return layer_norm_last_axis(p, "embeddings.LayerNorm", x, cfg.layer_norm_eps);
}

// ============================================================================
// Vanilla BERT (kBert) self-attention
// ============================================================================
//
// HF naming under `bert.encoder.layer.{i}`:
//   attention.self.query/key/value.{weight,bias}
//   attention.output.dense.{weight,bias}
//   attention.output.LayerNorm.{weight,bias}
//   intermediate.dense.{weight,bias}
//   output.dense.{weight,bias}
//   output.LayerNorm.{weight,bias}

array bert_self_attention(const ParamStore& p,
                          const std::string& prefix,
                          const array& hidden,        // [B, T, H]
                          const array& add_mask,      // [B, 1, 1, T]
                          const BertConfig& cfg) {
    array q = linear_apply(p, prefix + ".attention.self.query", hidden);
    array k = linear_apply(p, prefix + ".attention.self.key",   hidden);
    array v = linear_apply(p, prefix + ".attention.self.value", hidden);

    array Q = split_heads(q, cfg.num_attention_heads);
    array K = split_heads(k, cfg.num_attention_heads);
    array V = split_heads(v, cfg.num_attention_heads);

    int head_size = cfg.hidden_size / cfg.num_attention_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    // Fused Metal kernel: avoids 5 separate shader dispatches.
    // add_mask [B, 1, 1, T] broadcasts to [B, n_heads, T, T].
    array context = mc::fast::scaled_dot_product_attention(
        Q, K, V, scale, add_mask);                                     // [B, n_h, T, head]
    array merged = merge_heads(context);                                // [B, T, H]

    array out = linear_apply(p, prefix + ".attention.output.dense", merged);
    array residual = mc::add(out, hidden);
    return layer_norm_last_axis(p, prefix + ".attention.output.LayerNorm",
                                residual, cfg.layer_norm_eps);
}

// FFN block (shared between BERT and DeBERTa-v2).
array bert_ffn_block(const ParamStore& p,
                     const std::string& prefix,
                     const array& hidden,
                     const BertConfig& cfg) {
    array h = linear_apply(p, prefix + ".intermediate.dense", hidden);
    h = gelu(h);                                   // exact GELU (HF "gelu")
    h = linear_apply(p, prefix + ".output.dense", h);
    array residual = mc::add(h, hidden);
    return layer_norm_last_axis(p, prefix + ".output.LayerNorm",
                                residual, cfg.layer_norm_eps);
}

array bert_layer(const ParamStore& p,
                 const std::string& prefix,
                 const array& hidden,
                 const array& add_mask,
                 const BertConfig& cfg) {
    array attn_out = bert_self_attention(p, prefix, hidden, add_mask, cfg);
    return bert_ffn_block(p, prefix, attn_out, cfg);
}

// ============================================================================
// DeBERTa-v2 disentangled self-attention
// ============================================================================
//
// Naming under `deberta.encoder.layer.{i}`:
//   attention.self.query_proj.{weight,bias}
//   attention.self.key_proj.{weight,bias}
//   attention.self.value_proj.{weight,bias}
//   attention.self.pos_query_proj.{weight,bias}   -- only if !share_att_key && p2c
//   attention.self.pos_key_proj.{weight,bias}     -- only if !share_att_key && c2p
//   attention.output.dense + LayerNorm
//   intermediate.dense
//   output.dense + LayerNorm
// Shared: deberta.encoder.rel_embeddings.weight  [2*position_buckets, H]
//         deberta.encoder.LayerNorm.{weight,bias}  -- only if norm_rel_ebd_layer_norm

// Build the raw relative-position matrix for query length q_len and key
// length k_len:  rel[i, j] = i - j, shape [q_len, k_len].
array build_relative_position_matrix(int q_len, int k_len) {
    std::vector<int32_t> q(q_len);
    for (int i = 0; i < q_len; ++i) q[i] = i;
    std::vector<int32_t> k(k_len);
    for (int j = 0; j < k_len; ++j) k[j] = j;
    array qa = array(q.data(), {q_len, 1}, mc::int32);
    array ka = array(k.data(), {1, k_len}, mc::int32);
    return mc::subtract(qa, ka);                // [q_len, k_len]
}

// `make_log_bucket_position` from HF: log-spaced bucketing of relative
// positions. Input/output shape is [q_len, k_len], dtype int32.
array make_log_bucket_position(const array& rel_pos,
                               int bucket_size,
                               int max_position) {
    // sign = sign(rel_pos)
    array sign = mc::sign(mc::astype(rel_pos, mc::float32));
    int mid = bucket_size / 2;
    // abs_pos = where((rel_pos < mid) & (rel_pos > -mid), mid - 1, abs(rel_pos))
    array abs_pos_full = mc::abs(rel_pos);
    array lt_mid = mc::less(rel_pos, array(mid));
    array gt_neg = mc::greater(rel_pos, array(-mid));
    array near_zero = mc::logical_and(lt_mid, gt_neg);
    array fill = mc::full(rel_pos.shape(), array(mid - 1), rel_pos.dtype());
    array abs_pos = mc::where(near_zero, fill, abs_pos_full);
    // log_pos = ceil(log(abs_pos / mid) / log((max_position - 1) / mid) * (mid - 1)) + mid
    float log_denom = std::log(
        static_cast<float>(max_position - 1) / static_cast<float>(mid));
    array logarg = mc::divide(mc::astype(abs_pos, mc::float32),
                              array(static_cast<float>(mid)));
    array log_pos_f = mc::ceil(
        mc::add(
            mc::multiply(
                mc::divide(mc::log(logarg), array(log_denom)),
                array(static_cast<float>(mid - 1))),
            array(static_cast<float>(mid))));
    array log_pos = mc::astype(log_pos_f, mc::int32);
    // bucket_pos = where(abs_pos <= mid, rel_pos, log_pos * sign)
    array le_mid = mc::less_equal(abs_pos, array(mid));
    array log_signed = mc::astype(mc::multiply(log_pos_f, sign), mc::int32);
    return mc::where(le_mid, rel_pos, log_signed);
}

// Compute, ONCE per layer, the bucketed relative position matrix used by
// disentangled attention. Returns float32 with `make_log_bucket_position`
// already applied (keep it int-valued but stored as int32). Shape [q, k].
array deberta_bucketed_rel(int q_len, int k_len, const BertConfig& cfg) {
    // Pass `bucket_size = position_buckets` to mirror HF
    // `build_relative_position(query_size, key_size, bucket_size=self.position_buckets,
    //  max_position=self.max_relative_positions)` exactly.
    int bucket = cfg.position_buckets;
    int max_pos = (cfg.max_relative_positions > 0)
        ? cfg.max_relative_positions : bucket;
    array rel = build_relative_position_matrix(q_len, k_len);
    return make_log_bucket_position(rel, bucket, max_pos);   // [q, k] int32
}

// Run the DeBERTa-v2 disentangled attention for one layer.
//   hidden:    [B, T, H]
//   rel_emb:   [2*pb, H]   (already LayerNorm'd if norm_rel_ebd_layer_norm)
//   bucketed:  [T, T] int32 - HF `make_log_bucket_position(q-k)` matrix.
//                            Each cell is in [-pb+1, pb-1].
//   add_mask:  [B, 1, 1, T]
//
// Mirrors `DisentangledSelfAttention.disentangled_attention_bias` from
// transformers/models/deberta_v2/modeling_deberta_v2.py. att_span equals
// `position_buckets` (= pos_ebd_size when buckets > 0). For square
// attention (q_len == k_len) we can reuse the same bucketed matrix for
// both c2p (index = clamp(rel + att_span)) and p2c (index = clamp(-rel
// + att_span)) gathers.
array deberta_self_attention(const ParamStore& p,
                             const std::string& prefix,
                             const array& hidden,
                             const array& rel_emb,
                             const array& bucketed,
                             const array& add_mask,
                             const BertConfig& cfg) {
    int n_heads   = cfg.num_attention_heads;
    int head_size = cfg.hidden_size / n_heads;

    array Qc = linear_apply(p, prefix + ".attention.self.query_proj", hidden);
    array Kc = linear_apply(p, prefix + ".attention.self.key_proj",   hidden);
    array Vc = linear_apply(p, prefix + ".attention.self.value_proj", hidden);

    array Q = split_heads(Qc, n_heads);   // [B, n_h, T, head]
    array K = split_heads(Kc, n_heads);
    array V = split_heads(Vc, n_heads);

    // HF: scale_factor = 1 + #pos_att_type, attn = matmul / sqrt(d * scale_factor).
    int scale_factor = 1 + (cfg.pos_att_p2c ? 1 : 0) + (cfg.pos_att_c2p ? 1 : 0);
    float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_size * scale_factor));

    // Content-content
    array scores = mc::matmul(Q, mc::transpose(K, {0, 1, 3, 2}));
    scores = mc::multiply(scores, array(inv_scale));

    // Disentangled position scores
    if (cfg.relative_attention) {
        int B  = static_cast<int>(scores.shape()[0]);
        int Hh = static_cast<int>(scores.shape()[1]);
        int Tq = static_cast<int>(scores.shape()[2]);
        int Tk = static_cast<int>(scores.shape()[3]);
        int att_span = cfg.position_buckets;

        // Project rel_emb to multi-head Q/K positions. Add a batch dim so
        // linear_apply / split_heads work the same way as for content.
        array re = mc::expand_dims(rel_emb, 0);    // [1, 2*pb, H]

        // c2p:  c2p_idx = clamp(rel + att_span, 0, 2*att_span - 1)
        if (cfg.pos_att_c2p) {
            std::string pk_prefix = cfg.share_att_key
                ? (prefix + ".attention.self.key_proj")
                : (prefix + ".attention.self.pos_key_proj");
            array pos_k = linear_apply(p, pk_prefix, re);          // [1, 2*pb, H]
            array pos_K = split_heads(pos_k, n_heads);              // [1, n_h, 2*pb, head]
            array c2p   = mc::matmul(Q, mc::transpose(pos_K, {0, 1, 3, 2}));
            c2p         = mc::multiply(c2p, array(inv_scale));      // [B, n_h, Tq, 2*pb]
            // [Tq, Tk] -> [B, n_h, Tq, Tk]
            array c2p_idx = mc::clip(mc::add(bucketed, array(att_span)),
                                     array(0), array(2 * att_span - 1));
            array idx_b = mc::broadcast_to(
                mc::reshape(c2p_idx, {1, 1, Tq, Tk}), {B, Hh, Tq, Tk});
            array c2p_g = mc::take_along_axis(c2p, idx_b, /*axis*/ -1);
            scores = mc::add(scores, c2p_g);
        }

        // p2c:  p2c_idx = clamp(-rel + att_span, 0, 2*att_span - 1)
        // p2c is [B, n_h, Tk, 2*pb]; HF gathers along its last axis with the
        // same [Tq, Tk] index matrix and then transposes the last two axes.
        if (cfg.pos_att_p2c) {
            std::string pq_prefix = cfg.share_att_key
                ? (prefix + ".attention.self.query_proj")
                : (prefix + ".attention.self.pos_query_proj");
            array pos_q = linear_apply(p, pq_prefix, re);          // [1, 2*pb, H]
            array pos_Q = split_heads(pos_q, n_heads);              // [1, n_h, 2*pb, head]
            array p2c   = mc::matmul(K, mc::transpose(pos_Q, {0, 1, 3, 2}));
            p2c         = mc::multiply(p2c, array(inv_scale));      // [B, n_h, Tk, 2*pb]
            array p2c_idx = mc::clip(
                mc::add(mc::negative(bucketed), array(att_span)),
                array(0), array(2 * att_span - 1));                  // [Tq, Tk]
            // Square attention only: HF expands [Tq, Tk] to [B, n_h, Tk, Tk].
            // For Tq = Tk this just broadcasts.
            array idx_b = mc::broadcast_to(
                mc::reshape(p2c_idx, {1, 1, Tq, Tk}), {B, Hh, Tk, Tk});
            array p2c_g = mc::take_along_axis(p2c, idx_b, /*axis*/ -1);
            // Transpose last two axes -> [B, n_h, Tq, Tk]
            array p2c_t = mc::transpose(p2c_g, {0, 1, 3, 2});
            scores = mc::add(scores, p2c_t);
        }
    }

    // Mask + softmax
    scores = mc::add(scores, add_mask);
    array probs = mc::softmax(scores, /*axes*/ {-1});

    array context = mc::matmul(probs, V);             // [B, n_h, T, head]
    array merged  = merge_heads(context);             // [B, T, H]

    array out = linear_apply(p, prefix + ".attention.output.dense", merged);
    array residual = mc::add(out, hidden);
    return layer_norm_last_axis(p, prefix + ".attention.output.LayerNorm",
                                residual, cfg.layer_norm_eps);
}

array deberta_layer(const ParamStore& p,
                    const std::string& prefix,
                    const array& hidden,
                    const array& rel_emb,
                    const array& bucketed,
                    const array& add_mask,
                    const BertConfig& cfg) {
    array attn_out = deberta_self_attention(p, prefix,
                                            hidden, rel_emb, bucketed,
                                            add_mask, cfg);
    return bert_ffn_block(p, prefix, attn_out, cfg);
}

// ============================================================================
// Forward driver
// ============================================================================

// Number of layers we actually need to run to materialise hidden_states[-3].
//
// `feature_layer = -3` -> we need hidden_states[num_layers - 3 + 1] in 1-indexed
// hidden_states (since hidden_states[0] is the embedding output). That equals
// the OUTPUT of layer (num_layers - 3), so we run (num_layers - 2) layers.
//
// To be safe across `feature_layer` values, we evaluate:
//   needed_layers = num_hidden_layers + feature_layer + 1
// (matches HF: hidden_states[-3] indexes from the end, and the output of
//  layer i is at hidden_states[i+1]).
int needed_forward_layers(const BertConfig& cfg) {
    int N = cfg.num_hidden_layers;
    int idx = cfg.feature_layer;
    if (idx < 0) idx = N + 1 + idx;             // -3 -> N - 2 (1-based hidden_states)
    if (idx < 0)        idx = 0;
    if (idx > N)        idx = N;
    return idx;                                  // number of transformer blocks to run
}

array bert_forward(const ParamStore& p,
                   const BertConfig& cfg,
                   const array& input_ids,         // [B, T]
                   const array& token_type_ids,    // [B, T]
                   const array& attention_mask) {  // [B, T]
    array add_mask = attention_additive_mask(attention_mask);   // [B, 1, 1, T]

    array hidden = (cfg.kind == BertConfig::kBert)
        ? bert_embeddings   (p, input_ids, token_type_ids, cfg)
        : deberta_embeddings(p, input_ids, cfg);

    // Pre-compute relative position helpers for DeBERTa. `mlx::core::array`
    // has no default constructor, so we hold these in std::optional and only
    // build them on the DeBERTa path.
    std::optional<array> rel_emb;
    std::optional<array> bucketed;
    if (cfg.kind == BertConfig::kDebertaV2 && cfg.relative_attention) {
        array re = p.at("encoder.rel_embeddings.weight");
        if (cfg.norm_rel_ebd_layer_norm &&
            p.has("encoder.LayerNorm.weight")) {
            re = layer_norm_last_axis(p, "encoder.LayerNorm",
                                      re, cfg.layer_norm_eps);
        }
        rel_emb = re;
        int T = static_cast<int>(input_ids.shape().back());
        bucketed = deberta_bucketed_rel(T, T, cfg);
    }

    int n_run = needed_forward_layers(cfg);
    for (int i = 0; i < n_run; ++i) {
        std::string layer_prefix = "encoder.layer." + std::to_string(i);
        if (cfg.kind == BertConfig::kBert) {
            hidden = bert_layer(p, layer_prefix, hidden, add_mask, cfg);
        } else {
            hidden = deberta_layer(p, layer_prefix,
                                   hidden, *rel_emb, *bucketed, add_mask, cfg);
        }
    }

    return hidden;   // [B, T, H]
}

// ============================================================================
// Config parsing from safetensors metadata
// ============================================================================

int parse_int_meta(const std::unordered_map<std::string, std::string>& meta,
                   const std::string& key, int dflt) {
    auto it = meta.find(key);
    if (it == meta.end()) return dflt;
    try { return std::stoi(it->second); } catch (...) { return dflt; }
}
float parse_float_meta(const std::unordered_map<std::string, std::string>& meta,
                       const std::string& key, float dflt) {
    auto it = meta.find(key);
    if (it == meta.end()) return dflt;
    try { return std::stof(it->second); } catch (...) { return dflt; }
}
bool parse_bool_meta(const std::unordered_map<std::string, std::string>& meta,
                     const std::string& key, bool dflt) {
    auto it = meta.find(key);
    if (it == meta.end()) return dflt;
    const std::string& s = it->second;
    if (s == "1" || s == "true" || s == "True") return true;
    if (s == "0" || s == "false" || s == "False") return false;
    return dflt;
}

BertConfig parse_config_from_metadata(
    const std::unordered_map<std::string, std::string>& meta) {
    BertConfig cfg;
    auto it = meta.find("kind");
    if (it != meta.end() && it->second == "deberta_v2") cfg.kind = BertConfig::kDebertaV2;
    else                                                cfg.kind = BertConfig::kBert;

    cfg.hidden_size             = parse_int_meta  (meta, "hidden_size",             cfg.hidden_size);
    cfg.num_hidden_layers       = parse_int_meta  (meta, "num_hidden_layers",       cfg.num_hidden_layers);
    cfg.num_attention_heads     = parse_int_meta  (meta, "num_attention_heads",     cfg.num_attention_heads);
    cfg.intermediate_size       = parse_int_meta  (meta, "intermediate_size",       cfg.intermediate_size);
    cfg.max_position_embeddings = parse_int_meta  (meta, "max_position_embeddings", cfg.max_position_embeddings);
    cfg.type_vocab_size         = parse_int_meta  (meta, "type_vocab_size",         cfg.type_vocab_size);
    cfg.vocab_size              = parse_int_meta  (meta, "vocab_size",              cfg.vocab_size);
    cfg.layer_norm_eps          = parse_float_meta(meta, "layer_norm_eps",          cfg.layer_norm_eps);
    cfg.feature_layer           = parse_int_meta  (meta, "feature_layer",           cfg.feature_layer);

    cfg.relative_attention      = parse_bool_meta (meta, "relative_attention",      cfg.relative_attention);
    cfg.position_buckets        = parse_int_meta  (meta, "position_buckets",        cfg.position_buckets);
    cfg.max_relative_positions  = parse_int_meta  (meta, "max_relative_positions",  cfg.max_relative_positions);
    cfg.share_att_key           = parse_bool_meta (meta, "share_att_key",           cfg.share_att_key);
    cfg.position_biased_input   = parse_bool_meta (meta, "position_biased_input",   cfg.position_biased_input);
    cfg.norm_rel_ebd_layer_norm = parse_bool_meta (meta, "norm_rel_ebd_layer_norm", cfg.norm_rel_ebd_layer_norm);
    cfg.pos_att_p2c             = parse_bool_meta (meta, "pos_att_p2c",             cfg.pos_att_p2c);
    cfg.pos_att_c2p             = parse_bool_meta (meta, "pos_att_c2p",             cfg.pos_att_c2p);

    return cfg;
}

} // namespace

// ============================================================================
// BertRuntime
// ============================================================================

struct BertRuntime::Impl {
    bool loaded = false;
    BertConfig cfg;
    ParamStore params;
};

BertRuntime::BertRuntime() : impl_(std::make_unique<Impl>()) {}
BertRuntime::~BertRuntime() = default;
BertRuntime::BertRuntime(BertRuntime&&) noexcept = default;
BertRuntime& BertRuntime::operator=(BertRuntime&&) noexcept = default;

bool BertRuntime::is_loaded() const noexcept { return impl_ && impl_->loaded; }

int BertRuntime::hidden_size() const noexcept {
    return impl_ ? impl_->cfg.hidden_size : 0;
}

bool BertRuntime::load(const std::string& path, std::string* err) {
    auto set_err = [&](const std::string& s) { if (err) *err = s; };
    try {
        impl_->params = load_safetensors(path);
        impl_->cfg    = parse_config_from_metadata(impl_->params.metadata);
        if (impl_->cfg.hidden_size <= 0 || impl_->cfg.num_hidden_layers <= 0) {
            set_err("BERT config metadata missing or malformed in " + path);
            return false;
        }
        impl_->loaded = true;
        return true;
    } catch (const std::exception& e) {
        set_err(std::string("BertRuntime::load failed: ") + e.what());
        return false;
    }
}

BertInferOutput BertRuntime::infer(const BertInferInputs& in, std::string* err) {
    BertInferOutput out;
    if (!is_loaded()) {
        if (err) *err = "BertRuntime not loaded";
        return out;
    }
    try {
        const auto& cfg = impl_->cfg;
        int T = static_cast<int>(in.input_ids.size());
        if (T <= 0) {
            if (err) *err = "empty input_ids";
            return out;
        }
        if (in.word2ph.size() != static_cast<size_t>(T)) {
            if (err) *err = "word2ph length must match input_ids length";
            return out;
        }

        // Build inputs.
        std::vector<int32_t> ids(T);
        for (int i = 0; i < T; ++i) ids[i] = static_cast<int32_t>(in.input_ids[i]);
        array input_ids = array(ids.data(), {1, T}, mc::int32);

        std::vector<int32_t> tt(T, 0);
        if (!in.token_type_ids.empty()) {
            int n = std::min<int>(T, static_cast<int>(in.token_type_ids.size()));
            for (int i = 0; i < n; ++i) tt[i] = static_cast<int32_t>(in.token_type_ids[i]);
        }
        array tt_ids = array(tt.data(), {1, T}, mc::int32);

        std::vector<int32_t> mask(T, 1);
        array attn_mask = array(mask.data(), {1, T}, mc::int32);

        array hidden = bert_forward(impl_->params, cfg, input_ids, tt_ids, attn_mask);

        const int H = cfg.hidden_size;
        const int phone_count = static_cast<int>(in.phone_count);

        // Build gather indices on CPU (tiny work: one int per phone).
        // For token t with word2ph[t] repeats, push t that many times.
        std::vector<int32_t> indices;
        indices.reserve(static_cast<size_t>(phone_count));
        for (int t = 0; t < T; ++t) {
            int rep = static_cast<int>(in.word2ph[t]);
            for (int r = 0; r < rep && static_cast<int>(indices.size()) < phone_count; ++r)
                indices.push_back(t);
        }
        if (static_cast<int>(indices.size()) != phone_count) {
            if (err) *err = "word2ph expansion did not fill all phones";
            return out;
        }

        // word2ph expansion on GPU: hidden [1, T, H] -> squeeze -> [T, H]
        // -> take(indices) -> [phone_count, H].  No intermediate eval needed.
        array h2d = mc::squeeze(hidden, 0);                               // [T, H]
        array idx_arr = array(indices.data(), {phone_count}, mc::int32);
        array expanded = mc::take(h2d, idx_arr, /*axis=*/0);              // [phone_count, H]

        // Cast to fp32 (weights may be fp16/bf16); materialise once.
        if (expanded.dtype() != mc::float32)
            expanded = mc::astype(expanded, mc::float32);
        mc::eval(expanded);

        auto sh = expanded.shape();
        if (sh.size() != 2 || sh[0] != phone_count || sh[1] != H) {
            if (err) *err = "BERT word2ph expansion returned unexpected shape";
            return out;
        }
        std::vector<float> phone(static_cast<size_t>(phone_count) * H);
        std::memcpy(phone.data(), expanded.data<float>(),
                    phone.size() * sizeof(float));

        out.bert_features = std::move(phone);
        out.hidden_size   = H;
        out.phone_count   = phone_count;
        return out;
    } catch (const std::exception& e) {
        if (err) *err = std::string("BertRuntime::infer failed: ") + e.what();
        return out;
    }
}

} // namespace mlx_rt
} // namespace bv2

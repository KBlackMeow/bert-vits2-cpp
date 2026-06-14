// Native Apple-MLX BERT runtime, replacing the ONNX Runtime + CoreML EP
// path used by the existing C++ frontend.
//
// Three checkpoints are supported:
//
//   * ZH `chinese-roberta-wwm-ext-large` -> kBert (vanilla BERT-large with
//     absolute position + token-type embeddings).
//
//   * JP `deberta-v2-large-japanese-char-wwm`,
//     EN `deberta-v3-large` -> kDebertaV2 (disentangled relative attention,
//     position bucketing, optional `share_att_key`).
//
// All three are converted upstream to the same flat safetensors layout
// (see `src/convert_bert_to_mlx.py`), with the model `kind` and shape
// hyper-params stashed in safetensors metadata so the C++ side does not
// need a separate config.json.
//
// The runtime returns the same `hidden_states[-3]` slice that the
// Bert-VITS2 v2.3 ONNX exports produce, then expands tokens into phones
// via `word2ph`.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bv2 {
namespace mlx_rt {

struct BertConfig {
    enum Kind { kBert = 0, kDebertaV2 = 1 };

    Kind kind                   = kBert;
    int hidden_size             = 1024;
    int num_hidden_layers       = 24;
    int num_attention_heads     = 16;
    int intermediate_size       = 4096;
    int max_position_embeddings = 512;
    int type_vocab_size         = 2;
    int vocab_size              = 21128;
    float layer_norm_eps        = 1e-12f;

    // Negative index into hidden_states (matches HF semantics):
    //   feature_layer = -3 means hidden_states[-3] = output of layer
    //   (num_hidden_layers - 3). With layer_norm-after attention we still
    //   only need to run the first (num_hidden_layers - 3 + 1) layers.
    int feature_layer = -3;

    // DeBERTa-v2 / v3 only.
    bool relative_attention      = false;
    int  position_buckets        = 256;
    int  max_relative_positions  = 0;        // 0 -> reuse position_buckets
    bool share_att_key           = false;
    bool position_biased_input   = false;
    bool norm_rel_ebd_layer_norm = false;    // true if config.norm_rel_ebd contains "layer_norm"
    bool pos_att_p2c             = true;
    bool pos_att_c2p             = true;
    // GELU variant used inside the FFN. HF stores this as "hidden_act" with
    // values "gelu", "gelu_new" (≈ tanh approximation), "gelu_fast" etc.
    // For BERT-large and DeBERTa-v2/v3 the right choice is exact GELU.
    bool gelu_tanh_approx        = false;
};

struct BertInferInputs {
    std::vector<int64_t> input_ids;          // [T]
    std::vector<int64_t> token_type_ids;     // [T] (zeros / empty = no token-type embedding)
    std::vector<int64_t> word2ph;            // [T] (each entry: # of phones at this token)
    int64_t phone_count = 0;                 // sum(word2ph) as a safety check
};

struct BertInferOutput {
    // Row-major [1024, phone_count] (matching `Tensor zeros_bert(...)`).
    std::vector<float> bert_features;
    int hidden_size  = 0;
    int phone_count  = 0;
};

class BertRuntime {
public:
    BertRuntime();
    ~BertRuntime();

    BertRuntime(const BertRuntime&)            = delete;
    BertRuntime& operator=(const BertRuntime&) = delete;
    BertRuntime(BertRuntime&&) noexcept;
    BertRuntime& operator=(BertRuntime&&) noexcept;

    // `weights_path` is the .safetensors file produced by
    // `src/convert_bert_to_mlx.py`. Returns false on failure.
    bool load(const std::string& weights_path, std::string* error_out = nullptr);

    bool is_loaded() const noexcept;

    int hidden_size() const noexcept;

    // Run BERT, slice the third-from-last hidden state, and expand tokens
    // into phones using `word2ph`. Output layout is row-major
    // [hidden_size, phone_count] - exactly what `Tensor zeros_bert` uses.
    BertInferOutput infer(const BertInferInputs& in,
                          std::string* error_out = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mlx_rt
} // namespace bv2

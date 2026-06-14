// Bert-VITS2 v2.3 attention block (relative-position MHA + Conv-FFN +
// channel-axis LayerNorm), implemented on top of MLX C++.

#pragma once

#include <mlx/array.h>

#include <optional>
#include <string>

#include "bv2_mlx_modules.h"

namespace bv2 {
namespace mlx_rt {

using ::mlx::core::array;

struct EncoderCfg {
    int hidden_channels;
    int filter_channels;
    int n_heads;
    int n_layers;
    int kernel_size;
    int window_size = 4;
    int gin_channels = 0;
    int cond_layer_idx = 2;  // vits2: speaker condition fed at the 3rd block
};

// `attentions.MultiHeadAttention.forward`. Self-attention only (the only
// usage in v2.3).
//   x:        [B, C, T]
//   attn_mask:[B, 1, T, T] (float 0/1) or empty
//   prefix:   path to the MultiHeadAttention's params
//   k_chan:   hidden_channels / n_heads
array mha_apply(const ParamStore& p,
                const std::string& prefix,
                const array& x,
                const array& attn_mask,
                int n_heads,
                int window_size);

// `attentions.FFN.forward` with non-causal "same" padding (the v2.3 path).
//   x:      [B, C, T], x_mask: [B, 1, T] (float 0/1)
array ffn_apply(const ParamStore& p,
                const std::string& prefix,
                const array& x,
                const array& x_mask,
                int kernel_size);

// `attentions.Encoder.forward` (n_layers Transformer blocks with
// optional speaker conditioning at `cond_layer_idx`).
//   x:      [B, C, T] (already C = hidden_channels)
//   x_mask: [B, 1, T] (float 0/1)
//   g (optional): [B, gin_channels, 1] speaker embedding broadcast over T.
array encoder_apply(const ParamStore& p,
                    const std::string& prefix,
                    const array& x,
                    const array& x_mask,
                    const array* g,
                    const EncoderCfg& cfg);

} // namespace mlx_rt
} // namespace bv2

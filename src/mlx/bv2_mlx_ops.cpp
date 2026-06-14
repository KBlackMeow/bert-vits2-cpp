#include "bv2_mlx_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/random.h>

namespace mc = ::mlx::core;

namespace bv2 {
namespace mlx_rt {

// ---------------------------------------------------------------------------
// Random
// ---------------------------------------------------------------------------

array make_key(uint64_t seed) {
    return mc::random::key(seed);
}

array split_keys(array& key, int n) {
    auto pair = mc::random::split(key, n);
    // mc::random::split returns (subkey, key) pair when n=2; for n>2 use
    // the n-key form which is exposed as `random::split(key, num)`.
    // Both APIs exist in MLX; the multi-key one is what we want.
    (void)pair;
    return mc::random::split(key, n);
}

array randn(array& key, const std::vector<int>& shape) {
    // Forks `key` into (subkey, key); we update the live key in place.
    auto pair = mc::random::split(key);
    array subkey = pair.first;
    key = pair.second;
    return mc::random::normal(shape, mc::float32, /*loc*/ 0.0f, /*scale*/ 1.0f, subkey);
}

// ---------------------------------------------------------------------------
// Mask helpers
// ---------------------------------------------------------------------------

array sequence_mask_f32(const array& lengths, int max_length) {
    // x: arange(max_length) -> [max_length]
    array x = mc::arange(0, max_length, 1, mc::int32);
    // mask = x[None, :] < lengths[:, None] -> [B, max_length]
    array x2 = mc::expand_dims(x, 0);
    array lengths2 = mc::expand_dims(mc::astype(lengths, mc::int32), 1);
    array mask = mc::less(x2, lengths2);
    return mc::astype(mask, mc::float32);
}

array generate_path(const array& duration, const array& mask) {
    // duration: [B, 1, T_x],   mask: [B, 1, T_y, T_x]
    // returns:  [B, 1, T_y, T_x]
    auto shape = mask.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("generate_path: mask must be 4D");
    }
    int B   = shape[0];
    int T_y = shape[2];
    int T_x = shape[3];

    array cum = mc::cumsum(duration, /*axis*/ -1);    // [B, 1, T_x]
    array cum_flat = mc::reshape(cum, {B * T_x});     // [B*T_x]
    array path = sequence_mask_f32(cum_flat, T_y);    // [B*T_x, T_y]
    path = mc::reshape(path, {B, T_x, T_y});          // [B, T_x, T_y]

    // path = path - F.pad(path, [[0,0],[1,0],[0,0]])[:, :-1]
    // i.e. subtract the previous-row sequence_mask, leaving only the new
    // step at each row. Achieved by zero-padding row dim from top and
    // dropping the last row.
    array shifted = mc::pad(path, /*axis*/ {1},
                            /*low_pad*/ {1}, /*high_pad*/ {0},
                            /*pad_value*/ array(0.0f));                          // [B, T_x+1, T_y]
    shifted = mc::slice(shifted, /*starts*/ {0, 0, 0},
                        /*stops*/ {B, T_x, T_y});                                // [B, T_x, T_y]
    path = mc::subtract(path, shifted);
    // -> path.unsqueeze(1).transpose(2, 3) * mask  (final [B, 1, T_y, T_x])
    path = mc::expand_dims(path, 1);                                             // [B, 1, T_x, T_y]
    path = mc::transpose(path, {0, 1, 3, 2});                                    // [B, 1, T_y, T_x]
    return mc::multiply(path, mask);
}

// ---------------------------------------------------------------------------
// Activation / fused ops
// ---------------------------------------------------------------------------

array fused_add_tanh_sigmoid_multiply(const array& x_in,
                                      const array& g_l,
                                      int n_channels) {
    array s = mc::add(x_in, g_l);
    auto sh = s.shape();
    if (sh.size() != 3) {
        throw std::runtime_error("fused_act: expected [B, 2C, T]");
    }
    int B = sh[0], C2 = sh[1], T = sh[2];
    if (C2 != 2 * n_channels) {
        throw std::runtime_error("fused_act: channel size mismatch");
    }
    array t_part = mc::slice(s, {0, 0, 0}, {B, n_channels, T});
    array s_part = mc::slice(s, {0, n_channels, 0}, {B, 2 * n_channels, T});
    return mc::multiply(mc::tanh(t_part), mc::sigmoid(s_part));
}

array leaky_relu_01(const array& x) {
    // mc::leaky_relu(x, slope) — in MLX 0.21 it lives in mlx::nn but we
    // implement it directly to avoid the Python-only layer.
    array slope_x = mc::multiply(x, array(0.1f));
    return mc::maximum(x, slope_x);
}

array gelu(const array& x) {
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    static const float inv_sqrt2 = 0.7071067811865475f;
    array t = mc::erf(mc::multiply(x, array(inv_sqrt2)));
    return mc::multiply(mc::multiply(x, array(0.5f)),
                        mc::add(array(1.0f), t));
}

// ---------------------------------------------------------------------------
// LayerNorm (channel-axis)
// ---------------------------------------------------------------------------

array layer_norm_channels(const array& x,
                          const array& gamma,
                          const array& beta,
                          float eps) {
    // x is [B, C, T]; we want layer_norm over C. Move C to last, normalize,
    // move back.
    array y = mc::transpose(x, {0, 2, 1});                  // [B, T, C]
    y = mc::fast::layer_norm(y, gamma, beta, eps);
    y = mc::transpose(y, {0, 2, 1});                        // [B, C, T]
    return y;
}

// ---------------------------------------------------------------------------
// Padding & convolutions
// ---------------------------------------------------------------------------

array pad_same_1d(const array& x, int kernel_size) {
    if (kernel_size <= 1) return x;
    int pad_l = (kernel_size - 1) / 2;
    int pad_r = kernel_size / 2;
    // Pad along the last axis (T).
    return mc::pad(x, /*axes*/ {-1},
                   /*low_pad*/ {pad_l}, /*high_pad*/ {pad_r},
                   /*value*/ array(0.0f));
}

array conv1d_pt(const array& x,
                const array& weight,
                const array* bias_or_null,
                int stride,
                int padding,
                int dilation,
                int groups) {
    // x: [B, C_in, T]
    // weight (PyTorch): [C_out, C_in/g, K]  (saved as-is from torch state_dict)
    // MLX conv1d:  input [B, T, C_in], weight [C_out, K, C_in/g]
    array x_blc = mc::transpose(x, {0, 2, 1});                 // [B, T, C_in]
    array w_pt  = mc::transpose(weight, {0, 2, 1});            // [C_out, K, C_in/g]
    array y = mc::conv1d(x_blc, w_pt,
                         /*stride*/ stride,
                         /*padding*/ padding,
                         /*dilation*/ dilation,
                         /*groups*/ groups);
    if (bias_or_null != nullptr) {
        // bias is [C_out]; broadcast over [B, T, C_out].
        y = mc::add(y, *bias_or_null);
    }
    return mc::transpose(y, {0, 2, 1});                        // [B, C_out, T_out]
}

array conv_transpose1d_pt(const array& x,
                          const array& weight,
                          const array* bias_or_null,
                          int stride,
                          int padding,
                          int dilation,
                          int groups) {
    // PyTorch ConvTranspose1d weight: [C_in, C_out/g, K].
    // MLX conv_transpose1d expects weight [C_out, K, C_in/g].
    // -> permute(1, 2, 0).
    array x_blc = mc::transpose(x, {0, 2, 1});                 // [B, T, C_in]
    array w_mlx = mc::transpose(weight, {1, 2, 0});            // [C_out, K, C_in/g]
    array y = mc::conv_transpose1d(x_blc, w_mlx,
                                   stride, padding, dilation, groups);
    if (bias_or_null != nullptr) {
        y = mc::add(y, *bias_or_null);
    }
    return mc::transpose(y, {0, 2, 1});                        // [B, C_out, T_out]
}

} // namespace mlx_rt
} // namespace bv2

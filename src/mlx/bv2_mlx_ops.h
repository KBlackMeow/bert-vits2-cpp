// Common helper ops for the MLX runtime. Everything operates on
// `mlx::core::array` tensors. Layout convention used throughout the
// runtime is PyTorch-style [B, C, T]; conv wrappers transpose internally.

#pragma once

#include <cstdint>
#include <vector>

#include <mlx/array.h>
#include <mlx/random.h>

namespace bv2 {
namespace mlx_rt {

using ::mlx::core::array;

struct ParamStore;  // fwd

// ---------------------------------------------------------------------------
// Random
// ---------------------------------------------------------------------------

// Initialise a per-runtime PRNG key from a 64-bit seed, like
// `random.key(seed)` in MLX-Python.
array make_key(uint64_t seed);

// Split `key` into `n` keys; returns shape [n, 2].
array split_keys(array& key, int n);

// Standard-normal sample with the given shape, drawing from `key` and
// advancing it. Equivalent to `torch.randn(shape)`.
array randn(array& key, const std::vector<int>& shape);

// ---------------------------------------------------------------------------
// Mask helpers
// ---------------------------------------------------------------------------

// `commons.sequence_mask`: build a [B, max_len] bool/float mask from
// `lengths` with shape [B]. Returns float (0/1) of dtype float32.
array sequence_mask_f32(const array& lengths, int max_length);

// `commons.generate_path`: durations [B, 1, T_x] -> attn matrix
// [B, 1, T_y, T_x] respecting `mask` [B, 1, T_y, T_x].
array generate_path(const array& duration, const array& mask);

// ---------------------------------------------------------------------------
// Activation / fused ops
// ---------------------------------------------------------------------------

// `fused_add_tanh_sigmoid_multiply`: x_in [B, 2C, T], g_l [B, 2C, T] ->
//   tanh(x_in[:, :C] + g_l[:, :C]) * sigmoid(x_in[:, C:] + g_l[:, C:])
array fused_add_tanh_sigmoid_multiply(const array& x_in,
                                      const array& g_l,
                                      int n_channels);

// LeakyReLU with negative_slope=0.1 (HiFi-GAN's `LRELU_SLOPE`).
array leaky_relu_01(const array& x);

// Exact GELU (matches PyTorch `F.gelu(x)` with the default
// `approximate="none"`). MLX's C++ core doesn't expose `gelu` as a free
// function, so we provide our own.
//
//   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
array gelu(const array& x);

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

// Channel-axis LayerNorm (Bert-VITS2 LayerNorm).
//   x:    [B, C, T]
//   gamma:[C]
//   beta: [C]
// Returns [B, C, T] = layer_norm-along-C(x).
array layer_norm_channels(const array& x,
                          const array& gamma,
                          const array& beta,
                          float eps = 1e-5f);

// "Same" 1d padding such that conv1d(input, kernel) preserves length.
//   pad_l = (kernel_size - 1) // 2
//   pad_r = kernel_size // 2
array pad_same_1d(const array& x, int kernel_size);

// Generic Conv1d that takes a PyTorch-layout weight and bias
// stored as [C_out, C_in/groups, K] and returns [B, C_out, T_out].
//   x: [B, C_in, T]
array conv1d_pt(const array& x,
                const array& weight,           // [C_out, C_in/g, K]
                const array* bias_or_null,     // [C_out] or nullptr
                int stride   = 1,
                int padding  = 0,
                int dilation = 1,
                int groups   = 1);

// ConvTranspose1d with PyTorch-layout weight stored as [C_in, C_out/groups, K].
//   x: [B, C_in, T]
array conv_transpose1d_pt(const array& x,
                          const array& weight,         // [C_in, C_out/g, K]
                          const array* bias_or_null,
                          int stride   = 1,
                          int padding  = 0,
                          int dilation = 1,
                          int groups   = 1);

} // namespace mlx_rt
} // namespace bv2

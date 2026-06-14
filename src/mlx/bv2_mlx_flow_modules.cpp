#include "bv2_mlx_flow_modules.h"
#include "bv2_mlx_ops.h"
#include "bv2_mlx_transforms.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/ops.h>

namespace mc = ::mlx::core;

namespace bv2 {
namespace mlx_rt {

// ---------------------------------------------------------------------------
// DDSConv
// ---------------------------------------------------------------------------

array dds_conv_apply(const ParamStore& p,
                     const std::string& prefix,
                     const array& x_in,
                     const array& x_mask,
                     int n_layers,
                     int kernel_size,
                     const array* g) {
    array x = x_in;
    if (g != nullptr) {
        x = mc::add(x, *g);
    }
    for (int i = 0; i < n_layers; ++i) {
        int dilation = 1;
        for (int d = 0; d < i; ++d) dilation *= kernel_size;
        std::string sep = prefix + ".convs_sep." + std::to_string(i);
        std::string oxo = prefix + ".convs_1x1." + std::to_string(i);
        std::string n1  = prefix + ".norms_1." + std::to_string(i);
        std::string n2  = prefix + ".norms_2." + std::to_string(i);
        // depth-separable conv: groups=channels (read at runtime from x.shape[1]).
        int channels = static_cast<int>(x.shape()[1]);
        Conv1dCfg cfg;
        cfg.groups   = channels;
        cfg.dilation = dilation;
        cfg.padding  = (kernel_size * dilation - dilation) / 2;
        array y = conv1d_apply(p, sep, mc::multiply(x, x_mask), cfg);
        y = layer_norm_channel_apply(p, n1, y);
        y = gelu(y);                    // exact GELU (matches torch F.gelu)
        y = conv1d_apply(p, oxo, y);    // 1x1 conv
        y = layer_norm_channel_apply(p, n2, y);
        y = gelu(y);
        x = mc::add(x, y);
    }
    return mc::multiply(x, x_mask);
}

// ---------------------------------------------------------------------------
// Flip
// ---------------------------------------------------------------------------

array flip_channels(const array& x) {
    // `torch.flip(x, [1])`. Implemented via `take` with reversed indices
    // because not all MLX C++ versions accept negative-stride slices.
    int C = static_cast<int>(x.shape()[1]);
    std::vector<int32_t> idx(C);
    for (int i = 0; i < C; ++i) idx[i] = C - 1 - i;
    array idx_arr = array(idx.data(), {C}, mc::int32);
    return mc::take(x, idx_arr, /*axis*/ 1);
}

// ---------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------

array log_flow_inverse(const array& x, const array& x_mask) {
    return mc::multiply(mc::exp(x), x_mask);
}

// ---------------------------------------------------------------------------
// ElementwiseAffine
// ---------------------------------------------------------------------------

array elementwise_affine_inverse(const ParamStore& p,
                                 const std::string& prefix,
                                 const array& x,
                                 const array& x_mask) {
    const array& m    = p.at(prefix + ".m");        // [C, 1]
    const array& logs = p.at(prefix + ".logs");     // [C, 1]
    array m_b    = mc::expand_dims(m,    0);        // [1, C, 1]
    array logs_b = mc::expand_dims(logs, 0);
    array y = mc::subtract(x, m_b);
    y = mc::multiply(y, mc::exp(mc::negative(logs_b)));
    return mc::multiply(y, x_mask);
}

// ---------------------------------------------------------------------------
// ConvFlow (reverse)
// ---------------------------------------------------------------------------

array conv_flow_inverse(const ParamStore& p,
                        const std::string& prefix,
                        const array& x,
                        const array& x_mask,
                        const array* g_or_null,
                        int filter_channels,
                        int kernel_size,
                        int n_layers,
                        int num_bins,
                        float tail_bound) {
    auto sh = x.shape();
    int B = sh[0], C = sh[1], T = sh[2];
    if (C % 2 != 0) {
        throw std::runtime_error("ConvFlow: channels must be divisible by 2");
    }
    int half = C / 2;
    array x0 = mc::slice(x, {0, 0,    0}, {B, half, T});
    array x1 = mc::slice(x, {0, half, 0}, {B, C,    T});

    array h = conv1d_apply(p, prefix + ".pre", x0);                 // [B, F, T]
    h = dds_conv_apply(p, prefix + ".convs", h, x_mask,
                       n_layers, kernel_size, g_or_null);
    h = conv1d_apply(p, prefix + ".proj", h);                       // [B, half*(3*N-1), T]
    h = mc::multiply(h, x_mask);

    // h.reshape(B, half, -1, T).permute(0, 1, 3, 2) -> [B, half, T, 3*N - 1]
    int Q = (3 * num_bins - 1);
    h = mc::reshape(h, {B, half, Q, T});
    h = mc::transpose(h, {0, 1, 3, 2});                              // [B, half, T, Q]

    // splits: widths [..., :N], heights [..., N:2N], derivs [..., 2N:]
    array unnorm_w = mc::slice(h,
                               /*starts*/ {0, 0, 0, 0},
                               /*stops*/  {B, half, T, num_bins});
    array unnorm_h = mc::slice(h,
                               /*starts*/ {0, 0, 0, num_bins},
                               /*stops*/  {B, half, T, 2 * num_bins});
    array unnorm_d = mc::slice(h,
                               /*starts*/ {0, 0, 0, 2 * num_bins},
                               /*stops*/  {B, half, T, Q});

    float scale = 1.0f / std::sqrt(static_cast<float>(filter_channels));
    unnorm_w = mc::multiply(unnorm_w, array(scale));
    unnorm_h = mc::multiply(unnorm_h, array(scale));
    // derivatives are kept un-scaled (matching transforms.py).

    array x1_new = unconstrained_rqs_inverse(x1,
                                             unnorm_w, unnorm_h, unnorm_d,
                                             tail_bound);
    array out = mc::concatenate({x0, x1_new}, /*axis*/ 1);
    return mc::multiply(out, x_mask);
}

// ---------------------------------------------------------------------------
// TransformerCouplingLayer (reverse)
// ---------------------------------------------------------------------------

array transformer_coupling_layer_inverse(const ParamStore& p,
                                         const std::string& prefix,
                                         const array& x,
                                         const array& x_mask,
                                         const array* g_or_null,
                                         const EncoderCfg& cfg,
                                         int half_channels) {
    auto sh = x.shape();
    int B = sh[0], C = sh[1], T = sh[2];
    int half = half_channels;
    if (C != 2 * half) {
        throw std::runtime_error("TransformerCouplingLayer: channel mismatch");
    }
    array x0 = mc::slice(x, {0, 0,    0}, {B, half, T});
    array x1 = mc::slice(x, {0, half, 0}, {B, C,    T});

    array h = conv1d_apply(p, prefix + ".pre", x0);                 // [B, hid, T]
    h = mc::multiply(h, x_mask);
    h = encoder_apply(p, prefix + ".enc", h, x_mask, g_or_null, cfg);
    array stats = conv1d_apply(p, prefix + ".post", h);             // [B, half, T]   (mean_only=True)
    stats = mc::multiply(stats, x_mask);

    // mean_only -> m=stats, logs=zeros => x1 = (x1 - m) * exp(0) = x1 - m
    array x1_new = mc::subtract(x1, stats);

    return mc::concatenate({x0, x1_new}, /*axis*/ 1);
}

} // namespace mlx_rt
} // namespace bv2

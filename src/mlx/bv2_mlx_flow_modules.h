// Flow building blocks used by the Stochastic Duration Predictor and the
// Transformer Coupling flow in Bert-VITS2 v2.3 (inference, reverse-only).

#pragma once

#include <mlx/array.h>

#include <string>

#include "bv2_mlx_attentions.h"
#include "bv2_mlx_modules.h"

namespace bv2 {
namespace mlx_rt {

using ::mlx::core::array;

// `modules.DDSConv.forward`. Dilated depth-separable conv stack.
//   x:      [B, C, T]
//   x_mask: [B, 1, T]
//   g (optional): conditioning, broadcast-added before the loop.
array dds_conv_apply(const ParamStore& p,
                     const std::string& prefix,
                     const array& x_in,
                     const array& x_mask,
                     int n_layers,
                     int kernel_size,
                     const array* g = nullptr);

// `modules.Flip` (channel flip, reverse direction is identity-of-flip).
array flip_channels(const array& x);

// `modules.Log.forward(reverse=True)`.
//   y = exp(x) * x_mask
array log_flow_inverse(const array& x, const array& x_mask);

// `modules.ElementwiseAffine.forward(reverse=True)`.
//   y = (x - m) * exp(-logs) * x_mask
array elementwise_affine_inverse(const ParamStore& p,
                                 const std::string& prefix,
                                 const array& x,
                                 const array& x_mask);

// `modules.ConvFlow.forward(reverse=True)`. Splits x at half-channels,
// computes spline parameters from x0 via DDSConv, applies the spline to x1.
//   in_channels: 2 in the SDP usage (so each half is 1 channel).
array conv_flow_inverse(const ParamStore& p,
                        const std::string& prefix,
                        const array& x,
                        const array& x_mask,
                        const array* g_or_null,
                        int filter_channels,
                        int kernel_size,
                        int n_layers,
                        int num_bins,
                        float tail_bound);

// `modules.TransformerCouplingLayer.forward(reverse=True)`.
// Uses the v2.3 `Encoder` as the conditioning network. Splits at half-channels,
// reads pre/enc/post params under `<prefix>.pre/.enc/.post`.
array transformer_coupling_layer_inverse(const ParamStore& p,
                                         const std::string& prefix,
                                         const array& x,
                                         const array& x_mask,
                                         const array* g_or_null,
                                         const EncoderCfg& cfg,
                                         int half_channels);

} // namespace mlx_rt
} // namespace bv2

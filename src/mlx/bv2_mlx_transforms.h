// Inverse `piecewise_rational_quadratic_transform` with linear tails,
// matching `transforms.py` from upstream Bert-VITS2. Inference (sampling)
// only ever calls the inverse pass.

#pragma once

#include <mlx/array.h>

namespace bv2 {
namespace mlx_rt {

using ::mlx::core::array;

// Inverse spline, matching `transforms.unconstrained_rational_quadratic_spline`
// with tails="linear", tail_bound=5.0.
//
// Tensors are 4-D `[B, C, T, ?]`:
//   inputs                : [B, C, T]
//   unnormalized_widths   : [B, C, T, num_bins]
//   unnormalized_heights  : [B, C, T, num_bins]
//   unnormalized_derivs   : [B, C, T, num_bins - 1]   (interior derivatives)
//
// All math is done elementwise except for the bin-index `gather`s. The
// linear-tail outside-region keeps `outputs == inputs`.
array unconstrained_rqs_inverse(const array& inputs,
                                const array& unnormalized_widths,
                                const array& unnormalized_heights,
                                const array& unnormalized_derivatives,
                                float tail_bound = 5.0f,
                                float min_bin_width   = 1e-3f,
                                float min_bin_height  = 1e-3f,
                                float min_derivative  = 1e-3f);

} // namespace mlx_rt
} // namespace bv2

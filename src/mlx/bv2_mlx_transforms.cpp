#include "bv2_mlx_transforms.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <mlx/array.h>
#include <mlx/ops.h>

namespace mc = ::mlx::core;

namespace bv2 {
namespace mlx_rt {

namespace {

// softplus(x) = log(1 + exp(x)) numerically stable.
array softplus(const array& x) {
    array zero = mc::full(x.shape(), array(0.0f), x.dtype());
    array maxv = mc::maximum(x, zero);
    array minv = mc::minimum(x, zero);
    return mc::add(maxv, mc::log(mc::add(array(1.0f), mc::exp(mc::subtract(minv, maxv)))));
}

// Piecewise softmax along last axis with "minimum bin" floor.
//   widths_or_heights = min + (1 - min*N) * softmax(unnormalized)
// Returns the cumulative offsets `cumwidths` of shape [..., N+1] (with
// the first element forced to `left` and the last to `right`).
struct CumDecomp {
    array cumwidths;  // [..., N+1]
    array widths;     // [..., N]
};

CumDecomp build_cum(const array& unnormalized,
                    int num_bins, float left, float right,
                    float min_value) {
    array sm = mc::softmax(unnormalized, /*axes*/ {-1});
    float scale = 1.0f - min_value * static_cast<float>(num_bins);
    array widths = mc::add(array(min_value), mc::multiply(sm, array(scale)));   // [..., N]
    array cum = mc::cumsum(widths, /*axis*/ -1);                                // [..., N]
    // Pad front with 0 -> [..., N+1].
    cum = mc::pad(cum, /*axes*/ {-1}, /*low*/ {1}, /*high*/ {0},
                  /*value*/ array(0.0f));
    cum = mc::add(mc::multiply(cum, array(right - left)), array(left));

    auto sh = cum.shape();
    int rank = static_cast<int>(sh.size());

    // Force boundary endpoints: cum[..., 0] = left, cum[..., -1] = right.
    auto shape_ends = sh;
    shape_ends[rank - 1] = 1;
    array left_arr  = mc::full(shape_ends, array(left),  cum.dtype());
    array right_arr = mc::full(shape_ends, array(right), cum.dtype());

    std::vector<int> mid_starts(rank, 0); mid_starts[rank - 1] = 1;
    std::vector<int> mid_stops(sh.begin(), sh.end()); mid_stops[rank - 1] = num_bins;
    array middle = mc::slice(cum, mid_starts, mid_stops);                       // [..., N-1]
    cum = mc::concatenate({left_arr, middle, right_arr}, /*axis*/ -1);

    // widths_final = cum[..., 1:] - cum[..., :-1]
    std::vector<int> hi_starts(rank, 0); hi_starts[rank - 1] = 1;
    std::vector<int> hi_stops(sh.begin(), sh.end()); hi_stops[rank - 1] = num_bins + 1;
    std::vector<int> lo_starts(rank, 0);
    std::vector<int> lo_stops(sh.begin(), sh.end()); lo_stops[rank - 1] = num_bins;
    array hi = mc::slice(cum, hi_starts, hi_stops);
    array lo = mc::slice(cum, lo_starts, lo_stops);
    array widths_final = mc::subtract(hi, lo);                                   // [..., N]

    return {cum, widths_final};
}

// `searchsorted(bin_locations, inputs)`. Returns per-element bin index
// (0..num_bins-1).
//   bin_locations: [..., N+1] (sorted ascending along last axis)
//   inputs:        [...]
array searchsorted(const array& bin_locations, const array& inputs) {
    // We replicate `torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1`,
    // matching `transforms.searchsorted` which adds eps to the last bin.
    auto sh_b = bin_locations.shape();
    int N = static_cast<int>(sh_b.back()) - 1;
    // inputs has rank R = bin_locations.ndim - 1 (we want broadcast-compatible).
    array x = mc::expand_dims(inputs, -1);                  // [..., 1]
    array ge = mc::greater_equal(x, bin_locations);
    array ge_f = mc::astype(ge, mc::int32);
    array sum_ge = mc::sum(ge_f, /*axes*/ {-1});            // [...]
    array bin_idx = mc::subtract(sum_ge, array(1));
    // Clamp to [0, N-1] in case of numerical edge cases.
    bin_idx = mc::clip(bin_idx, array(0), array(N - 1));
    return bin_idx;
}

// Gather the bin's value from `cum_or_widths` [..., N+1 or N] using
// `bin_idx` [...]. Returns [...] shaped result.
array gather_per_bin(const array& src, const array& bin_idx) {
    array idx = mc::expand_dims(bin_idx, -1);
    array g = mc::take_along_axis(src, idx, /*axis*/ -1);
    return mc::squeeze(g, /*axis*/ -1);
}

} // namespace

array unconstrained_rqs_inverse(const array& inputs,
                                const array& unnormalized_widths,
                                const array& unnormalized_heights,
                                const array& unnormalized_derivatives,
                                float tail_bound,
                                float min_bin_width,
                                float min_bin_height,
                                float min_derivative) {
    int num_bins = static_cast<int>(unnormalized_widths.shape().back());

    // For the linear-tails variant we pad the derivatives with the boundary
    // value `softplus^-1(1 - min_derivative)`, exactly as in transforms.py.
    float c_val = std::log(std::exp(1.0f - min_derivative) - 1.0f);
    auto sh = unnormalized_derivatives.shape();
    auto sh_pad = sh;
    sh_pad.back() = 1;
    array boundary = mc::full(sh_pad, array(c_val), unnormalized_derivatives.dtype());
    array unn_d = mc::concatenate({boundary, unnormalized_derivatives, boundary}, /*axis*/ -1);

    // Build cumwidths / cumheights / widths / heights / derivatives.
    CumDecomp w_cum = build_cum(unnormalized_widths,  num_bins, -tail_bound, tail_bound, min_bin_width);
    CumDecomp h_cum = build_cum(unnormalized_heights, num_bins, -tail_bound, tail_bound, min_bin_height);
    array derivatives = mc::add(array(min_derivative), softplus(unn_d));        // [..., N+1]

    // For inverse pass, search-sort against cumheights.
    array bin_idx = searchsorted(h_cum.cumwidths, inputs);                      // [...]

    // Per-element gathers
    array input_cumwidths       = gather_per_bin(w_cum.cumwidths,            bin_idx);
    array input_bin_widths      = gather_per_bin(w_cum.widths,               bin_idx);
    array input_cumheights      = gather_per_bin(h_cum.cumwidths,            bin_idx);
    array input_heights         = gather_per_bin(h_cum.widths,               bin_idx);

    // delta = heights / widths -> [..., N]
    array delta = mc::divide(h_cum.widths, w_cum.widths);
    array input_delta = gather_per_bin(delta, bin_idx);

    array input_derivatives = gather_per_bin(derivatives, bin_idx);
    // derivatives_plus_one = derivatives[..., 1:][bin_idx]
    auto rk_d = derivatives.shape().size();
    std::vector<int> starts_d(rk_d, 0);
    starts_d.back() = 1;
    std::vector<int> stops_d(derivatives.shape().begin(), derivatives.shape().end());
    array deriv_p1 = mc::slice(derivatives, starts_d, stops_d);                 // [..., N]
    array input_derivatives_plus_one = gather_per_bin(deriv_p1, bin_idx);

    // Compute root of the cubic, exactly as in transforms.rational_quadratic_spline
    // (inverse branch).
    array two = array(2.0f);
    array four = array(4.0f);

    array d_minus = mc::subtract(inputs, input_cumheights);
    array d_sum   = mc::add(input_derivatives, input_derivatives_plus_one);
    array d_sub   = mc::subtract(d_sum, mc::multiply(two, input_delta));        // d_in + d_out - 2*delta

    array a = mc::add(mc::multiply(d_minus, d_sub),
                      mc::multiply(input_heights,
                                   mc::subtract(input_delta, input_derivatives)));
    array b = mc::subtract(mc::multiply(input_heights, input_derivatives),
                           mc::multiply(d_minus, d_sub));
    array c = mc::multiply(mc::negative(input_delta), d_minus);

    array discr = mc::subtract(mc::multiply(b, b),
                               mc::multiply(four, mc::multiply(a, c)));
    discr = mc::maximum(discr, array(0.0f));   // numerical safety
    array sqrt_d = mc::sqrt(discr);

    array root = mc::divide(mc::multiply(two, c),
                            mc::negative(mc::add(b, sqrt_d)));
    array outputs_inside = mc::add(mc::multiply(root, input_bin_widths),
                                   input_cumwidths);

    // Linear tails: keep inputs unchanged outside [-tail_bound, tail_bound].
    array tb_lo = array(-tail_bound);
    array tb_hi = array( tail_bound);
    array inside = mc::logical_and(mc::greater_equal(inputs, tb_lo),
                                   mc::less_equal   (inputs, tb_hi));
    return mc::where(inside, outputs_inside, inputs);
}

} // namespace mlx_rt
} // namespace bv2

#include "bv2_mlx_modules.h"
#include "bv2_mlx_ops.h"

#include <optional>
#include <stdexcept>
#include <utility>

#include <mlx/array.h>
#include <mlx/io.h>
#include <mlx/ops.h>

namespace mc = ::mlx::core;

namespace bv2 {
namespace mlx_rt {

// ---------------------------------------------------------------------------
// safetensors loader
// ---------------------------------------------------------------------------

ParamStore load_safetensors(const std::string& path) {
    auto loaded = mc::load_safetensors(path);
    ParamStore ps;
    ps.metadata = std::move(loaded.second);
    for (auto& kv : loaded.first) {
        ps.params.emplace(kv.first, std::move(kv.second));
    }
    return ps;
}

// Cast an array to float32 if it is a half-precision float type.
// Used by the apply helpers below so that weights can be stored fp16 on disk
// (and in ParamStore) to save ~50 % RAM, while all arithmetic stays fp32.
static inline array to_fp32(const array& a) {
    const auto dt = a.dtype();
    if (dt == mc::float16 || dt == mc::bfloat16) return mc::astype(a, mc::float32);
    return a;
}

// ---------------------------------------------------------------------------
// Layer apply helpers
// ---------------------------------------------------------------------------

namespace {

const array* get_bias(const ParamStore& p, const std::string& prefix) {
    auto it = p.params.find(prefix + ".bias");
    if (it == p.params.end()) return nullptr;
    return &it->second;
}

} // namespace

array conv1d_apply(const ParamStore& p,
                   const std::string& prefix,
                   const array& x,
                   const Conv1dCfg& cfg) {
    array w = to_fp32(p.at(prefix + ".weight"));
    const array* b_raw = get_bias(p, prefix);
    std::optional<array> b_opt;
    if (b_raw) b_opt = to_fp32(*b_raw);
    const array* b = b_opt ? &*b_opt : nullptr;
    return conv1d_pt(x, w, b, cfg.stride, cfg.padding, cfg.dilation, cfg.groups);
}

array conv1d_same(const ParamStore& p,
                  const std::string& prefix,
                  const array& x,
                  int dilation,
                  int groups) {
    array w = to_fp32(p.at(prefix + ".weight"));
    int kernel = static_cast<int>(w.shape().back());
    int padding = ((kernel - 1) * dilation) / 2;
    Conv1dCfg cfg;
    cfg.padding  = padding;
    cfg.dilation = dilation;
    cfg.groups   = groups;
    const array* b_raw = get_bias(p, prefix);
    std::optional<array> b_opt;
    if (b_raw) b_opt = to_fp32(*b_raw);
    const array* b = b_opt ? &*b_opt : nullptr;
    return conv1d_pt(x, w, b, cfg.stride, cfg.padding, cfg.dilation, cfg.groups);
}

array conv_transpose1d_apply(const ParamStore& p,
                             const std::string& prefix,
                             const array& x,
                             int stride,
                             int padding) {
    array w = to_fp32(p.at(prefix + ".weight"));
    const array* b_raw = get_bias(p, prefix);
    std::optional<array> b_opt;
    if (b_raw) b_opt = to_fp32(*b_raw);
    const array* b = b_opt ? &*b_opt : nullptr;
    return conv_transpose1d_pt(x, w, b, stride, padding, /*dilation*/ 1, /*groups*/ 1);
}

array linear_apply(const ParamStore& p,
                   const std::string& prefix,
                   const array& x) {
    array w = to_fp32(p.at(prefix + ".weight"));  // [out, in]
    array w_t = mc::transpose(w, {1, 0});          // [in, out]
    array y = mc::matmul(x, w_t);
    const array* b_raw = get_bias(p, prefix);
    if (b_raw != nullptr) {
        array b = to_fp32(*b_raw);
        y = mc::add(y, b);
    }
    return y;
}

array embedding_apply(const ParamStore& p,
                      const std::string& prefix,
                      const array& ids) {
    array w = to_fp32(p.at(prefix + ".weight"));  // [num, dim] always fp32
    return mc::take(w, ids, /*axis*/ 0);
}

array layer_norm_channel_apply(const ParamStore& p,
                               const std::string& prefix,
                               const array& x,
                               float eps) {
    array gamma = to_fp32(p.at(prefix + ".gamma"));
    array beta  = to_fp32(p.at(prefix + ".beta"));
    return layer_norm_channels(x, gamma, beta, eps);
}

} // namespace mlx_rt
} // namespace bv2

// Tiny helpers wrapping MLX `array`s for the modules that the Bert-VITS2
// generator uses. Each "module" is just a free function reading weights
// from a `ParamStore` by name. This keeps the porting close to the
// Python `state_dict` and makes the safetensors layout the source of truth.

#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>

namespace bv2 {
namespace mlx_rt {

using ::mlx::core::array;

// ---------------------------------------------------------------------------
// ParamStore
// ---------------------------------------------------------------------------

struct ParamStore {
    // Map from full state_dict key (e.g. "enc_p.proj.weight") to its array.
    std::unordered_map<std::string, array> params;
    // safetensors metadata (e.g. {"format": "bv2-mlx-v1"}).
    std::unordered_map<std::string, std::string> metadata;

    // Strict lookup. Throws if key absent.
    const array& at(const std::string& key) const {
        auto it = params.find(key);
        if (it == params.end()) {
            throw std::runtime_error("MLX param missing: " + key);
        }
        return it->second;
    }

    bool has(const std::string& key) const {
        return params.find(key) != params.end();
    }
};

// Load a safetensors file into a ParamStore.
ParamStore load_safetensors(const std::string& path);

// ---------------------------------------------------------------------------
// Conv / Linear / Embedding helpers
// ---------------------------------------------------------------------------

struct Conv1dCfg {
    int stride   = 1;
    int padding  = 0;
    int dilation = 1;
    int groups   = 1;
};

// Apply Conv1d using params under `<prefix>.weight` (and optionally
// `<prefix>.bias`).
//   x: [B, C_in, T] -> [B, C_out, T_out]
array conv1d_apply(const ParamStore& p,
                   const std::string& prefix,
                   const array& x,
                   const Conv1dCfg& cfg = {});

// Same but the kernel uses 'same' padding (kernel_size derived from the
// stored weight tensor).
array conv1d_same(const ParamStore& p,
                  const std::string& prefix,
                  const array& x,
                  int dilation = 1,
                  int groups   = 1);

// Apply ConvTranspose1d using params under `<prefix>.weight` (and bias).
//   x: [B, C_in, T] -> [B, C_out, T_out]
array conv_transpose1d_apply(const ParamStore& p,
                             const std::string& prefix,
                             const array& x,
                             int stride,
                             int padding);

// Linear: y = x @ W.T (+ bias).
//   x: [..., in_features]
//   W: [out_features, in_features] under `<prefix>.weight` (+ optional bias).
array linear_apply(const ParamStore& p,
                   const std::string& prefix,
                   const array& x);

// Embedding lookup with weights under `<prefix>.weight` shape
// [num_embeddings, embed_dim].
//   ids: [...] integer
//   ret: [..., embed_dim]
array embedding_apply(const ParamStore& p,
                      const std::string& prefix,
                      const array& ids);

// LayerNorm along the channel axis (Bert-VITS2's `modules.LayerNorm`).
// Reads `<prefix>.gamma` and `<prefix>.beta` of shape [C].
//   x: [B, C, T] -> [B, C, T]
array layer_norm_channel_apply(const ParamStore& p,
                               const std::string& prefix,
                               const array& x,
                               float eps = 1e-5f);

} // namespace mlx_rt
} // namespace bv2

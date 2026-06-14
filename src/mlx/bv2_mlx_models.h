// Top-level Bert-VITS2 v2.3 modules. See `bv2_mlx_runtime.h` for the
// public synthesis interface.

#pragma once

#include <mlx/array.h>

#include <string>

#include "bv2_mlx_attentions.h"
#include "bv2_mlx_modules.h"
#include "bv2_mlx_runtime.h"

namespace bv2 {
namespace mlx_rt {

using ::mlx::core::array;

// `models_onnx.TextEncoder.forward`. Returns the encoded x, the prior mean,
// the prior log-std, and `x_mask` (passed in pre-built).
struct TextEncoderOut {
    array x;       // [B, hidden, T_x]
    array m_p;     // [B, inter, T_x]
    array logs_p;  // [B, inter, T_x]
};

TextEncoderOut text_encoder_apply(const ParamStore& p,
                                  const array& x_ids,         // [B, T_x] int
                                  const array& tone_ids,      // [B, T_x] int
                                  const array& lang_ids,      // [B, T_x] int
                                  const array& bert_zh,       // [1024, T_x] float
                                  const array& bert_jp,       // [1024, T_x] float
                                  const array& bert_en,       // [1024, T_x] float
                                  const array& x_mask,        // [B, 1, T_x]
                                  const array* g,             // [B, gin, 1] optional
                                  const MlxConfig& cfg);

// `models_onnx.DurationPredictor.forward`.
//   x: [B, hidden, T_x], x_mask: [B, 1, T_x], g: [B, gin, 1] -> [B, 1, T_x]
array duration_predictor_apply(const ParamStore& p,
                               const std::string& prefix,
                               const array& x,
                               const array& x_mask,
                               const array* g,
                               int kernel_size,
                               int filter_channels);

// `models_onnx.StochasticDurationPredictor.forward(reverse=True path)`.
//   x: [B, hidden, T_x], x_mask: [B, 1, T_x], z: [B, 2, T_x] (initial noise),
//   g: optional [B, gin, 1] -> [B, 1, T_x] (logw).
array stochastic_duration_predictor_apply(const ParamStore& p,
                                          const std::string& prefix,
                                          const array& x,
                                          const array& x_mask,
                                          const array& z_init,
                                          const array* g,
                                          int filter_channels,
                                          int kernel_size,
                                          int n_flows);

// `models_onnx.TransformerCouplingBlock.forward(reverse=True)`.
//   x: [B, C, T], x_mask: [B, 1, T], g: [B, gin, 1].
array transformer_coupling_block_inverse(const ParamStore& p,
                                         const std::string& prefix,
                                         const array& x,
                                         const array& x_mask,
                                         const array* g,
                                         const MlxConfig& cfg);

// `models_onnx.Generator.forward`.
//   x: [B, inter, T], g: [B, gin, 1] -> [B, 1, T_audio]
array generator_apply(const ParamStore& p,
                      const std::string& prefix,
                      const array& x,
                      const array* g,
                      const MlxConfig& cfg);

// Top-level inference glue: returns the audio array [1, 1, T_audio].
array synthesizer_infer(const ParamStore& p,
                        const MlxInferInputs& in,
                        const MlxConfig& cfg);

} // namespace mlx_rt
} // namespace bv2

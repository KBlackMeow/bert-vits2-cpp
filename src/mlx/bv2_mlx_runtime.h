// Native Apple-MLX inference back-end for Bert-VITS2 v2.3 (C++ only).
//
// The architecture, op semantics, and weight layout follow the upstream
// Bert-VITS2 v2.3 generator (`fishaudio/Bert-VITS2`, branch `master`,
// `models.py` + `onnx_modules/V230/models_onnx.py`). PyTorch tensors live
// in the [B, C, T] layout; MLX's conv ops are channels-last so all
// Conv1d / ConvTranspose1d wrappers transpose internally.
//
// The user is expected to pre-convert `G_*.pth` to a plain safetensors
// file via `src/convert_to_mlx.py`. The keys remain in their PyTorch
// state_dict form, with weight_norm pre-folded into a single `.weight`.
//
// Public surface:
//   bv2::mlx_rt::MlxVitsRuntime  ::load(model_dir)
//                                ::infer(features, params) -> audio (float)

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace bv2 {
namespace mlx_rt {

struct MlxConfig {
    int n_speakers      = 6;
    int inter_channels  = 192;
    int hidden_channels = 192;
    int filter_channels = 768;
    int n_heads         = 2;
    int n_layers        = 6;
    int kernel_size     = 3;
    float p_dropout     = 0.0f;        // inference: always 0
    int gin_channels    = 512;
    int sampling_rate   = 44100;
    // Flow / decoder
    int n_flow_layer        = 4;
    int n_layers_trans_flow = 4;
    bool use_transformer_flow = true;
    int spec_channels       = 1025;    // filter_length / 2 + 1
    // HiFi-GAN
    int upsample_initial_channel = 512;
    std::vector<int> upsample_rates       = {8, 8, 2, 2, 2};
    std::vector<int> upsample_kernel_sizes = {16, 16, 8, 2, 2};
    std::string resblock_type             = "1";
    std::vector<int> resblock_kernel_sizes  = {3, 7, 11};
    std::vector<std::vector<int>> resblock_dilation_sizes
        = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    // SDP / DP fixed in v2.3
    int dp_filter_channels  = 256;
    int sdp_filter_channels = 192;
    int sdp_n_flows         = 4;
    int sdp_kernel_size     = 3;
    int dp_kernel_size      = 3;
    // Frontend (independent of inference math, kept for reference).
    int n_vocab = 112;
};

// Inference-time inputs (already produced by the existing C++ front-end).
struct MlxInferInputs {
    std::vector<int64_t> phones;       // [T_x]
    std::vector<int64_t> tones;        // [T_x]
    std::vector<int64_t> languages;    // [T_x]
    std::vector<float>   bert_zh;      // [T_x * 1024], row-major (T_x, 1024)
    std::vector<float>   bert_jp;      // [T_x * 1024]
    std::vector<float>   bert_en;      // [T_x * 1024]
    int speaker_id    = 0;
    float sdp_ratio   = 0.0f;
    float noise_scale   = 0.6f;
    float noise_scale_w = 0.9f;
    float length_scale  = 1.0f;
    uint64_t seed       = 114514ULL;
};

class MlxVitsRuntime {
public:
    MlxVitsRuntime();
    ~MlxVitsRuntime();

    MlxVitsRuntime(const MlxVitsRuntime&)            = delete;
    MlxVitsRuntime& operator=(const MlxVitsRuntime&) = delete;
    MlxVitsRuntime(MlxVitsRuntime&&) noexcept;
    MlxVitsRuntime& operator=(MlxVitsRuntime&&) noexcept;

    // Load weights + the model config from <model_dir>/G_mlx.safetensors and
    // <model_dir>/config.json. Returns false on failure (msg in `error_out`).
    bool load(const std::string& model_dir, std::string* error_out = nullptr);

    // Same, but with explicit (decoupled) paths for the safetensors weights
    // file and the config.json. Useful when the files live in different
    // directories (e.g. weights under `mlx_model/` while `config.json` stays
    // next to the original ONNX bundle in `onnx/model/`).
    bool load_files(const std::string& weights_path,
                    const std::string& config_path,
                    std::string* error_out = nullptr);

    bool is_loaded() const noexcept;

    // Synthesize one utterance. Returns the audio in [-1, 1] @ sampling_rate Hz.
    std::vector<float> infer(const MlxInferInputs& in,
                             std::string* error_out = nullptr);

    int sampling_rate() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mlx_rt
} // namespace bv2

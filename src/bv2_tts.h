#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bv2 {

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;

    Tensor() = default;
    Tensor(std::vector<int64_t> s, float value = 0.0f);

    int64_t size() const;
    float & operator()(int64_t a, int64_t b, int64_t c);
    const float & operator()(int64_t a, int64_t b, int64_t c) const;
};

struct TextFeatures {
    std::string norm_text;
    std::vector<int64_t> phones;
    std::vector<int64_t> tones;
    std::vector<int64_t> languages;
    std::vector<std::string> bert_tokens;
    std::vector<int64_t> word2ph;
};

struct ModelPaths {
    std::string enc;
    std::string emb;
    std::string dp;
    std::string sdp;
    std::string flow;
    std::string dec;
    // Optional. Used by the MLX backend to resolve the VITS hyper-parameter
    // file. When empty, `resolve_mlx_vits_files` falls back to
    // `mlx_model/config.json` and `<onnx-dir>/config.json`. Set this to the
    // CLI `--config` so user overrides win.
    std::string config;
};

struct BertPaths {
    std::string zh_model;
    std::string zh_vocab;
    std::string jp_model;
    std::string jp_vocab;
    std::string en_model;
    std::string en_spm;
};

struct SynthesisOptions {
    int64_t speaker_id = 0;
    uint32_t seed = 114514;
    float noise_scale = 0.6f;
    float noise_scale_w = 0.9f;
    float length_scale = 1.0f;
    float sdp_ratio = 0.0f;
    int sample_rate = 44100;
    bool random_aux = false;
    bool use_emotion = false;
    bool use_cuda = false;
    int cuda_device_id = 0;
    // macOS CoreML execution provider (exposed as "MLX" device on macOS).
    bool use_mlx = false;
};

TextFeatures text_to_sequence(const std::string & text, const std::string & language);

std::vector<std::pair<std::string, std::string>> split_text_by_language(
    const std::string & text, const std::string & primary_lang = "ZH");

// Sentence-level ZH/JP disambiguation for kanji-heavy text (returns ZH or JP).
std::string classify_sentence_language(const std::string & sentence, const std::string & fallback = "ZH");

struct MixedSpanInfo {
    std::string text;
    std::string lang;
    TextFeatures full_features;
    int64_t phone_offset = 0;
    int64_t phone_begin_in_full = 0;
    int64_t phone_count = 0;
};

struct MixedSequence {
    TextFeatures features;
    std::vector<MixedSpanInfo> spans;
};

MixedSequence prepare_mixed_sequence(const std::string & text, const std::string & primary_lang = "ZH");
TextFeatures text_to_sequence_mixed(const std::string & text);

TextFeatures parse_phone_ids(const std::string & csv, const std::string & language);
TextFeatures parse_phone_ids(
    const std::string & phone_csv,
    const std::string & tone_csv,
    const std::string & language_csv,
    const std::string & default_language);

Tensor zeros_bert(int64_t phones);
void copy_bert_slice(
    Tensor & dst,
    const Tensor & src,
    int64_t src_begin,
    int64_t src_end,
    int64_t dst_begin);
Tensor random_bert(int64_t phones, uint32_t seed);
Tensor chinese_bert(
    const std::string & bert_onnx_path,
    const std::string & vocab_path,
    const TextFeatures & text,
    const SynthesisOptions & options);
Tensor japanese_bert(
    const std::string & bert_onnx_path,
    const std::string & vocab_path,
    const TextFeatures & text,
    const SynthesisOptions & options);
Tensor english_bert(
    const std::string & bert_onnx_path,
    const std::string & spm_model_path,
    const std::string & norm_text,
    const TextFeatures & text,
    const SynthesisOptions & options);
Tensor zeros_emotion();
Tensor random_emotion(uint32_t seed);
Tensor load_float_bin(const std::string & path, std::vector<int64_t> shape);

void preload_synthesis_model(
    const ModelPaths & paths,
    const SynthesisOptions & options);

void preload_bert_models(
    const BertPaths & paths,
    const SynthesisOptions & options);

std::vector<float> synthesize(
    const ModelPaths & paths,
    const TextFeatures & text,
    const SynthesisOptions & options,
    const Tensor * bert_zh,
    const Tensor * bert_jp,
    const Tensor * bert_en,
    const Tensor * emotion);

void write_wav(const std::string & path, const std::vector<float> & audio, int sample_rate);

std::vector<char> encode_wav(const std::vector<float> & audio, int sample_rate);

// Little-endian signed 16-bit PCM samples in [-1, 1].
std::vector<char> encode_pcm16(const std::vector<float> & audio);

} // namespace bv2

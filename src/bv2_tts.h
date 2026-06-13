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
};

struct SynthesisOptions {
    int64_t speaker_id = 0;
    uint32_t seed = 114514;
    float noise_scale = 0.8f;
    float noise_scale_w = 0.6f;
    float length_scale = 1.1f;
    float sdp_ratio = 0.0f;
    int sample_rate = 44100;
    bool random_aux = false;
    bool use_emotion = false;
    bool use_cuda = false;
    int cuda_device_id = 0;
};

TextFeatures text_to_sequence(const std::string & text, const std::string & language);

std::vector<std::pair<std::string, std::string>> split_text_by_language(
    const std::string & text, const std::string & primary_lang = "ZH");

TextFeatures text_to_sequence_mixed(const std::string & text);

TextFeatures parse_phone_ids(const std::string & csv, const std::string & language);
TextFeatures parse_phone_ids(
    const std::string & phone_csv,
    const std::string & tone_csv,
    const std::string & language_csv,
    const std::string & default_language);

Tensor zeros_bert(int64_t phones);
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

std::vector<float> synthesize(
    const ModelPaths & paths,
    const TextFeatures & text,
    const SynthesisOptions & options,
    const Tensor * bert_zh,
    const Tensor * bert_jp,
    const Tensor * bert_en,
    const Tensor * emotion);

void write_wav(const std::string & path, const std::vector<float> & audio, int sample_rate);

} // namespace bv2

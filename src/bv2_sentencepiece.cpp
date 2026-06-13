#include "bv2_sentencepiece.h"

#include <stdexcept>

#include <map>
#include <mutex>

#include <sentencepiece_processor.h>

namespace bv2 {

struct SentencePieceTokenizer::Impl {
    sentencepiece::SentencePieceProcessor processor;
};

SentencePieceTokenizer::SentencePieceTokenizer() = default;
SentencePieceTokenizer::~SentencePieceTokenizer() = default;

void SentencePieceTokenizer::load(const std::string & model_path) {
    auto impl = std::make_unique<Impl>();
    const auto status = impl->processor.Load(model_path);
    if (!status.ok()) {
        throw std::runtime_error("failed to load SentencePiece model: " + model_path + " (" + status.ToString() + ")");
    }
    impl_ = std::move(impl);
}

std::vector<int64_t> SentencePieceTokenizer::encode(const std::string & text) const {
    if (!impl_) throw std::runtime_error("SentencePiece model is not loaded");
    std::vector<int> ids;
    const auto status = impl_->processor.Encode(text, &ids);
    if (!status.ok()) {
        throw std::runtime_error("SentencePiece encode failed: " + status.ToString());
    }
    std::vector<int64_t> out;
    out.reserve(ids.size());
    for (int id : ids) out.push_back(static_cast<int64_t>(id));
    return out;
}

std::vector<std::string> SentencePieceTokenizer::encode_pieces(const std::string & text) const {
    if (!impl_) throw std::runtime_error("SentencePiece model is not loaded");
    std::vector<std::string> pieces;
    const auto status = impl_->processor.Encode(text, &pieces);
    if (!status.ok()) {
        throw std::runtime_error("SentencePiece encode failed: " + status.ToString());
    }
    return pieces;
}

const SentencePieceTokenizer & cached_sentencepiece(const std::string & model_path) {
    static std::mutex mutex;
    static std::map<std::string, std::unique_ptr<SentencePieceTokenizer>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(model_path);
    if (it != cache.end()) return *it->second;
    auto created = std::make_unique<SentencePieceTokenizer>();
    created->load(model_path);
    const SentencePieceTokenizer & ref = *created;
    cache.emplace(model_path, std::move(created));
    return ref;
}

} // namespace bv2

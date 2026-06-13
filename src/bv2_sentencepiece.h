#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bv2 {

class SentencePieceTokenizer {
public:
    SentencePieceTokenizer();
    ~SentencePieceTokenizer();

    SentencePieceTokenizer(const SentencePieceTokenizer &) = delete;
    SentencePieceTokenizer & operator=(const SentencePieceTokenizer &) = delete;

    void load(const std::string & model_path);
    std::vector<int64_t> encode(const std::string & text) const;
    std::vector<std::string> encode_pieces(const std::string & text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bv2

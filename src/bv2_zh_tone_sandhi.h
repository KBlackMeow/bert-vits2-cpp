#pragma once

#include <functional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef BV2_WITH_CPPJIEBA
#include "cppjieba/Jieba.hpp"
#endif

namespace bv2::zh {

struct WordPos {
    std::string word;
    std::string pos;
};

using FinalsLookup = std::function<std::vector<std::string>(const std::string & word)>;

class ToneSandhi {
public:
#ifdef BV2_WITH_CPPJIEBA
    ToneSandhi(FinalsLookup finals_lookup, cppjieba::Jieba & jieba);
#else
    explicit ToneSandhi(FinalsLookup finals_lookup);
#endif

    std::vector<WordPos> pre_merge_for_modify(std::vector<WordPos> seg) const;
    std::vector<std::string> modified_tone(
        const std::string & word,
        const std::string & pos,
        std::vector<std::string> finals) const;

private:
    std::vector<std::string> bu_sandhi(const std::string & word, std::vector<std::string> finals) const;
    std::vector<std::string> yi_sandhi(const std::string & word, std::vector<std::string> finals) const;
    std::vector<std::string> neural_sandhi(
        const std::string & word,
        const std::string & pos,
        std::vector<std::string> finals) const;
    std::vector<std::string> three_sandhi(const std::string & word, std::vector<std::string> finals) const;

    std::vector<std::string> split_word(const std::string & word) const;
    bool all_tone_three(const std::vector<std::string> & finals) const;
    bool is_reduplication(const std::string & word) const;

    std::vector<WordPos> merge_bu(std::vector<WordPos> seg) const;
    std::vector<WordPos> merge_yi(std::vector<WordPos> seg) const;
    std::vector<WordPos> merge_reduplication(std::vector<WordPos> seg) const;
    std::vector<WordPos> merge_continuous_three_tones(std::vector<WordPos> seg) const;
    std::vector<WordPos> merge_continuous_three_tones_2(std::vector<WordPos> seg) const;
    std::vector<WordPos> merge_er(std::vector<WordPos> seg) const;

    FinalsLookup finals_lookup_;
#ifdef BV2_WITH_CPPJIEBA
    cppjieba::Jieba * jieba_;
#endif
    std::unordered_set<std::string> must_neural_tone_words_;
    std::unordered_set<std::string> must_not_neural_tone_words_;
    std::string punc_;
};

} // namespace bv2::zh

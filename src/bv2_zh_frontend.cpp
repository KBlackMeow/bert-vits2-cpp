#include "bv2_zh_frontend.h"
#include "bv2_zh_tone_sandhi.h"
#include "bv2_text_internal.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef BV2_WITH_CPPJIEBA
#include "cppjieba/Jieba.hpp"
#endif

namespace bv2::zh {
namespace {

using internal::add_phone;
using internal::first_existing_path;
using internal::intersperse_blank;
using internal::punctuation_symbol;
using internal::utf8_chars;
using internal::zh_pinyin_table;
using internal::zh_pinyin_to_symbols;
using internal::normalize_chinese_numbers;
using internal::normalize_chinese_punctuation;
using internal::normalize_date_formats;

const std::string & zh_punctuation() {
    static const std::string value = "!,?…,.:'-";
    return value;
}

bool is_chinese_char(const std::string & ch) {
    if (ch.empty()) {
        return false;
    }
    const unsigned char c0 = static_cast<unsigned char>(ch[0]);
    if ((c0 & 0xE0) != 0xE0 || ch.size() < 3) {
        return false;
    }
    const unsigned char c1 = static_cast<unsigned char>(ch[1]);
    const unsigned char c2 = static_cast<unsigned char>(ch[2]);
    const uint32_t cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    return cp >= 0x4E00 && cp <= 0x9FFF;
}

bool is_allowed_char(const std::string & ch) {
    if (ch == "…") {
        return true;
    }
    return is_chinese_char(ch) || zh_punctuation().find(ch) != std::string::npos;
}

std::string trim_ascii_ws(const std::string & text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

std::string replace_special_chars(std::string text) {
    size_t pos = 0;
    while ((pos = text.find("嗯", pos)) != std::string::npos) {
        text.replace(pos, 3, "恩");
        pos += 3;
    }
    pos = 0;
    while ((pos = text.find("呣", pos)) != std::string::npos) {
        text.replace(pos, 3, "母");
        pos += 3;
    }
    return text;
}

std::string filter_allowed_chars(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (const std::string & ch : utf8_chars(text)) {
        if (is_allowed_char(ch)) {
            out += ch;
        }
    }
    return out;
}

std::string strip_english(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (std::isalpha(c)) {
            while (i < text.size() && std::isalpha(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            continue;
        }
        size_t n = 1;
        if ((c & 0xE0) == 0xC0) {
            n = 2;
        } else if ((c & 0xF0) == 0xE0) {
            n = 3;
        } else if ((c & 0xF8) == 0xF0) {
            n = 4;
        }
        if (i + n > text.size()) {
            n = 1;
        }
        out.append(text, i, n);
        i += n;
    }
    return out;
}

std::vector<std::string> split_sentences(const std::string & text) {
    std::vector<std::string> sentences;
    std::string current;
    for (const std::string & ch : utf8_chars(text)) {
        current += ch;
        if (zh_punctuation().find(ch) != std::string::npos) {
            const std::string trimmed = trim_ascii_ws(current);
            if (!trimmed.empty()) {
                sentences.push_back(trimmed);
            }
            current.clear();
        }
    }
    const std::string trimmed = trim_ascii_ws(current);
    if (!trimmed.empty()) {
        sentences.push_back(trimmed);
    }
    return sentences;
}

std::pair<std::string, std::string> split_initial_final(std::string py) {
    char tone = '5';
    if (!py.empty() && py.back() >= '1' && py.back() <= '5') {
        tone = py.back();
        py.pop_back();
    }
    static const char * two_initials[] = {"zh", "ch", "sh"};
    for (const char * ini : two_initials) {
        const std::string prefix(ini);
        if (py.size() >= prefix.size() && py.compare(0, prefix.size(), prefix) == 0) {
            return {prefix, py.substr(prefix.size()) + tone};
        }
    }
    static const char * one_initials[] = {
        "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
        "j", "q", "x", "r", "z", "c", "s", "y", "w",
    };
    if (!py.empty()) {
        const std::string first = py.substr(0, 1);
        for (const char * ini : one_initials) {
            if (first == ini) {
                return {first, py.substr(1) + tone};
            }
        }
    }
    return {"", py + tone};
}

std::vector<std::string> split_pipe(const std::string & value) {
    std::vector<std::string> out;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t sep = value.find('|', begin);
        if (sep == std::string::npos) {
            out.push_back(value.substr(begin));
            break;
        }
        out.push_back(value.substr(begin, sep - begin));
        begin = sep + 1;
    }
    return out;
}

const std::map<std::string, std::string> & zh_word_pinyin_table() {
    static std::map<std::string, std::string> table;
    static std::once_flag once;
    std::call_once(once, []() {
        const std::string path = first_existing_path({
            "src/zh_word_pinyin.tsv",
            "../src/zh_word_pinyin.tsv",
            "../../src/zh_word_pinyin.tsv",
        });
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("failed to open zh_word_pinyin.tsv: " + path);
        }
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                continue;
            }
            table[line.substr(0, tab)] = line.substr(tab + 1);
        }
    });
    return table;
}

std::vector<std::string> lookup_word_finals(const std::string & word) {
    const auto it = zh_word_pinyin_table().find(word);
    if (it == zh_word_pinyin_table().end()) {
        return {};
    }
    return split_pipe(it->second);
}

std::vector<std::string> lookup_char_finals(const std::string & word) {
    std::vector<std::string> finals;
    for (const std::string & ch : utf8_chars(word)) {
        const auto it = zh_pinyin_table().find(ch);
        if (it == zh_pinyin_table().end()) {
            finals.push_back(ch);
        } else {
            finals.push_back(split_initial_final(it->second).second);
        }
    }
    return finals;
}

std::pair<std::vector<std::string>, std::vector<std::string>> get_initials_finals(const std::string & word) {
    std::vector<std::string> initials;
    std::vector<std::string> finals;
    for (const std::string & ch : utf8_chars(word)) {
        const auto it = zh_pinyin_table().find(ch);
        if (it == zh_pinyin_table().end()) {
            initials.push_back(ch);
            finals.push_back(ch);
        } else {
            const auto split = split_initial_final(it->second);
            initials.push_back(split.first);
            finals.push_back(split.second);
        }
    }
    return {initials, finals};
}

std::string normalize_pinyin_key(std::string pinyin, const std::string & initial, const std::string & final_with_tone) {
    std::string v_without_tone = final_with_tone.substr(0, final_with_tone.size() - 1);
    if (!initial.empty()) {
        static const std::map<std::string, std::string> v_rep = {
            {"uei", "ui"},
            {"iou", "iu"},
            {"uen", "un"},
        };
        const auto it = v_rep.find(v_without_tone);
        if (it != v_rep.end()) {
            return initial + it->second;
        }
        return initial + v_without_tone;
    }

    static const std::map<std::string, std::string> pinyin_rep = {
        {"ing", "ying"},
        {"i", "yi"},
        {"in", "yin"},
        {"u", "wu"},
    };
    const auto rep_it = pinyin_rep.find(pinyin);
    if (rep_it != pinyin_rep.end()) {
        return rep_it->second;
    }
    if (!pinyin.empty()) {
        if (pinyin[0] == 'v') {
            return "yu" + pinyin.substr(1);
        }
        if (pinyin[0] == 'i') {
            return "y" + pinyin.substr(1);
        }
        if (pinyin[0] == 'u') {
            return "w" + pinyin.substr(1);
        }
    }
    return pinyin;
}

std::pair<std::vector<std::string>, int64_t> syllable_to_phones(
    const std::string & initial,
    const std::string & final_with_tone) {
    if (initial == final_with_tone) {
        const std::string punct = punctuation_symbol(initial);
        if (!punct.empty()) {
            return {{punct}, 0};
        }
        return {{"UNK"}, 0};
    }
    if (final_with_tone.empty() || final_with_tone.back() < '1' || final_with_tone.back() > '5') {
        return {{"UNK"}, 0};
    }
    const int64_t tone = final_with_tone.back() - '0';
    const std::string v_without_tone = final_with_tone.substr(0, final_with_tone.size() - 1);
    const std::string raw_pinyin = initial + v_without_tone;
    const std::string pinyin = normalize_pinyin_key(raw_pinyin, initial, final_with_tone);
    const auto & table = zh_pinyin_to_symbols();
    const auto it = table.find(pinyin);
    if (it == table.end()) {
        return {{"UNK"}, 0};
    }
    return {it->second, tone};
}

#ifdef BV2_WITH_CPPJIEBA
cppjieba::Jieba & jieba_instance() {
    static std::unique_ptr<cppjieba::Jieba> instance;
    static std::once_flag once;
    std::call_once(once, []() {
        const std::string dict_file = first_existing_path({
            "text/jieba/jieba.dict.utf8",
            "../text/jieba/jieba.dict.utf8",
            "../../text/jieba/jieba.dict.utf8",
        });
        const size_t slash = dict_file.find_last_of("/\\");
        const std::string dir = slash == std::string::npos ? "." : dict_file.substr(0, slash);
        instance = std::make_unique<cppjieba::Jieba>(
            dir + "/jieba.dict.utf8",
            dir + "/hmm_model.utf8",
            dir + "/user.dict.utf8",
            dir + "/idf.utf8",
            dir + "/stop_words.utf8");
    });
    return *instance;
}
#endif

ToneSandhi & tone_sandhi_instance() {
    static std::unique_ptr<ToneSandhi> instance;
    static std::once_flag once;
    std::call_once(once, []() {
        FinalsLookup lookup = [](const std::string & word) {
            return lookup_char_finals(word);
        };
#ifdef BV2_WITH_CPPJIEBA
        instance = std::make_unique<ToneSandhi>(lookup, jieba_instance());
#else
        instance = std::make_unique<ToneSandhi>(lookup);
#endif
    });
    return *instance;
}

void process_segment(const std::string & segment, TextFeatures & out) {
    const std::string cleaned = strip_english(segment);
    if (cleaned.empty()) {
        return;
    }

#ifdef BV2_WITH_CPPJIEBA
    std::vector<std::pair<std::string, std::string>> tagged;
    jieba_instance().Tag(cleaned, tagged);
    std::vector<WordPos> seg;
    seg.reserve(tagged.size());
    for (const auto & item : tagged) {
        seg.push_back({item.first, item.second});
    }
#else
    std::vector<WordPos> seg;
    for (const std::string & ch : utf8_chars(cleaned)) {
        seg.push_back({ch, "x"});
    }
#endif

    ToneSandhi & tone_sandhi = tone_sandhi_instance();
    seg = tone_sandhi.pre_merge_for_modify(std::move(seg));

    for (const WordPos & item : seg) {
        if (item.pos == "eng") {
            continue;
        }
        auto split = get_initials_finals(item.word);
        std::vector<std::string> & initials = split.first;
        std::vector<std::string> finals = tone_sandhi.modified_tone(item.word, item.pos, std::move(split.second));
        const std::vector<std::string> chars = utf8_chars(item.word);
        if (initials.size() != finals.size() || initials.size() != chars.size()) {
            continue;
        }
        for (size_t i = 0; i < initials.size(); ++i) {
            const auto phones_tone = syllable_to_phones(initials[i], finals[i]);
            int64_t phone_count = 0;
            for (const std::string & symbol : phones_tone.first) {
                add_phone(out, symbol, phones_tone.second, "ZH");
                ++phone_count;
            }
            out.word2ph.push_back(phone_count);
            out.bert_tokens.push_back(chars[i]);
        }
    }
}

} // namespace

std::string text_normalize(const std::string & text) {
    std::string normalized = replace_special_chars(text);
    normalized = normalize_date_formats(normalized);
    normalized = normalize_chinese_numbers(normalized);
    normalized = normalize_chinese_punctuation(std::move(normalized));
    return filter_allowed_chars(normalized);
}

TextFeatures text_to_sequence(const std::string & text) {
    const std::string normalized = text_normalize(text);
    TextFeatures raw;
    raw.norm_text = normalized;
    add_phone(raw, "_", 0, "ZH");
    raw.word2ph.push_back(1);

    for (const std::string & sentence : split_sentences(normalized)) {
        process_segment(sentence, raw);
    }

    add_phone(raw, "_", 0, "ZH");
    raw.word2ph.push_back(1);

    if (raw.bert_tokens.size() + 2 != raw.word2ph.size()) {
        throw std::runtime_error("Chinese frontend: BERT tokens and word2ph length mismatch");
    }
    return intersperse_blank(raw);
}

} // namespace bv2::zh

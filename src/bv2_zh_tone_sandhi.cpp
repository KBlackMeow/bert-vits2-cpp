#include "bv2_zh_tone_sandhi.h"
#include "bv2_text_internal.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace bv2::zh {
namespace {

using internal::first_existing_path;
using internal::utf8_chars;

std::unordered_set<std::string> load_word_set(const char * leaf) {
    const std::string path = first_existing_path({
        (std::string("src/") + leaf).c_str(),
        (std::string("../src/") + leaf).c_str(),
        (std::string("../../src/") + leaf).c_str(),
    });
    std::unordered_set<std::string> words;
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open tone sandhi word list: " + path);
    }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            words.insert(line);
        }
    }
    return words;
}

std::string utf8_suffix(const std::string & word, size_t count) {
    const std::vector<std::string> chars = utf8_chars(word);
    if (chars.size() <= count) {
        return word;
    }
    std::string out;
    for (size_t i = chars.size() - count; i < chars.size(); ++i) {
        out += chars[i];
    }
    return out;
}

bool utf8_starts_with(const std::string & word, const std::string & prefix) {
    return word.size() >= prefix.size() && word.compare(0, prefix.size(), prefix) == 0;
}

bool is_numeric_char(const std::string & ch) {
    if (ch.size() == 1) {
        const char c = ch[0];
        return c >= '0' && c <= '9';
    }
    static const char * numeric_chars[] = {
        "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "两", "半",
    };
    for (const char * item : numeric_chars) {
        if (ch == item) {
            return true;
        }
    }
    return false;
}

bool contains_char(const std::string & word, const std::string & target) {
    return word.find(target) != std::string::npos;
}

int64_t utf8_char_count(const std::string & word) {
    return static_cast<int64_t>(utf8_chars(word).size());
}

std::string utf8_char_at(const std::string & word, size_t index) {
    const std::vector<std::string> chars = utf8_chars(word);
    if (index >= chars.size()) {
        return {};
    }
    return chars[index];
}

char tone_digit(const std::string & final_tone) {
    return final_tone.empty() ? '5' : final_tone.back();
}

std::string set_tone(const std::string & final_tone, char tone) {
    if (final_tone.empty()) {
        return std::string(1, tone);
    }
    return final_tone.substr(0, final_tone.size() - 1) + tone;
}

bool set_contains_char(const std::string & chars, const std::string & ch) {
    const std::vector<std::string> items = utf8_chars(chars);
    for (const std::string & item : items) {
        if (item == ch) {
            return true;
        }
    }
    return false;
}

size_t utf8_index_of(const std::string & word, const std::string & target) {
    const std::vector<std::string> chars = utf8_chars(word);
    for (size_t i = 0; i < chars.size(); ++i) {
        if (chars[i] == target) {
            return i;
        }
    }
    return std::string::npos;
}

} // namespace

#ifdef BV2_WITH_CPPJIEBA
ToneSandhi::ToneSandhi(FinalsLookup finals_lookup, cppjieba::Jieba & jieba)
    : finals_lookup_(std::move(finals_lookup)),
      jieba_(&jieba),
      must_neural_tone_words_(load_word_set("zh_neural_tone_words.txt")),
      must_not_neural_tone_words_(load_word_set("zh_not_neural_tone_words.txt")),
      punc_("：，；。？！\"\"''':,;.?!") {
}
#else
ToneSandhi::ToneSandhi(FinalsLookup finals_lookup)
    : finals_lookup_(std::move(finals_lookup)),
      must_neural_tone_words_(load_word_set("zh_neural_tone_words.txt")),
      must_not_neural_tone_words_(load_word_set("zh_not_neural_tone_words.txt")),
      punc_("：，；。？！\"\"''':,;.?!") {
}
#endif

std::vector<std::string> ToneSandhi::split_word(const std::string & word) const {
#ifdef BV2_WITH_CPPJIEBA
    std::vector<std::string> word_list;
    jieba_->CutForSearch(word, word_list, true);
    std::sort(word_list.begin(), word_list.end(), [](const std::string & a, const std::string & b) {
        return a.size() < b.size();
    });
    if (word_list.empty()) {
        return {word, {}};
    }
    const std::string & first_subword = word_list.front();
    const size_t first_begin_idx = word.find(first_subword);
    if (first_begin_idx == 0) {
        return {first_subword, word.substr(first_subword.size())};
    }
    return {word.substr(0, word.size() - first_subword.size()), first_subword};
#else
    return {word, {}};
#endif
}

bool ToneSandhi::all_tone_three(const std::vector<std::string> & finals) const {
    if (finals.empty()) {
        return false;
    }
    for (const std::string & item : finals) {
        if (tone_digit(item) != '3') {
            return false;
        }
    }
    return true;
}

bool ToneSandhi::is_reduplication(const std::string & word) const {
    const std::vector<std::string> chars = utf8_chars(word);
    return chars.size() == 2 && chars[0] == chars[1];
}

std::vector<std::string> ToneSandhi::bu_sandhi(
    const std::string & word,
    std::vector<std::string> finals) const {
    const std::vector<std::string> chars = utf8_chars(word);
    if (chars.size() == 3 && chars[1] == "不") {
        finals[1] = set_tone(finals[1], '5');
        return finals;
    }
    for (size_t i = 0; i < chars.size(); ++i) {
        if (chars[i] == "不" && i + 1 < chars.size() && tone_digit(finals[i + 1]) == '4') {
            finals[i] = set_tone(finals[i], '2');
        }
    }
    return finals;
}

std::vector<std::string> ToneSandhi::yi_sandhi(
    const std::string & word,
    std::vector<std::string> finals) const {
    if (contains_char(word, "一")) {
        bool all_numeric = true;
        const std::vector<std::string> chars = utf8_chars(word);
        for (const std::string & ch : chars) {
            if (ch != "一" && !is_numeric_char(ch)) {
                all_numeric = false;
                break;
            }
        }
        if (all_numeric) {
            return finals;
        }
    }

    const std::vector<std::string> chars = utf8_chars(word);
    if (chars.size() == 3 && chars[1] == "一" && chars[0] == chars[2]) {
        finals[1] = set_tone(finals[1], '5');
        return finals;
    }
    if (utf8_starts_with(word, "第一") && finals.size() > 1) {
        finals[1] = set_tone(finals[1], '1');
        return finals;
    }

    for (size_t i = 0; i < chars.size(); ++i) {
        if (chars[i] != "一" || i + 1 >= chars.size()) {
            continue;
        }
        if (tone_digit(finals[i + 1]) == '4') {
            finals[i] = set_tone(finals[i], '2');
        } else if (!set_contains_char(punc_, chars[i + 1])) {
            finals[i] = set_tone(finals[i], '4');
        }
    }
    return finals;
}

std::vector<std::string> ToneSandhi::neural_sandhi(
    const std::string & word,
    const std::string & pos,
    std::vector<std::string> finals) const {
    const std::vector<std::string> chars = utf8_chars(word);
    const char pos0 = pos.empty() ? '\0' : pos[0];

    for (size_t j = 0; j < chars.size(); ++j) {
        if (j > 0
            && chars[j] == chars[j - 1]
            && (pos0 == 'n' || pos0 == 'v' || pos0 == 'a')
            && must_not_neural_tone_words_.count(word) == 0) {
            finals[j] = set_tone(finals[j], '5');
        }
    }

    const size_t ge_idx = utf8_index_of(word, "个");
    if (!chars.empty() && set_contains_char("吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶", chars.back())) {
        finals.back() = set_tone(finals.back(), '5');
    } else if (!chars.empty() && set_contains_char("的地得", chars.back())) {
        finals.back() = set_tone(finals.back(), '5');
    } else if (chars.size() > 1
        && set_contains_char("们子", chars.back())
        && (pos == "r" || pos == "n")
        && must_not_neural_tone_words_.count(word) == 0) {
        finals.back() = set_tone(finals.back(), '5');
    } else if (chars.size() > 1
        && set_contains_char("上下里", chars.back())
        && (pos == "s" || pos == "l" || pos == "f")) {
        finals.back() = set_tone(finals.back(), '5');
    } else if (chars.size() > 1
        && set_contains_char("来去", chars.back())
        && set_contains_char("上下进出回过起开", chars[chars.size() - 2])) {
        finals.back() = set_tone(finals.back(), '5');
    } else if ((ge_idx != std::string::npos && ge_idx > 0
            && (is_numeric_char(utf8_char_at(word, ge_idx - 1))
                || set_contains_char("几有两半多各整每做是", utf8_char_at(word, ge_idx - 1))))
        || word == "个") {
        if (ge_idx < finals.size()) {
            finals[ge_idx] = set_tone(finals[ge_idx], '5');
        }
    } else if (must_neural_tone_words_.count(word) > 0
        || must_neural_tone_words_.count(utf8_suffix(word, 2)) > 0) {
        finals.back() = set_tone(finals.back(), '5');
    }

    const std::vector<std::string> word_list = split_word(word);
    std::vector<std::vector<std::string>> finals_list;
    if (!word_list.empty()) {
        const size_t split = utf8_chars(word_list[0]).size();
        finals_list.push_back(std::vector<std::string>(finals.begin(), finals.begin() + static_cast<std::ptrdiff_t>(split)));
        finals_list.push_back(std::vector<std::string>(finals.begin() + static_cast<std::ptrdiff_t>(split), finals.end()));
    } else {
        finals_list.push_back(finals);
    }

    for (size_t i = 0; i < word_list.size() && i < finals_list.size(); ++i) {
        if (!finals_list[i].empty()
            && (must_neural_tone_words_.count(word_list[i]) > 0
                || must_neural_tone_words_.count(utf8_suffix(word_list[i], 2)) > 0)) {
            finals_list[i].back() = set_tone(finals_list[i].back(), '5');
        }
    }

    finals.clear();
    for (const std::vector<std::string> & part : finals_list) {
        finals.insert(finals.end(), part.begin(), part.end());
    }
    return finals;
}

std::vector<std::string> ToneSandhi::three_sandhi(
    const std::string & word,
    std::vector<std::string> finals) const {
    const std::vector<std::string> chars = utf8_chars(word);
    if (chars.size() == 2 && all_tone_three(finals)) {
        finals[0] = set_tone(finals[0], '2');
    } else if (chars.size() == 3) {
        const std::vector<std::string> word_list = split_word(word);
        if (all_tone_three(finals)) {
            const size_t first_len = utf8_chars(word_list[0]).size();
            if (first_len == 2) {
                finals[0] = set_tone(finals[0], '2');
                finals[1] = set_tone(finals[1], '2');
            } else if (first_len == 1 && finals.size() > 1) {
                finals[1] = set_tone(finals[1], '2');
            }
        } else if (word_list.size() == 2) {
            const size_t split = utf8_chars(word_list[0]).size();
            std::vector<std::string> left(finals.begin(), finals.begin() + static_cast<std::ptrdiff_t>(split));
            std::vector<std::string> right(finals.begin() + static_cast<std::ptrdiff_t>(split), finals.end());
            if (left.size() == 2 && all_tone_three(left)) {
                left[0] = set_tone(left[0], '2');
            }
            if (right.size() == 2 && all_tone_three(right)) {
                right[0] = set_tone(right[0], '2');
            } else if (right.size() >= 1
                && !all_tone_three(right)
                && tone_digit(right[0]) == '3'
                && !left.empty()
                && tone_digit(left.back()) == '3') {
                left.back() = set_tone(left.back(), '2');
            }
            finals = left;
            finals.insert(finals.end(), right.begin(), right.end());
        }
    } else if (chars.size() == 4) {
        std::vector<std::string> left(finals.begin(), finals.begin() + 2);
        std::vector<std::string> right(finals.begin() + 2, finals.end());
        finals.clear();
        if (all_tone_three(left)) {
            left[0] = set_tone(left[0], '2');
        }
        if (all_tone_three(right)) {
            right[0] = set_tone(right[0], '2');
        }
        finals.insert(finals.end(), left.begin(), left.end());
        finals.insert(finals.end(), right.begin(), right.end());
    }
    return finals;
}

std::vector<WordPos> ToneSandhi::merge_bu(std::vector<WordPos> seg) const {
    std::vector<WordPos> new_seg;
    std::string last_word;
    for (const WordPos & item : seg) {
        std::string word = item.word;
        if (last_word == "不") {
            word = last_word + word;
        }
        if (word != "不") {
            new_seg.push_back({word, item.pos});
        }
        last_word = item.word;
    }
    if (last_word == "不") {
        new_seg.push_back({"不", "d"});
    }
    return new_seg;
}

std::vector<WordPos> ToneSandhi::merge_yi(std::vector<WordPos> seg) const {
    struct MutableWordPos {
        std::string word;
        std::string pos;
        bool removed = false;
    };
    std::vector<MutableWordPos> new_seg;
    new_seg.reserve(seg.size());
    for (const WordPos & item : seg) {
        new_seg.push_back({item.word, item.pos, false});
    }

    for (size_t i = 0; i < seg.size(); ++i) {
        const std::string & word = seg[i].word;
        const std::string & pos = seg[i].pos;
        if (i > 0
            && word == "一"
            && i + 1 < seg.size()
            && seg[i - 1].word == seg[i + 1].word
            && seg[i - 1].pos == "v") {
            new_seg[i - 1].word = seg[i - 1].word + "一" + seg[i + 1].word;
            new_seg[i].removed = true;
            new_seg[i + 1].removed = true;
        } else if (i >= 2
            && seg[i - 1].word == "一"
            && seg[i - 2].word == word
            && pos == "v") {
            new_seg[i].removed = true;
        }
    }

    std::vector<WordPos> filtered;
    filtered.reserve(new_seg.size());
    for (const MutableWordPos & item : new_seg) {
        if (!item.removed && !item.word.empty()) {
            filtered.push_back({item.word, item.pos});
        }
    }

    std::vector<WordPos> merged;
    merged.reserve(filtered.size());
    for (const WordPos & item : filtered) {
        if (!merged.empty() && merged.back().word == "一") {
            merged.back().word += item.word;
        } else {
            merged.push_back(item);
        }
    }
    return merged;
}

std::vector<WordPos> ToneSandhi::merge_reduplication(std::vector<WordPos> seg) const {
    std::vector<WordPos> new_seg;
    for (const WordPos & item : seg) {
        if (!new_seg.empty() && item.word == new_seg.back().word) {
            new_seg.back().word += item.word;
        } else {
            new_seg.push_back(item);
        }
    }
    return new_seg;
}

std::vector<WordPos> ToneSandhi::merge_continuous_three_tones(std::vector<WordPos> seg) const {
    std::vector<WordPos> new_seg;
    std::vector<bool> merge_last(seg.size(), false);
    for (size_t i = 0; i < seg.size(); ++i) {
        const WordPos & item = seg[i];
        if (i > 0
            && all_tone_three(finals_lookup_(seg[i - 1].word))
            && all_tone_three(finals_lookup_(item.word))
            && !merge_last[i - 1]
            && !is_reduplication(seg[i - 1].word)
            && utf8_char_count(seg[i - 1].word) + utf8_char_count(item.word) <= 3) {
            new_seg.back().word += item.word;
            merge_last[i] = true;
        } else {
            new_seg.push_back(item);
        }
    }
    return new_seg;
}

std::vector<WordPos> ToneSandhi::merge_continuous_three_tones_2(std::vector<WordPos> seg) const {
    std::vector<WordPos> new_seg;
    std::vector<bool> merge_last(seg.size(), false);
    for (size_t i = 0; i < seg.size(); ++i) {
        const WordPos & item = seg[i];
        const std::vector<std::string> prev_finals = i > 0 ? finals_lookup_(seg[i - 1].word) : std::vector<std::string>{};
        const std::vector<std::string> cur_finals = finals_lookup_(item.word);
        if (i > 0
            && !prev_finals.empty()
            && !cur_finals.empty()
            && tone_digit(prev_finals.back()) == '3'
            && tone_digit(cur_finals.front()) == '3'
            && !merge_last[i - 1]
            && !is_reduplication(seg[i - 1].word)
            && utf8_char_count(seg[i - 1].word) + utf8_char_count(item.word) <= 3) {
            new_seg.back().word += item.word;
            merge_last[i] = true;
        } else {
            new_seg.push_back(item);
        }
    }
    return new_seg;
}

std::vector<WordPos> ToneSandhi::merge_er(std::vector<WordPos> seg) const {
    std::vector<WordPos> new_seg;
    for (size_t i = 0; i < seg.size(); ++i) {
        const WordPos & item = seg[i];
        if (i > 0 && item.word == "儿" && new_seg.back().word != "#") {
            new_seg.back().word += item.word;
        } else {
            new_seg.push_back(item);
        }
    }
    return new_seg;
}

std::vector<WordPos> ToneSandhi::pre_merge_for_modify(std::vector<WordPos> seg) const {
    seg = merge_bu(std::move(seg));
    seg = merge_yi(std::move(seg));
    seg = merge_reduplication(std::move(seg));
    seg = merge_continuous_three_tones(std::move(seg));
    seg = merge_continuous_three_tones_2(std::move(seg));
    seg = merge_er(std::move(seg));
    return seg;
}

std::vector<std::string> ToneSandhi::modified_tone(
    const std::string & word,
    const std::string & pos,
    std::vector<std::string> finals) const {
    finals = bu_sandhi(word, std::move(finals));
    finals = yi_sandhi(word, std::move(finals));
    finals = neural_sandhi(word, pos, std::move(finals));
    finals = three_sandhi(word, std::move(finals));
    return finals;
}

} // namespace bv2::zh

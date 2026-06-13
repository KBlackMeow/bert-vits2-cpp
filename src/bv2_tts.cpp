#include "bv2_tts.h"
#include "bv2_openjtalk.h"
#include "bv2_sentencepiece.h"
#include "bv2_text_internal.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#ifdef BV2_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#include <windows.h>
#endif
#endif

namespace bv2 {
namespace {

constexpr int64_t kBertDim = 1024;
constexpr int64_t kEmotionDim = 512;

} // namespace

namespace internal {

const std::map<std::string, int64_t> & symbol_ids() {
    static const std::map<std::string, int64_t> ids = {
        {"_", 0}, {"AA", 1}, {"E", 2}, {"EE", 3}, {"En", 4}, {"N", 5}, {"OO", 6},
        {"V", 7}, {"a", 8}, {"a:", 9}, {"aa", 10}, {"ae", 11}, {"ah", 12},
        {"ai", 13}, {"an", 14}, {"ang", 15}, {"ao", 16}, {"aw", 17}, {"ay", 18},
        {"b", 19}, {"by", 20}, {"c", 21}, {"ch", 22}, {"d", 23}, {"dh", 24},
        {"dy", 25}, {"e", 26}, {"e:", 27}, {"eh", 28}, {"ei", 29}, {"en", 30},
        {"eng", 31}, {"er", 32}, {"ey", 33}, {"f", 34}, {"g", 35}, {"gy", 36},
        {"h", 37}, {"hh", 38}, {"hy", 39}, {"i", 40}, {"i0", 41}, {"i:", 42},
        {"ia", 43}, {"ian", 44}, {"iang", 45}, {"iao", 46}, {"ie", 47},
        {"ih", 48}, {"in", 49}, {"ing", 50}, {"iong", 51}, {"ir", 52},
        {"iu", 53}, {"iy", 54}, {"j", 55}, {"jh", 56}, {"k", 57}, {"ky", 58},
        {"l", 59}, {"m", 60}, {"my", 61}, {"n", 62}, {"ng", 63}, {"ny", 64},
        {"o", 65}, {"o:", 66}, {"ong", 67}, {"ou", 68}, {"ow", 69}, {"oy", 70},
        {"p", 71}, {"py", 72}, {"q", 73}, {"r", 74}, {"ry", 75}, {"s", 76},
        {"sh", 77}, {"t", 78}, {"th", 79}, {"ts", 80}, {"ty", 81}, {"u", 82},
        {"u:", 83}, {"ua", 84}, {"uai", 85}, {"uan", 86}, {"uang", 87},
        {"uh", 88}, {"ui", 89}, {"un", 90}, {"uo", 91}, {"uw", 92}, {"v", 93},
        {"van", 94}, {"ve", 95}, {"vn", 96}, {"w", 97}, {"x", 98}, {"y", 99},
        {"z", 100}, {"zh", 101}, {"zy", 102}, {"!", 103}, {"?", 104},
        {",", 106}, {".", 107}, {"'", 108}, {"-", 109}, {"SP", 110}, {"UNK", 111},
    };
    return ids;
}

int64_t language_id(const std::string & language) {
    if (language == "ZH") return 0;
    if (language == "JP") return 1;
    if (language == "EN") return 2;
    throw std::runtime_error("language must be one of ZH, JP, EN");
}

int64_t tone_start(const std::string & language) {
    if (language == "ZH") return 0;
    if (language == "JP") return 6;
    if (language == "EN") return 8;
    throw std::runtime_error("language must be one of ZH, JP, EN");
}

void add_phone(TextFeatures & out, const std::string & symbol, int64_t tone, const std::string & language) {
    const auto it = symbol_ids().find(symbol);
    if (it == symbol_ids().end()) {
        throw std::runtime_error("unknown phone symbol: " + symbol);
    }
    out.phones.push_back(it->second);
    out.tones.push_back(tone + tone_start(language));
    out.languages.push_back(language_id(language));
}

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

bool file_exists(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return static_cast<bool>(in);
}

std::string first_existing_path(std::initializer_list<const char *> paths) {
    for (const char * path : paths) {
        if (file_exists(path)) return path;
    }
    throw std::runtime_error("required frontend asset was not found");
}

std::vector<int64_t> parse_i64_csv(const std::string & csv) {
    std::vector<int64_t> values;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        values.push_back(std::stoll(item));
    }
    return values;
}

std::vector<std::string> split_ws(const std::string & value) {
    std::stringstream ss(value);
    std::vector<std::string> out;
    std::string item;
    while (ss >> item) out.push_back(item);
    return out;
}

const std::map<std::string, std::string> & zh_pinyin_table() {
    static std::map<std::string, std::string> table;
    if (!table.empty()) return table;

    const std::string path = first_existing_path({
        "src/zh_pinyin.tsv",
        "../src/zh_pinyin.tsv",
        "../../src/zh_pinyin.tsv",
    });
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        table[line.substr(0, tab)] = line.substr(tab + 1);
    }
    return table;
}

const std::map<std::string, std::vector<std::string>> & zh_pinyin_to_symbols() {
    static std::map<std::string, std::vector<std::string>> table;
    if (!table.empty()) return table;

    const std::string path = first_existing_path({
        "text/opencpop-strict.txt",
        "../text/opencpop-strict.txt",
        "../../text/opencpop-strict.txt",
    });
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        table[line.substr(0, tab)] = split_ws(line.substr(tab + 1));
    }
    return table;
}

std::vector<std::string> utf8_chars(const std::string & text) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        size_t n = 1;
        if ((c & 0xE0) == 0xC0) n = 2;
        else if ((c & 0xF0) == 0xE0) n = 3;
        else if ((c & 0xF8) == 0xF0) n = 4;
        if (i + n > text.size()) n = 1;
        chars.push_back(text.substr(i, n));
        i += n;
    }
    return chars;
}

std::string normalize_pinyin_base(std::string base) {
    if (base == "uei") return "ui";
    if (base == "iou") return "iu";
    if (base == "uen") return "un";
    return base;
}

std::string punctuation_symbol(const std::string & ch) {
    if (ch == " " || ch == "\t" || ch == "\r" || ch == "\n") return "SP";
    if (ch == "," || ch == "." || ch == "!" || ch == "?" || ch == "'" || ch == "-") return ch;
    if (ch == "\xEF\xBC\x8C" || ch == "\xE3\x80\x81") return ",";
    if (ch == "\xE3\x80\x82") return ".";
    if (ch == "\xEF\xBC\x81") return "!";
    if (ch == "\xEF\xBC\x9F") return "?";
    if (ch == "\xEF\xBC\x9B" || ch == "\xEF\xBC\x9A") return ",";
    return {};
}

TextFeatures intersperse_blank(const TextFeatures & in) {
    TextFeatures out;
    out.norm_text = in.norm_text;
    out.phones.reserve(in.phones.size() * 2 + 1);
    out.tones.reserve(in.tones.size() * 2 + 1);
    out.languages.reserve(in.languages.size() * 2 + 1);
    for (size_t i = 0; i < in.phones.size(); ++i) {
        out.phones.push_back(0);
        out.tones.push_back(0);
        out.languages.push_back(in.languages[i]);
        out.phones.push_back(in.phones[i]);
        out.tones.push_back(in.tones[i]);
        out.languages.push_back(in.languages[i]);
    }
    out.phones.push_back(0);
    out.tones.push_back(0);
    out.languages.push_back(in.languages.empty() ? 0 : in.languages.back());
    out.bert_tokens = in.bert_tokens;
    out.word2ph = in.word2ph;
    if (!out.word2ph.empty()) {
        for (int64_t & n : out.word2ph) n *= 2;
        out.word2ph[0] += 1;
    }
    return out;
}

std::string digits_to_chinese_number(const std::string & digits, bool is_year = false) {
    static const char * digit_cn[] = {
        "零", "一", "二", "三", "四", "五", "六", "七", "八", "九"
    };
    // Year: digit-by-digit reading
    if (is_year) {
        std::string out;
        for (char c : digits) out += digit_cn[c - '0'];
        return out;
    }
    int64_t n = 0;
    for (char c : digits) {
        if (n > 99999999) return digits; // too large, keep as-is
        n = n * 10 + (c - '0');
    }
    if (n == 0) return "零";

    std::string out;
    auto push_ge = [&](int64_t v) {
        if (v > 0) out += digit_cn[v];
    };
    auto push_shi = [&](int64_t v) {
        if (v > 0) {
            if (v > 1 || !out.empty()) out += digit_cn[v];
            out += "十";
        }
    };
    auto push_bai = [&](int64_t v) {
        if (v > 0) { out += digit_cn[v]; out += "百"; }
    };
    auto push_qian = [&](int64_t v) {
        if (v > 0) { out += digit_cn[v]; out += "千"; }
    };

    if (n >= 100000000) { int64_t yi = n / 100000000; push_ge(yi); out += "亿"; n %= 100000000; }
    if (n >= 10000) {
        int64_t wan = n / 10000;
        int64_t w_qian = wan / 1000; wan %= 1000;
        int64_t w_bai  = wan / 100;  wan %= 100;
        int64_t w_shi  = wan / 10;   wan %= 10;
        int64_t w_ge   = wan;
        bool has_wan = w_qian > 0 || w_bai > 0 || w_shi > 0 || w_ge > 0;
        if (has_wan) {
            if (out.empty() && w_qian == 0 && w_bai == 0 && w_shi == 0) { /* 20000  etc — ok */ }
            push_qian(w_qian);
            push_bai(w_bai);
            push_shi(w_shi);
            push_ge(w_ge);
            out += "万";
        } else if (out.empty()) {
            out += "万"; // n >= 10000 but the 万 part is 0, e.g. 10000
        }
        n %= 10000;
    }
    push_qian(n / 1000); n %= 1000;
    push_bai(n / 100);   n %= 100;
    push_shi(n / 10);    n %= 10;
    push_ge(n);

    return out;
}

// Convert date formats like 2025-06-13, 2025/06/13, 06-13 to 年月日 format
std::string normalize_date_formats(const std::string & text) {
    std::string result;
    result.reserve(text.size() + 32);
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        // Check for digit — start of a potential date
        if ((c & 0x80) == 0 && c >= '0' && c <= '9') {
            // Collect the number parts and separators
            std::vector<std::string> parts;
            std::string sep_type;
            size_t pos = i;
            std::string current;
            while (pos < text.size()) {
                const unsigned char pc = static_cast<unsigned char>(text[pos]);
                if ((pc & 0x80) == 0 && pc >= '0' && pc <= '9') {
                    current += static_cast<char>(pc);
                    ++pos;
                } else if ((pc == '-' || pc == '/' || pc == '.') && !current.empty()) {
                    char sc = static_cast<char>(pc);
                    if (sep_type.empty()) sep_type = std::string(1, sc);
                    else if (sep_type[0] != sc) break;
                    parts.push_back(current);
                    current.clear();
                    ++pos;
                } else {
                    break;
                }
            }
            if (!current.empty()) parts.push_back(current);

            // Check if this looks like a date
            bool converted = false;
            if (parts.size() == 3 && !sep_type.empty()) {
                // YYYY-MM-DD or MM-DD-YYYY etc
                if (parts[0].size() == 4) {
                    // YYYY-MM-DD
                    result += parts[0] + "年" + parts[1] + "月" + parts[2] + "日";
                    converted = true;
                } else if (parts[2].size() == 4) {
                    // MM-DD-YYYY
                    result += parts[2] + "年" + parts[0] + "月" + parts[1] + "日";
                    converted = true;
                }
            } else if (parts.size() == 2 && !sep_type.empty() && parts[0].size() <= 2 && parts[1].size() <= 2) {
                // MM-DD
                result += parts[0] + "月" + parts[1] + "日";
                converted = true;
            }

            if (converted) {
                i = pos;
            } else {
                result.append(text, i, pos - i);
                i = pos;
            }
        } else {
            size_t n = 1;
            if ((c & 0xE0) == 0xC0) n = 2;
            else if ((c & 0xF0) == 0xE0) n = 3;
            else if ((c & 0xF8) == 0xF0) n = 4;
            result.append(text, i, n);
            i += n;
        }
    }
    return result;
}

static bool is_char_nian(const std::string & text, size_t pos) {
    // UTF-8 "年" = E5 B9 B4
    return pos + 2 < text.size()
        && static_cast<unsigned char>(text[pos]) == 0xE5
        && static_cast<unsigned char>(text[pos + 1]) == 0xB9
        && static_cast<unsigned char>(text[pos + 2]) == 0xB4;
}

std::string normalize_chinese_numbers(const std::string & text) {
    std::string result;
    std::string digit_buf;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if ((c & 0x80) == 0 && c >= '0' && c <= '9') {
            digit_buf += static_cast<char>(c);
            ++i;
        } else {
            if (!digit_buf.empty()) {
                bool is_year = is_char_nian(text, i);
                result += digits_to_chinese_number(digit_buf, is_year);
                digit_buf.clear();
            }
            size_t n = 1;
            if ((c & 0xE0) == 0xC0) n = 2;
            else if ((c & 0xF0) == 0xE0) n = 3;
            else if ((c & 0xF8) == 0xF0) n = 4;
            result.append(text, i, n);
            i += n;
        }
    }
    if (!digit_buf.empty()) result += digits_to_chinese_number(digit_buf);
    return result;
}

std::string number_to_english_words(const std::string & digits) {
    static const char * ones[] = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"
    };
    static const char * tens[] = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
    };
    int n = 0;
    for (char c : digits) {
        if (n > 9999) return digits;
        n = n * 10 + (c - '0');
    }
    if (n < 20) return ones[n];
    if (n < 100) {
        if (n % 10 == 0) return tens[n / 10];
        return std::string(tens[n / 10]) + " " + ones[n % 10];
    }
    if (n < 1000) {
        std::string out = std::string(ones[n / 100]) + " hundred";
        if (n % 100 > 0) out += " " + number_to_english_words(std::to_string(n % 100));
        return out;
    }
    if (n < 10000) {
        std::string out = std::string(ones[n / 1000]) + " thousand";
        if (n % 1000 > 0) out += " " + number_to_english_words(std::to_string(n % 1000));
        return out;
    }
    return digits;
}

std::string normalize_english_numbers(const std::string & text) {
    std::string result;
    std::string digit_buf;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if ((c & 0x80) == 0 && c >= '0' && c <= '9') {
            digit_buf += static_cast<char>(c);
            ++i;
        } else {
            if (!digit_buf.empty()) {
                result += number_to_english_words(digit_buf);
                digit_buf.clear();
            }
            size_t n = 1;
            if ((c & 0xE0) == 0xC0) n = 2;
            else if ((c & 0xF0) == 0xE0) n = 3;
            else if ((c & 0xF8) == 0xF0) n = 4;
            result.append(text, i, n);
            i += n;
        }
    }
    if (!digit_buf.empty()) result += number_to_english_words(digit_buf);
    return result;
}

TextFeatures zh_text_to_sequence(const std::string & text) {
    const std::string normalized = normalize_chinese_numbers(normalize_date_formats(text));
    TextFeatures raw;
    raw.norm_text = normalized;
    add_phone(raw, "_", 0, "ZH");
    raw.word2ph.push_back(1);

    const auto & char_to_pinyin = zh_pinyin_table();
    const auto & pinyin_to_symbols = zh_pinyin_to_symbols();

    for (const auto & ch : utf8_chars(normalized)) {
        const std::string punct = punctuation_symbol(ch);
        if (!punct.empty()) {
            add_phone(raw, punct, 0, "ZH");
            raw.bert_tokens.push_back(punct);
            raw.word2ph.push_back(1);
            continue;
        }

        // ASCII letters → rough pinyin approximation for mixed Chinese-English text
        if (ch.size() == 1) {
            const char c = ch[0];
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
                static const char * letter_py[] = {
                    "ei1", "bi4", "si1", "di4", "yi4", "ai2 fu2", "ji2",
                    "ai2 qu3", "ai4", "jie2", "kai1", "ai2 er3", "ai2 mu2",
                    "en1", "ou1", "pi1", "kiu1", "a4", "ai2 si1", "ti4",
                    "you1", "wei2", "da2 bu4 liu2", "ai2 ke4 si1", "wai4", "zei2"
                };
                const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                const int idx = lower - 'a';
                if (idx >= 0 && idx < 26) {
                    std::string py = letter_py[idx];
                    int64_t tone = 5;
                    if (!py.empty() && py.back() >= '1' && py.back() <= '5') {
                        tone = py.back() - '0';
                        py.pop_back();
                    }
                    // Handle multi-syllable letters like "ai2 fu2" → just use first syllable
                    size_t space = py.find(' ');
                    if (space != std::string::npos) py = py.substr(0, space);
                    py = normalize_pinyin_base(py);
                    const auto sym_it = pinyin_to_symbols.find(py);
                    if (sym_it != pinyin_to_symbols.end()) {
                        int64_t phone_count = 0;
                        for (const std::string & symbol : sym_it->second) {
                            add_phone(raw, symbol, tone, "ZH");
                            ++phone_count;
                        }
                        raw.bert_tokens.push_back(ch);
                        raw.word2ph.push_back(phone_count);
                    }
                }
                continue;
            }
        }

        const auto py_it = char_to_pinyin.find(ch);
        if (py_it == char_to_pinyin.end()) {
            continue;
        }

        std::string py = py_it->second;
        int64_t tone = 5;
        if (!py.empty() && py.back() >= '1' && py.back() <= '5') {
            tone = py.back() - '0';
            py.pop_back();
        }
        py = normalize_pinyin_base(py);

        const auto sym_it = pinyin_to_symbols.find(py);
        if (sym_it == pinyin_to_symbols.end()) {
            add_phone(raw, "UNK", 0, "ZH");
            raw.bert_tokens.push_back(ch);
            raw.word2ph.push_back(1);
            continue;
        }

        int64_t phone_count = 0;
        for (const std::string & symbol : sym_it->second) {
            add_phone(raw, symbol, tone, "ZH");
            ++phone_count;
        }
        raw.bert_tokens.push_back(ch);
        raw.word2ph.push_back(phone_count);
    }

    add_phone(raw, "_", 0, "ZH");
    raw.word2ph.push_back(1);
    return intersperse_blank(raw);
}

std::vector<int64_t> distribute_phone(int64_t n_phone, int64_t n_word) {
    std::vector<int64_t> phones_per_word(static_cast<size_t>(n_word), 0);
    for (int64_t task = 0; task < n_phone; ++task) {
        auto it = std::min_element(phones_per_word.begin(), phones_per_word.end());
        *it += 1;
    }
    return phones_per_word;
}

} // namespace internal

namespace {

using namespace internal;

bool is_japanese_punctuation_char(const std::string & ch) {
    static const std::array<const char *, 12> marks = {
        ",", ".", "!", "?", "'", "-", "、", "。", "！", "？", "…", "・",
    };
    for (const char * mark : marks) {
        if (ch == mark) return true;
    }
    return false;
}

bool is_english_punctuation_token(const std::string & token) {
    return token == "!" || token == "?" || token == "…" || token == "," || token == "." || token == "'" || token == "-";
}

std::string normalize_english_text(std::string text) {
    static const std::vector<std::pair<std::string, std::string>> rep = {
        {"：", ","}, {"；", ","}, {"，", ","}, {"。", "."}, {"！", "!"}, {"？", "?"},
        {"\n", "."}, {"．", "."}, {"…", "..."}, {"···", "..."}, {"・・・", "..."},
        {"·", ","}, {"・", ","}, {"、", ","}, {"$", "."}, {"“", "'"}, {"”", "'"},
        {"\"", "'"}, {"‘", "'"}, {"’", "'"}, {"（", "'"}, {"）", "'"}, {"(", "'"},
        {")", "'"}, {"《", "'"}, {"》", "'"}, {"【", "'"}, {"】", "'"}, {"[", "'"},
        {"]", "'"}, {"—", "-"}, {"−", "-"}, {"～", "-"}, {"~", "-"}, {"「", "'"}, {"」", "'"},
    };
    for (const auto & item : rep) {
        size_t pos = 0;
        while ((pos = text.find(item.first, pos)) != std::string::npos) {
            text.replace(pos, item.first.size(), item.second);
            pos += item.second.size();
        }
    }
    return text;
}

std::string normalize_japanese_text(std::string text) {
    return normalize_english_text(std::move(text));
}

std::string post_replace_en_phone(std::string ph) {
    static const std::map<std::string, std::string> rep = {
        {"：", ","}, {"；", ","}, {"，", ","}, {"。", "."}, {"！", "!"}, {"？", "?"},
        {"\n", "."}, {"·", ","}, {"、", ","}, {"…", "..."}, {"···", "..."}, {"・・・", "..."},
        {"v", "V"},
    };
    const auto it = rep.find(ph);
    if (it != rep.end()) ph = it->second;
    if (symbol_ids().count(ph)) return ph;
    return "UNK";
}

std::pair<std::string, int64_t> refine_en_phone(std::string phn) {
    int64_t tone = 3;
    if (!phn.empty() && std::isdigit(static_cast<unsigned char>(phn.back()))) {
        tone = phn.back() - '0' + 1;
        phn.pop_back();
    }
    for (char & c : phn) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return {phn, tone};
}

const std::map<std::string, std::vector<std::vector<std::string>>> & cmu_dict() {
    static std::map<std::string, std::vector<std::vector<std::string>>> dict;
    if (!dict.empty()) return dict;

    const std::string path = first_existing_path({
        "text/cmudict.rep",
        "../text/cmudict.rep",
        "../../text/cmudict.rep",
    });
    std::ifstream in(path);
    std::string line;
    int64_t line_index = 0;
    while (std::getline(in, line)) {
        ++line_index;
        if (line_index < 49) continue;
        if (line.empty()) continue;
        const size_t split = line.find("  ");
        if (split == std::string::npos) continue;
        std::string word = line.substr(0, split);
        std::string pron = line.substr(split + 2);
        if (!pron.empty() && pron.back() == '\r') pron.pop_back();

        std::vector<std::vector<std::string>> syllables;
        std::stringstream ss(pron);
        std::string syllable;
        while (std::getline(ss, syllable, '-')) {
            if (!syllable.empty() && syllable.front() == ' ') syllable.erase(syllable.begin());
            std::vector<std::string> phones = split_ws(syllable);
            if (!phones.empty()) syllables.push_back(std::move(phones));
        }
        if (!syllables.empty()) dict[word] = std::move(syllables);
    }
    return dict;
}

std::vector<std::string> rough_english_word_to_phones(const std::string & word);

std::pair<std::vector<std::string>, std::vector<int64_t>> english_word_phones(const std::string & word) {
    std::vector<std::string> phones;
    std::vector<int64_t> tones;
    std::string upper = word;
    for (char & c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    const auto & dict = cmu_dict();
    const auto it = dict.find(upper);
    if (it != dict.end()) {
        for (const auto & syllable : it->second) {
            for (const std::string & phn : syllable) {
                auto refined = refine_en_phone(phn);
                phones.push_back(post_replace_en_phone(refined.first));
                tones.push_back(refined.second);
            }
        }
        return {phones, tones};
    }

    // Handle all-digit sequences: map each digit to its English word
    bool all_digits = !word.empty();
    for (char c : word) {
        if (!std::isdigit(static_cast<unsigned char>(c))) { all_digits = false; break; }
    }
    if (all_digits) {
        static const std::map<char, const char *> digit_word = {
            {'0', "zero"}, {'1', "one"}, {'2', "two"}, {'3', "three"}, {'4', "four"},
            {'5', "five"}, {'6', "six"}, {'7', "seven"}, {'8', "eight"}, {'9', "nine"},
        };
        for (char c : word) {
            const auto dw = digit_word.find(c);
            if (dw != digit_word.end()) {
                auto result = english_word_phones(dw->second);
                phones.insert(phones.end(), result.first.begin(), result.first.end());
                tones.insert(tones.end(), result.second.begin(), result.second.end());
            }
        }
        return {phones, tones};
    }

    for (const auto & ph : rough_english_word_to_phones(word)) {
        phones.push_back(post_replace_en_phone(ph));
        tones.push_back(0);
    }
    return {phones, tones};
}

std::vector<std::vector<std::string>> english_text_to_words(const std::vector<std::string> & pieces) {
    std::vector<std::vector<std::string>> words;
    for (size_t idx = 0; idx < pieces.size(); ++idx) {
        const std::string & piece = pieces[idx];
        if (!piece.empty() && piece[0] == '\xe2' && piece.size() >= 3 && piece.substr(0, 3) == "\xe2\x96\x81") {
            words.push_back({piece.substr(3)});
            continue;
        }
        if (is_english_punctuation_token(piece)) {
            if (idx == pieces.size() - 1) {
                words.push_back({piece});
            } else if ((idx + 1 < pieces.size())
                && !(pieces[idx + 1].size() >= 3 && pieces[idx + 1].substr(0, 3) == "\xe2\x96\x81")
                && !is_english_punctuation_token(pieces[idx + 1])) {
                if (words.empty()) words.push_back({});
                words.back().push_back(piece);
            } else {
                words.push_back({piece});
            }
        } else {
            if (words.empty()) words.push_back({});
            words.back().push_back(piece);
        }
    }
    return words;
}

TextFeatures jp_text_to_sequence_fallback(const std::string & text) {
    TextFeatures raw;
    const std::string norm = normalize_japanese_text(text);
    raw.norm_text = norm;
    add_phone(raw, "_", 0, "JP");
    raw.word2ph.push_back(1);

    std::vector<std::string> segments;
    std::string current;
    for (const auto & ch : utf8_chars(norm)) {
        if (is_japanese_punctuation_char(ch) || is_space(ch.empty() ? '\0' : ch[0])) {
            if (!current.empty()) {
                segments.push_back(current);
                current.clear();
            }
            if (!ch.empty() && !is_space(ch[0])) segments.push_back(ch);
        } else {
            current += ch;
        }
    }
    if (!current.empty()) segments.push_back(current);

    for (const std::string & segment : segments) {
        if (segment.size() == 1 && is_japanese_punctuation_char(segment)) {
            add_phone(raw, segment, 0, "JP");
            raw.bert_tokens.push_back(segment);
            raw.word2ph.push_back(1);
            continue;
        }

        const auto chars = utf8_chars(segment);
        int64_t phone_count = 0;
        for (const auto & ch : chars) {
            if (is_japanese_punctuation_char(ch)) add_phone(raw, ch, 0, "JP");
            else add_phone(raw, "UNK", 0, "JP");
            raw.bert_tokens.push_back(ch);
            ++phone_count;
        }
        const auto distributed = distribute_phone(phone_count, static_cast<int64_t>(chars.size()));
        for (int64_t n : distributed) raw.word2ph.push_back(n);
    }

    add_phone(raw, "_", 0, "JP");
    raw.word2ph.push_back(1);
    return intersperse_blank(raw);
}

} // namespace

TextFeatures jp_text_to_sequence(const std::string & text) {
    if (openjtalk_available()) {
        return jp_text_to_sequence_openjtalk(text);
    }
    return jp_text_to_sequence_fallback(text);
}

namespace {

TextFeatures en_text_to_sequence(const std::string & text) {
    SentencePieceTokenizer sp;
    const std::string spm_path = first_existing_path({
        "bert/deberta-v3-large/spm.model",
        "../bert/deberta-v3-large/spm.model",
        "../../bert/deberta-v3-large/spm.model",
    });
    sp.load(spm_path);

    const std::string norm = normalize_english_numbers(normalize_english_text(text));
    const auto pieces = sp.encode_pieces(norm);
    const auto words = english_text_to_words(pieces);

    TextFeatures raw;
    std::vector<std::string> phone_symbols;
    std::vector<int64_t> tone_values;
    std::vector<int64_t> phone_len;

    for (const auto & word : words) {
        std::vector<std::string> temp_phones;
        std::vector<int64_t> temp_tones;
        std::vector<std::string> parts = word;
        if (parts.size() > 1) {
            bool has_apostrophe = false;
            for (const auto & part : parts) {
                if (part == "'") has_apostrophe = true;
            }
            if (has_apostrophe) {
                std::string joined = parts[0];
                for (size_t i = 1; i < parts.size(); ++i) joined += parts[i];
                parts = {joined};
            }
        }

        for (const std::string & part : parts) {
            if (is_english_punctuation_token(part)) {
                temp_phones.push_back(part);
                temp_tones.push_back(0);
                continue;
            }
            auto result = english_word_phones(part);
            temp_phones.insert(temp_phones.end(), result.first.begin(), result.first.end());
            temp_tones.insert(temp_tones.end(), result.second.begin(), result.second.end());
        }
        phone_symbols.insert(phone_symbols.end(), temp_phones.begin(), temp_phones.end());
        tone_values.insert(tone_values.end(), temp_tones.begin(), temp_tones.end());
        phone_len.push_back(static_cast<int64_t>(temp_phones.size()));
    }

    std::vector<int64_t> word2ph;
    for (size_t i = 0; i < words.size(); ++i) {
        const auto distributed = distribute_phone(
            phone_len[i],
            static_cast<int64_t>(words[i].size()));
        word2ph.insert(word2ph.end(), distributed.begin(), distributed.end());
    }

    TextFeatures padded;
    padded.norm_text = norm;
    add_phone(padded, "_", 0, "EN");
    padded.word2ph.push_back(1);
    for (size_t i = 0; i < phone_symbols.size(); ++i) {
        add_phone(padded, phone_symbols[i], tone_values[i], "EN");
    }
    add_phone(padded, "_", 0, "EN");
    for (int64_t n : word2ph) padded.word2ph.push_back(n);
    padded.word2ph.push_back(1);
    return intersperse_blank(padded);
}

std::vector<std::string> rough_english_word_to_phones(const std::string & word) {
    static const std::map<char, std::vector<std::string>> table = {
        {'a', {"ae"}}, {'b', {"b"}}, {'c', {"k"}}, {'d', {"d"}}, {'e', {"eh"}},
        {'f', {"f"}}, {'g', {"g"}}, {'h', {"hh"}}, {'i', {"ih"}}, {'j', {"jh"}},
        {'k', {"k"}}, {'l', {"l"}}, {'m', {"m"}}, {'n', {"n"}}, {'o', {"ow"}},
        {'p', {"p"}}, {'q', {"k"}}, {'r', {"r"}}, {'s', {"s"}}, {'t', {"t"}},
        {'u', {"uw"}}, {'v', {"V"}}, {'w', {"w"}}, {'x', {"k", "s"}},
        {'y', {"y"}}, {'z', {"z"}},
    };
    std::vector<std::string> phones;
    for (char c : word) {
        const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        const auto it = table.find(lower);
        if (it != table.end()) {
            phones.insert(phones.end(), it->second.begin(), it->second.end());
        }
    }
    return phones;
}

Tensor sequence_mask(const std::vector<int64_t> & lengths, int64_t max_length) {
    Tensor mask({static_cast<int64_t>(lengths.size()), 1, max_length});
    for (int64_t b = 0; b < static_cast<int64_t>(lengths.size()); ++b) {
        for (int64_t t = 0; t < max_length; ++t) {
            mask(b, 0, t) = t < lengths[b] ? 1.0f : 0.0f;
        }
    }
    return mask;
}

Tensor generate_path(const Tensor & duration, const Tensor & mask) {
    const int64_t batch = mask.shape[0];
    const int64_t ty = mask.shape[2];
    const int64_t tx = mask.shape[3];
    Tensor path({batch, 1, ty, tx});

    for (int64_t b = 0; b < batch; ++b) {
        int64_t prev = 0;
        for (int64_t x = 0; x < tx; ++x) {
            int64_t next = prev + static_cast<int64_t>(duration.data[(b * tx) + x]);
            next = std::max<int64_t>(prev, std::min<int64_t>(next, ty));
            for (int64_t y = prev; y < next; ++y) {
                const int64_t idx = ((b * ty + y) * tx) + x;
                path.data[idx] = mask.data[idx] > 0.0f ? 1.0f : 0.0f;
            }
            prev = next;
        }
    }
    return path;
}

Tensor matmul_attn(const Tensor & attn, const Tensor & value) {
    const int64_t batch = value.shape[0];
    const int64_t channels = value.shape[1];
    const int64_t tx = value.shape[2];
    const int64_t ty = attn.shape[2];
    Tensor out({batch, channels, ty});

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t y = 0; y < ty; ++y) {
                float sum = 0.0f;
                for (int64_t x = 0; x < tx; ++x) {
                    const float a = attn.data[((b * ty + y) * tx) + x];
                    sum += a * value(b, c, x);
                }
                out(b, c, y) = sum;
            }
        }
    }
    return out;
}

#ifdef BV2_WITH_ONNXRUNTIME
void configure_execution_provider(Ort::SessionOptions & opts, const SynthesisOptions & options) {
    if (!options.use_cuda) return;
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = options.cuda_device_id;
    opts.AppendExecutionProvider_CUDA(cuda_options);
}

#ifdef _WIN32
std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) return {};
    const int count = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (count <= 0) throw std::runtime_error("failed to convert path to UTF-16: " + s);
    std::wstring out(static_cast<size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), count);
    return out;
}

Ort::Session load_session(Ort::Env & env, Ort::SessionOptions & opts, const std::string & path) {
    const std::wstring wide = utf8_to_wide(path);
    return Ort::Session(env, wide.c_str(), opts);
}
#else
Ort::Session load_session(Ort::Env & env, Ort::SessionOptions & opts, const std::string & path) {
    return Ort::Session(env, path.c_str(), opts);
}
#endif

struct OrtBundle {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "bert-vits2-cpp"};
    Ort::SessionOptions opts;
    Ort::Session enc{nullptr};
    Ort::Session emb{nullptr};
    Ort::Session dp{nullptr};
    Ort::Session sdp{nullptr};
    Ort::Session flow{nullptr};
    Ort::Session dec{nullptr};

    OrtBundle(const ModelPaths & paths, const SynthesisOptions & options) {
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetLogSeverityLevel(3);
        configure_execution_provider(opts, options);
        enc = load_session(env, opts, paths.enc);
        emb = load_session(env, opts, paths.emb);
        dp = load_session(env, opts, paths.dp);
        sdp = load_session(env, opts, paths.sdp);
        flow = load_session(env, opts, paths.flow);
        dec = load_session(env, opts, paths.dec);
    }
};

std::string model_cache_key(const ModelPaths & paths, const SynthesisOptions & options) {
    std::ostringstream ss;
    ss << paths.enc << '|'
       << paths.emb << '|'
       << paths.dp << '|'
       << paths.sdp << '|'
       << paths.flow << '|'
       << paths.dec << '|'
       << (options.use_cuda ? "cuda" : "cpu") << '|'
       << options.cuda_device_id;
    return ss.str();
}

std::shared_ptr<OrtBundle> cached_ort_bundle(const ModelPaths & paths, const SynthesisOptions & options) {
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<OrtBundle>> cache;
    const std::string key = model_cache_key(paths, options);
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto created = std::make_shared<OrtBundle>(paths, options);
    cache[key] = created;
    return created;
}

struct BertSessionCache {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "bert-vits2-bert-cache"};
    Ort::SessionOptions opts;
    Ort::Session session{nullptr};

    BertSessionCache(const std::string & path, const SynthesisOptions & options) {
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetLogSeverityLevel(3);
        configure_execution_provider(opts, options);
        session = load_session(env, opts, path);
    }
};

std::string bert_cache_key(const std::string & path, const SynthesisOptions & options) {
    std::ostringstream ss;
    ss << path << '|'
       << (options.use_cuda ? "cuda" : "cpu") << '|'
       << options.cuda_device_id;
    return ss.str();
}

std::shared_ptr<BertSessionCache> cached_bert_session(const std::string & path, const SynthesisOptions & options) {
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<BertSessionCache>> cache;
    const std::string key = bert_cache_key(path, options);
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto created = std::make_shared<BertSessionCache>(path, options);
    cache[key] = created;
    return created;
}

template <typename T>
Ort::Value make_ort_tensor(std::vector<T> & data, const std::vector<int64_t> & shape) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<T>(mem, data.data(), data.size(), shape.data(), shape.size());
}

Tensor tensor_from_ort(Ort::Value & value) {
    auto info = value.GetTensorTypeAndShapeInfo();
    Tensor out;
    out.shape = info.GetShape();
    const size_t count = info.GetElementCount();
    const float * ptr = value.GetTensorData<float>();
    out.data.assign(ptr, ptr + count);
    return out;
}

std::vector<Ort::Value> run_session(
    Ort::Session & session,
    const std::vector<const char *> & input_names,
    std::vector<Ort::Value> inputs,
    const std::vector<const char *> & output_names) {
    return session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        inputs.data(),
        inputs.size(),
        output_names.data(),
        output_names.size());
}
#endif

template <typename T>
void write_le(std::ofstream & out, T value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

} // namespace

Tensor::Tensor(std::vector<int64_t> s, float value) : shape(std::move(s)) {
    data.resize(static_cast<size_t>(size()), value);
}

int64_t Tensor::size() const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

float & Tensor::operator()(int64_t a, int64_t b, int64_t c) {
    return data[static_cast<size_t>((a * shape[1] + b) * shape[2] + c)];
}

const float & Tensor::operator()(int64_t a, int64_t b, int64_t c) const {
    return data[static_cast<size_t>((a * shape[1] + b) * shape[2] + c)];
}

std::vector<std::pair<std::string, std::string>> split_text_by_language(
    const std::string & text, const std::string & primary_lang) {
    std::vector<std::pair<std::string, std::string>> spans;
    if (text.empty()) return spans;

    const auto chars = utf8_chars(text);
    std::string current_span;
    std::string current_lang;

    for (const auto & ch : chars) {
        const unsigned char c0 = static_cast<unsigned char>(ch[0]);
        std::string char_lang;
        if (ch.size() >= 3 && c0 == 0xE3) {
            const unsigned char c1 = static_cast<unsigned char>(ch[1]);
            // Japanese: Hiragana U+3040–U+309F, Katakana U+30A0–U+30FF
            if ((c1 == 0x81 && c0 == 0xE3) || (c1 == 0x82) || (c1 == 0x83 && c0 == 0xE3)) {
                const uint32_t cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (static_cast<unsigned char>(ch[2]) & 0x3F);
                if ((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF))
                    char_lang = "JP";
            }
        }
        if (char_lang.empty() && ch.size() >= 3 && c0 >= 0xE4 && c0 <= 0xE9) {
            // CJK Unified range roughly U+4E00–U+9FFF
            const uint32_t cp = ((c0 & 0x0F) << 12) | ((static_cast<unsigned char>(ch[1]) & 0x3F) << 6) | (static_cast<unsigned char>(ch[2]) & 0x3F);
            if (cp >= 0x4E00 && cp <= 0x9FFF)
                char_lang = (primary_lang == "JP") ? "JP" : "ZH";
        }
        if (char_lang.empty() && ch.size() == 1) {
            const char c = ch[0];
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
                char_lang = "EN";
            else if (c >= '0' && c <= '9')
                char_lang = current_lang.empty() ? (primary_lang.empty() ? "EN" : primary_lang) : current_lang;
        }

        if (char_lang.empty()) {
            // Punctuation / space — keep with current span
            char_lang = current_lang.empty() ? "ZH" : current_lang;
        }

        if (current_lang.empty()) {
            current_lang = char_lang;
        }

        if (char_lang != current_lang) {
            spans.push_back({current_span, current_lang});
            current_span.clear();
            current_lang = char_lang;
        }

        current_span += ch;
    }

    if (!current_span.empty()) {
        spans.push_back({current_span, current_lang});
    }

    return spans;
}

TextFeatures text_to_sequence_mixed(const std::string & text) {
    auto spans = split_text_by_language(text);
    if (spans.empty()) {
        return text_to_sequence(text, "ZH");
    }
    if (spans.size() == 1) {
        return text_to_sequence(text, spans[0].second);
    }

    // Process each span with its language frontend
    std::vector<TextFeatures> span_features;
    span_features.reserve(spans.size());
    for (const auto & [span_text, lang] : spans) {
        span_features.push_back(text_to_sequence(span_text, lang));
    }

    // Merge phones, tones, languages (simple concatenation)
    TextFeatures merged;
    size_t total_phones = 0;
    for (const auto & f : span_features) {
        total_phones += f.phones.size();
    }
    merged.phones.reserve(total_phones);
    merged.tones.reserve(total_phones);
    merged.languages.reserve(total_phones);

    for (const auto & f : span_features) {
        merged.phones.insert(merged.phones.end(), f.phones.begin(), f.phones.end());
        merged.tones.insert(merged.tones.end(), f.tones.begin(), f.tones.end());
        merged.languages.insert(merged.languages.end(), f.languages.begin(), f.languages.end());
    }

    // bert_tokens and word2ph are NOT merged — mixed BERT is handled per-span
    // in synthesize_features, not via the merged TextFeatures.

    return merged;
}

TextFeatures text_to_sequence(const std::string & text, const std::string & language) {
    if (language == "MIX") return text_to_sequence_mixed(text);
    if (language == "ZH") return zh_text_to_sequence(text);
    if (language == "JP") return jp_text_to_sequence(text);
    if (language == "EN") return en_text_to_sequence(text);

    TextFeatures out;
    add_phone(out, "_", 0, language);
    add_phone(out, "_", 0, language);
    out.word2ph = {1, 1};
    return intersperse_blank(out);
}

TextFeatures parse_phone_ids(const std::string & csv, const std::string & language) {
    TextFeatures out;
    out.phones = parse_i64_csv(csv);
    out.tones.assign(out.phones.size(), tone_start(language));
    out.languages.assign(out.phones.size(), language_id(language));
    if (out.phones.empty()) {
        throw std::runtime_error("--phones cannot be empty");
    }
    return out;
}

TextFeatures parse_phone_ids(
    const std::string & phone_csv,
    const std::string & tone_csv,
    const std::string & language_csv,
    const std::string & default_language) {
    TextFeatures out = parse_phone_ids(phone_csv, default_language);
    if (!tone_csv.empty()) {
        out.tones = parse_i64_csv(tone_csv);
    }
    if (!language_csv.empty()) {
        out.languages = parse_i64_csv(language_csv);
    }
    if (out.phones.size() != out.tones.size() || out.phones.size() != out.languages.size()) {
        throw std::runtime_error("phones, tones, and languages must have the same length");
    }
    return out;
}

Tensor zeros_bert(int64_t phones) {
    return Tensor({phones, kBertDim}, 0.0f);
}

Tensor random_bert(int64_t phones, uint32_t seed) {
    Tensor t({phones, kBertDim});
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float & v : t.data) v = dist(rng);
    return t;
}

std::map<std::string, int64_t> load_vocab_map(const std::string & vocab_path) {
    std::map<std::string, int64_t> vocab;
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file) {
        throw std::runtime_error("failed to open BERT vocab: " + vocab_path);
    }
    std::string token;
    int64_t id = 0;
    while (std::getline(vocab_file, token)) {
        if (!token.empty() && token.back() == '\r') token.pop_back();
        vocab[token] = id++;
    }
    return vocab;
}

Tensor bert_feature_from_input_ids(
    const std::string & bert_onnx_path,
    const std::vector<int64_t> & input_ids,
    const std::vector<int64_t> & word2ph,
    int64_t phone_count,
    const SynthesisOptions & options,
    bool use_token_type_ids) {
#ifndef BV2_WITH_ONNXRUNTIME
    (void)bert_onnx_path; (void)input_ids; (void)word2ph; (void)phone_count; (void)options; (void)use_token_type_ids;
    throw std::runtime_error("this binary was built without ONNX Runtime support");
#else
    if (word2ph.empty()) return zeros_bert(phone_count);
    if (static_cast<int64_t>(word2ph.size()) != static_cast<int64_t>(input_ids.size())) {
        throw std::runtime_error("BERT word2ph length does not match token count");
    }

    const int64_t tokens = static_cast<int64_t>(input_ids.size());
    std::vector<int64_t> ids = input_ids;
    std::vector<int64_t> attention_mask(static_cast<size_t>(tokens), 1);
    std::vector<int64_t> token_type_ids(static_cast<size_t>(tokens), 0);
    const std::vector<int64_t> shape = {1, tokens};

    auto cached = cached_bert_session(bert_onnx_path, options);

    std::vector<Ort::Value> inputs;
    std::vector<const char *> input_names;
    inputs.emplace_back(make_ort_tensor(ids, shape));
    input_names.push_back("input_ids");
    inputs.emplace_back(make_ort_tensor(attention_mask, shape));
    input_names.push_back("attention_mask");
    if (use_token_type_ids) {
        inputs.emplace_back(make_ort_tensor(token_type_ids, shape));
        input_names.push_back("token_type_ids");
    }
    auto outputs = run_session(
        cached->session,
        input_names,
        std::move(inputs),
        {"hidden"});
    Tensor hidden = tensor_from_ort(outputs[0]);

    Tensor phone_bert({phone_count, kBertDim});
    int64_t out_t = 0;
    for (int64_t tok = 0; tok < tokens; ++tok) {
        const int64_t repeats = word2ph[static_cast<size_t>(tok)];
        for (int64_t r = 0; r < repeats; ++r) {
            if (out_t >= phone_bert.shape[0]) break;
            for (int64_t d = 0; d < kBertDim; ++d) {
                phone_bert.data[static_cast<size_t>(out_t * kBertDim + d)] =
                    hidden.data[static_cast<size_t>(tok * kBertDim + d)];
            }
            ++out_t;
        }
    }
    if (out_t != phone_bert.shape[0]) {
        throw std::runtime_error("BERT repeat length does not match phone length");
    }
    return phone_bert;
#endif
}

Tensor vocab_bert(
    const std::string & bert_onnx_path,
    const std::string & vocab_path,
    const TextFeatures & text,
    const SynthesisOptions & options,
    bool use_token_type_ids) {
#ifndef BV2_WITH_ONNXRUNTIME
    (void)bert_onnx_path; (void)vocab_path; (void)text; (void)options; (void)use_token_type_ids;
    throw std::runtime_error("this binary was built without ONNX Runtime support");
#else
    if (text.word2ph.empty() || text.bert_tokens.empty()) {
        return zeros_bert(static_cast<int64_t>(text.phones.size()));
    }
    if (text.word2ph.size() != text.bert_tokens.size() + 2) {
        throw std::runtime_error("BERT word2ph shape does not match tokenized text");
    }

    const std::string onnx_parent = "../" + bert_onnx_path;
    const std::string onnx_grandparent = "../../" + bert_onnx_path;
    const std::string vocab_parent = "../" + vocab_path;
    const std::string vocab_grandparent = "../../" + vocab_path;
    const std::string resolved_onnx = first_existing_path({
        bert_onnx_path.c_str(),
        onnx_parent.c_str(),
        onnx_grandparent.c_str(),
    });
    const std::string resolved_vocab = first_existing_path({
        vocab_path.c_str(),
        vocab_parent.c_str(),
        vocab_grandparent.c_str(),
    });
    const auto vocab = load_vocab_map(resolved_vocab);
    auto token_id = [&](const std::string & value) -> int64_t {
        const auto it = vocab.find(value);
        if (it != vocab.end()) return it->second;
        const auto unk = vocab.find("[UNK]");
        return unk == vocab.end() ? 100 : unk->second;
    };

    std::vector<int64_t> input_ids;
    input_ids.reserve(text.bert_tokens.size() + 2);
    input_ids.push_back(token_id("[CLS]"));
    for (const auto & value : text.bert_tokens) input_ids.push_back(token_id(value));
    input_ids.push_back(token_id("[SEP]"));
    return bert_feature_from_input_ids(
        resolved_onnx,
        input_ids,
        text.word2ph,
        static_cast<int64_t>(text.phones.size()),
        options,
        use_token_type_ids);
#endif
}

Tensor chinese_bert(
    const std::string & bert_onnx_path,
    const std::string & vocab_path,
    const TextFeatures & text,
    const SynthesisOptions & options) {
    return vocab_bert(bert_onnx_path, vocab_path, text, options, true);
}

Tensor japanese_bert(
    const std::string & bert_onnx_path,
    const std::string & vocab_path,
    const TextFeatures & text,
    const SynthesisOptions & options) {
    return vocab_bert(bert_onnx_path, vocab_path, text, options, false);
}

Tensor english_bert(
    const std::string & bert_onnx_path,
    const std::string & spm_model_path,
    const std::string & norm_text,
    const TextFeatures & text,
    const SynthesisOptions & options) {
#ifndef BV2_WITH_ONNXRUNTIME
    (void)bert_onnx_path; (void)spm_model_path; (void)norm_text; (void)text; (void)options;
    throw std::runtime_error("this binary was built without ONNX Runtime support");
#else
    if (text.word2ph.empty()) {
        return zeros_bert(static_cast<int64_t>(text.phones.size()));
    }

    const std::string onnx_parent = "../" + bert_onnx_path;
    const std::string onnx_grandparent = "../../" + bert_onnx_path;
    const std::string spm_parent = "../" + spm_model_path;
    const std::string spm_grandparent = "../../" + spm_model_path;
    const std::string resolved_onnx = first_existing_path({
        bert_onnx_path.c_str(),
        onnx_parent.c_str(),
        onnx_grandparent.c_str(),
    });
    const std::string resolved_spm = first_existing_path({
        spm_model_path.c_str(),
        spm_parent.c_str(),
        spm_grandparent.c_str(),
    });

    SentencePieceTokenizer sp;
    sp.load(resolved_spm);
    std::vector<int64_t> input_ids = sp.encode(normalize_english_numbers(normalize_english_text(norm_text)));
    input_ids.insert(input_ids.begin(), 1);
    input_ids.push_back(2);

    if (input_ids.size() != text.word2ph.size()) {
        throw std::runtime_error("English BERT token count does not match word2ph length");
    }
    return bert_feature_from_input_ids(
        resolved_onnx,
        input_ids,
        text.word2ph,
        static_cast<int64_t>(text.phones.size()),
        options,
        false);
#endif
}

Tensor zeros_emotion() {
    return Tensor({kEmotionDim, 1}, 0.0f);
}

Tensor random_emotion(uint32_t seed) {
    Tensor t({kEmotionDim, 1});
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float & v : t.data) v = dist(rng);
    return t;
}

Tensor load_float_bin(const std::string & path, std::vector<int64_t> shape) {
    Tensor t(std::move(shape));
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open float bin: " + path);
    in.read(reinterpret_cast<char *>(t.data.data()), static_cast<std::streamsize>(t.data.size() * sizeof(float)));
    if (in.gcount() != static_cast<std::streamsize>(t.data.size() * sizeof(float))) {
        throw std::runtime_error("float bin has unexpected size: " + path);
    }
    return t;
}

void preload_synthesis_model(
    const ModelPaths & paths,
    const SynthesisOptions & options) {
#ifndef BV2_WITH_ONNXRUNTIME
    (void)paths; (void)options;
    throw std::runtime_error("this binary was built without ONNX Runtime support");
#else
    (void)cached_ort_bundle(paths, options);
#endif
}

std::vector<float> synthesize(
    const ModelPaths & paths,
    const TextFeatures & text,
    const SynthesisOptions & options,
    const Tensor * bert_zh,
    const Tensor * bert_jp,
    const Tensor * bert_en,
    const Tensor * emotion) {
#ifndef BV2_WITH_ONNXRUNTIME
    (void)paths; (void)text; (void)options; (void)bert_zh; (void)bert_jp; (void)bert_en; (void)emotion;
    throw std::runtime_error("this binary was built without ONNX Runtime support");
#else
    if (text.phones.size() != text.tones.size() || text.phones.size() != text.languages.size()) {
        throw std::runtime_error("text feature lengths do not match");
    }

    const int64_t tx = static_cast<int64_t>(text.phones.size());
    auto ort = cached_ort_bundle(paths, options);

    std::vector<int64_t> x = text.phones;
    std::vector<int64_t> tone = text.tones;
    std::vector<int64_t> language = text.languages;
    std::vector<int64_t> sid = {options.speaker_id};
    const std::vector<int64_t> seq_shape = {1, tx};
    const std::vector<int64_t> sid_shape = {1};

    auto sid_tensor = make_ort_tensor(sid, sid_shape);
    std::vector<Ort::Value> emb_inputs;
    emb_inputs.emplace_back(std::move(sid_tensor));
    auto emb_out = run_session(ort->emb, {"sid"}, std::move(emb_inputs), {"g"});
    Tensor g2 = tensor_from_ort(emb_out[0]);
    Tensor g({g2.shape[0], g2.shape[1], 1});
    std::copy(g2.data.begin(), g2.data.end(), g.data.begin());

    Tensor zh = bert_zh ? *bert_zh : zeros_bert(tx);
    Tensor jp = bert_jp ? *bert_jp : zeros_bert(tx);
    Tensor en = bert_en ? *bert_en : zeros_bert(tx);
    Tensor emo = emotion ? *emotion : zeros_emotion();

    auto x_tensor = make_ort_tensor(x, seq_shape);
    auto tone_tensor = make_ort_tensor(tone, seq_shape);
    auto language_tensor = make_ort_tensor(language, seq_shape);
    auto zh_tensor = make_ort_tensor(zh.data, zh.shape);
    auto jp_tensor = make_ort_tensor(jp.data, jp.shape);
    auto en_tensor = make_ort_tensor(en.data, en.shape);
    auto g_tensor = make_ort_tensor(g.data, g.shape);

    std::vector<Ort::Value> enc_inputs;
    enc_inputs.emplace_back(std::move(x_tensor));
    enc_inputs.emplace_back(std::move(tone_tensor));
    enc_inputs.emplace_back(std::move(language_tensor));
    enc_inputs.emplace_back(std::move(zh_tensor));
    enc_inputs.emplace_back(std::move(jp_tensor));
    enc_inputs.emplace_back(std::move(en_tensor));
    if (options.use_emotion) {
        auto emo_tensor = make_ort_tensor(emo.data, emo.shape);
        enc_inputs.emplace_back(std::move(emo_tensor));
    }
    enc_inputs.emplace_back(std::move(g_tensor));
    std::vector<const char *> enc_input_names = {"x", "t", "language", "bert_0", "bert_1", "bert_2"};
    if (options.use_emotion) {
        enc_input_names.push_back("emo");
    }
    enc_input_names.push_back("g");
    auto enc_out = run_session(
        ort->enc,
        enc_input_names,
        std::move(enc_inputs),
        {"xout", "m_p", "logs_p", "x_mask"});

    Tensor xout = tensor_from_ort(enc_out[0]);
    Tensor m_p = tensor_from_ort(enc_out[1]);
    Tensor logs_p = tensor_from_ort(enc_out[2]);
    Tensor x_mask = tensor_from_ort(enc_out[3]);

    std::mt19937 rng(options.seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Tensor zin({xout.shape[0], 2, xout.shape[2]});
    for (float & v : zin.data) v = dist(rng) * options.noise_scale_w;

    auto sdp_x = xout;
    auto sdp_mask = x_mask;
    auto sdp_zin = zin;
    auto sdp_g = g;
    std::vector<Ort::Value> sdp_inputs;
    sdp_inputs.emplace_back(make_ort_tensor(sdp_x.data, sdp_x.shape));
    sdp_inputs.emplace_back(make_ort_tensor(sdp_mask.data, sdp_mask.shape));
    sdp_inputs.emplace_back(make_ort_tensor(sdp_zin.data, sdp_zin.shape));
    sdp_inputs.emplace_back(make_ort_tensor(sdp_g.data, sdp_g.shape));
    auto sdp_out = run_session(
        ort->sdp,
        {"x", "x_mask", "zin", "g"},
        std::move(sdp_inputs),
        {"logw"});

    auto dp_x = xout;
    auto dp_mask = x_mask;
    auto dp_g = g;
    std::vector<Ort::Value> dp_inputs;
    dp_inputs.emplace_back(make_ort_tensor(dp_x.data, dp_x.shape));
    dp_inputs.emplace_back(make_ort_tensor(dp_mask.data, dp_mask.shape));
    dp_inputs.emplace_back(make_ort_tensor(dp_g.data, dp_g.shape));
    auto dp_out = run_session(
        ort->dp,
        {"x", "x_mask", "g"},
        std::move(dp_inputs),
        {"logw"});

    Tensor sdp_logw = tensor_from_ort(sdp_out[0]);
    Tensor dp_logw = tensor_from_ort(dp_out[0]);
    Tensor w_ceil(sdp_logw.shape);
    int64_t y_len = 0;
    for (size_t i = 0; i < w_ceil.data.size(); ++i) {
        const float logw = sdp_logw.data[i] * options.sdp_ratio + dp_logw.data[i] * (1.0f - options.sdp_ratio);
        const float w = std::exp(logw) * x_mask.data[i] * options.length_scale;
        const int64_t wc = static_cast<int64_t>(std::ceil(w));
        w_ceil.data[i] = static_cast<float>(wc);
        y_len += wc;
    }
    y_len = std::max<int64_t>(1, std::min<int64_t>(y_len, 100000));

    Tensor y_mask = sequence_mask({y_len}, y_len);
    Tensor attn_mask({1, 1, y_len, tx});
    for (int64_t y = 0; y < y_len; ++y) {
        for (int64_t xidx = 0; xidx < tx; ++xidx) {
            attn_mask.data[static_cast<size_t>(y * tx + xidx)] = y_mask(0, 0, y) * x_mask(0, 0, xidx);
        }
    }

    Tensor attn = generate_path(w_ceil, attn_mask);
    Tensor m_aligned = matmul_attn(attn, m_p);
    Tensor logs_aligned = matmul_attn(attn, logs_p);

    Tensor z_p(m_aligned.shape);
    for (size_t i = 0; i < z_p.data.size(); ++i) {
        z_p.data[i] = m_aligned.data[i] + dist(rng) * std::exp(logs_aligned.data[i]) * options.noise_scale;
    }

    auto flow_z = z_p;
    auto flow_mask = y_mask;
    auto flow_g = g;
    std::vector<Ort::Value> flow_inputs;
    flow_inputs.emplace_back(make_ort_tensor(flow_z.data, flow_z.shape));
    flow_inputs.emplace_back(make_ort_tensor(flow_mask.data, flow_mask.shape));
    flow_inputs.emplace_back(make_ort_tensor(flow_g.data, flow_g.shape));
    auto flow_out = run_session(
        ort->flow,
        {"z_p", "y_mask", "g"},
        std::move(flow_inputs),
        {"z"});
    Tensor z = tensor_from_ort(flow_out[0]);

    for (int64_t c = 0; c < z.shape[1]; ++c) {
        for (int64_t t = 0; t < z.shape[2]; ++t) {
            z(0, c, t) *= y_mask(0, 0, t);
        }
    }

    auto dec_z = z;
    auto dec_g = g;
    std::vector<Ort::Value> dec_inputs;
    dec_inputs.emplace_back(make_ort_tensor(dec_z.data, dec_z.shape));
    dec_inputs.emplace_back(make_ort_tensor(dec_g.data, dec_g.shape));
    auto dec_out = run_session(
        ort->dec,
        {"z_in", "g"},
        std::move(dec_inputs),
        {"o"});

    Tensor audio_tensor = tensor_from_ort(dec_out[0]);
    return audio_tensor.data;
#endif
}

void write_wav(const std::string & path, const std::vector<float> & audio, int sample_rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open output wav: " + path);

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate * channels * bits_per_sample / 8);
    const uint16_t block_align = static_cast<uint16_t>(channels * bits_per_sample / 8);
    const uint32_t data_size = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    write_le(out, riff_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_le<uint32_t>(out, 16);
    write_le<uint16_t>(out, 1);
    write_le(out, channels);
    write_le<uint32_t>(out, static_cast<uint32_t>(sample_rate));
    write_le(out, byte_rate);
    write_le(out, block_align);
    write_le(out, bits_per_sample);
    out.write("data", 4);
    write_le(out, data_size);

    for (float sample : audio) {
        const float clamped = std::max(-1.0f, std::min(1.0f, sample));
        const auto s = static_cast<int16_t>(std::lrint(clamped * 32767.0f));
        write_le(out, s);
    }
}

} // namespace bv2

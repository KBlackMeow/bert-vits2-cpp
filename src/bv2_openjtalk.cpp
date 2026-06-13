#include "bv2_openjtalk.h"
#include "bv2_text_internal.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef BV2_WITH_OPENJTALK
#include "njd.h"
#include "jpcommon.h"
#include "text2mecab.h"
#include "mecab2njd.h"
#include "njd2jpcommon.h"
#include "njd_set_pronunciation.h"
#include "njd_set_digit.h"
#include "njd_set_accent_phrase.h"
#include "njd_set_accent_type.h"
#include "njd_set_unvoiced_vowel.h"
#include "njd_set_long_vowel.h"
#include "mecab.h"
#endif

namespace bv2 {
namespace {

#ifdef BV2_WITH_OPENJTALK

struct JpNjdNode {
    std::string string;
    std::string pos;
    std::string orig;
    std::string read;
    std::string pron;
    int acc = 0;
    int mora_size = 0;
};

class OpenJtalkEngine {
public:
    static OpenJtalkEngine & instance() {
        static OpenJtalkEngine engine;
        return engine;
    }

    bool ready() const { return ready_; }

    std::vector<JpNjdNode> run_frontend(const std::string & text) {
        ensure_ready();
        char buff[8192] = {};
        text2mecab(buff, text.c_str());
        Mecab_analysis(&mecab_, buff);
        mecab2njd(&njd_, Mecab_get_feature(&mecab_), Mecab_get_size(&mecab_));
        njd_set_pronunciation(&njd_);
        njd_set_digit(&njd_);
        njd_set_accent_phrase(&njd_);
        njd_set_accent_type(&njd_);
        njd_set_unvoiced_vowel(&njd_);
        njd_set_long_vowel(&njd_);

        std::vector<JpNjdNode> nodes;
        for (NJDNode * node = njd_.head; node != nullptr; node = node->next) {
            JpNjdNode out;
            if (node->string) out.string = node->string;
            if (node->pos) out.pos = node->pos;
            if (node->orig) out.orig = node->orig;
            if (node->read) out.read = node->read;
            if (node->pron) out.pron = node->pron;
            out.acc = node->acc;
            out.mora_size = node->mora_size;
            nodes.push_back(std::move(out));
        }

        NJD_refresh(&njd_);
        Mecab_refresh(&mecab_);
        return nodes;
    }

    std::vector<std::string> make_label(const std::vector<JpNjdNode> & nodes) {
        ensure_ready();
        NJD_refresh(&njd_);
        for (const auto & feature : nodes) {
            auto * node = static_cast<NJDNode *>(std::calloc(1, sizeof(NJDNode)));
            NJDNode_initialize(node);
            NJDNode_set_string(node, feature.string.c_str());
            NJDNode_set_pos(node, feature.pos.c_str());
            NJDNode_set_pos_group1(node, "*");
            NJDNode_set_pos_group2(node, "*");
            NJDNode_set_pos_group3(node, "*");
            NJDNode_set_ctype(node, "*");
            NJDNode_set_cform(node, "*");
            NJDNode_set_orig(node, feature.orig.c_str());
            NJDNode_set_read(node, feature.read.c_str());
            NJDNode_set_pron(node, feature.pron.c_str());
            NJDNode_set_acc(node, feature.acc);
            NJDNode_set_mora_size(node, feature.mora_size);
            NJDNode_set_chain_rule(node, "*");
            NJDNode_set_chain_flag(node, 0);
            NJD_push_node(&njd_, node);
        }

        JPCommon_refresh(&jpcommon_);
        njd2jpcommon(&jpcommon_, &njd_);
        JPCommon_make_label(&jpcommon_);
        const int label_size = JPCommon_get_label_size(&jpcommon_);
        char ** label_feature = JPCommon_get_label_feature(&jpcommon_);

        std::vector<std::string> labels;
        labels.reserve(static_cast<size_t>(label_size));
        for (int i = 0; i < label_size; ++i) {
            if (label_feature[i] != nullptr) labels.emplace_back(label_feature[i]);
        }

        JPCommon_refresh(&jpcommon_);
        NJD_refresh(&njd_);
        return labels;
    }

private:
    OpenJtalkEngine() {
        Mecab_initialize(&mecab_);
        NJD_initialize(&njd_);
        JPCommon_initialize(&jpcommon_);
        ready_ = load_dictionary();
    }

    ~OpenJtalkEngine() {
        Mecab_clear(&mecab_);
        NJD_clear(&njd_);
        JPCommon_clear(&jpcommon_);
    }

    OpenJtalkEngine(const OpenJtalkEngine &) = delete;
    OpenJtalkEngine & operator=(const OpenJtalkEngine &) = delete;

    bool file_exists(const std::string & path) const {
        std::ifstream in(path, std::ios::binary);
        return static_cast<bool>(in);
    }

    std::string resolve_dictionary_dir() const {
        static const std::array<const char *, 6> candidates = {
            "third_party/open_jtalk_dic_utf_8-1.11",
            "../third_party/open_jtalk_dic_utf_8-1.11",
            "../../third_party/open_jtalk_dic_utf_8-1.11",
            "bert-vits2-cpp/third_party/open_jtalk_dic_utf_8-1.11",
            "../bert-vits2-cpp/third_party/open_jtalk_dic_utf_8-1.11",
            "../../bert-vits2-cpp/third_party/open_jtalk_dic_utf_8-1.11",
        };
        for (const char * candidate : candidates) {
            const std::string dic = std::string(candidate) + "/sys.dic";
            if (file_exists(dic)) return candidate;
        }
        return {};
    }

    bool load_dictionary() {
        const std::string dict_dir = resolve_dictionary_dir();
        if (dict_dir.empty()) return false;
        return Mecab_load(&mecab_, dict_dir.c_str()) == TRUE;
    }

    void ensure_ready() const {
        if (!ready_) {
            throw std::runtime_error("OpenJTalk dictionary was not found under third_party/open_jtalk_dic_utf_8-1.11");
        }
    }

    Mecab mecab_{};
    NJD njd_{};
    JPCommon jpcommon_{};
    bool ready_ = false;
};

bool is_jp_punctuation_token(const std::string & token) {
    return token == "!" || token == "?" || token == "…" || token == ","
        || token == "." || token == "'" || token == "-";
}

bool is_symbol_token(const std::string & ch) {
    static const std::array<const char *, 5> tokens = {"・", "、", "。", "？", "！"};
    for (const char * token : tokens) {
        if (ch == token) return true;
    }
    return false;
}

bool is_no_yomi_token(const std::string & ch) {
    static const std::array<const char *, 10> tokens = {
        "「", "」", "『", "』", "―", "（", "）", "［", "］", "[",
    };
    for (const char * token : tokens) {
        if (ch == token) return true;
    }
    return ch == "]";
}

bool is_japanese_codepoint(uint32_t cp) {
    if (cp == 0x3005) return true;
    if (cp >= 0x3040 && cp <= 0x309F) return true;
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0xFF11 && cp <= 0xFF19) return true;
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;
    if (cp >= 0xFF66 && cp <= 0xFF9D) return true;
    return false;
}

uint32_t utf8_to_codepoint(const std::string & ch) {
    const unsigned char b0 = static_cast<unsigned char>(ch[0]);
    if (b0 < 0x80) return b0;
    if ((b0 & 0xE0) == 0xC0 && ch.size() >= 2) {
        return ((b0 & 0x1F) << 6) | (static_cast<unsigned char>(ch[1]) & 0x3F);
    }
    if ((b0 & 0xF0) == 0xE0 && ch.size() >= 3) {
        return ((b0 & 0x0F) << 12)
            | ((static_cast<unsigned char>(ch[1]) & 0x3F) << 6)
            | (static_cast<unsigned char>(ch[2]) & 0x3F);
    }
    if ((b0 & 0xF8) == 0xF0 && ch.size() >= 4) {
        return ((b0 & 0x07) << 18)
            | ((static_cast<unsigned char>(ch[1]) & 0x3F) << 12)
            | ((static_cast<unsigned char>(ch[2]) & 0x3F) << 6)
            | (static_cast<unsigned char>(ch[3]) & 0x3F);
    }
    return b0;
}

bool is_mark_yomi(const std::string & yomi) {
    if (yomi.empty()) return false;
    for (const auto & ch : internal::utf8_chars(yomi)) {
        const uint32_t cp = utf8_to_codepoint(ch);
        if (std::isalnum(static_cast<unsigned char>(cp)) || is_japanese_codepoint(cp)) {
            return false;
        }
    }
    return true;
}

std::string hira_to_kata(std::string text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const unsigned char b0 = static_cast<unsigned char>(text[i]);
        if (b0 < 0x80) {
            out.push_back(text[i++]);
            continue;
        }
        size_t len = 1;
        if ((b0 & 0xE0) == 0xC0 && i + 1 < text.size()) len = 2;
        else if ((b0 & 0xF0) == 0xE0 && i + 2 < text.size()) len = 3;
        else if ((b0 & 0xF8) == 0xF0 && i + 3 < text.size()) len = 4;
        const std::string ch = text.substr(i, len);
        i += len;
        if (len == 3 && static_cast<unsigned char>(ch[0]) == 0xE3
            && static_cast<unsigned char>(ch[1]) == 0x81) {
            const unsigned char b2 = static_cast<unsigned char>(ch[2]);
            if (b2 >= 0x81 && b2 <= 0x8A) {
                out.push_back(static_cast<char>(0xE3));
                out.push_back(static_cast<char>(0x82));
                out.push_back(static_cast<char>(b2 + 0x20));
                continue;
            }
            if (b2 == 0x9B) {
                out += "\xE3\x83\xBC";
                continue;
            }
            if (b2 == 0x9D) {
                out += "\xE3\x83\x9D";
                continue;
            }
            if (b2 == 0xA4) {
                out += "\xE3\x83\xB4";
                continue;
            }
        }
        out += ch;
    }
    return out;
}

std::string replace_jp_punctuation(std::string text) {
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
    while (true) {
        const size_t pos = text.find("゙");
        if (pos == std::string::npos) break;
        text.erase(pos, strlen("゙"));
    }
    return text;
}

std::string normalize_jp_text(std::string text) {
    return replace_jp_punctuation(std::move(text));
}

std::vector<std::string> kata_to_phonemes(const std::string & kata) {
    if (kata.empty()) return {};
    if (kata == "ー") return {"ー"};
    if (is_jp_punctuation_token(kata) || is_mark_yomi(kata)) return {kata};

    auto & engine = OpenJtalkEngine::instance();
    const auto labels = engine.make_label(engine.run_frontend(kata));
    std::vector<std::string> phones;
    static const std::regex phone_re(R"(\-([^\+]*)\+)");
    for (size_t i = 1; i + 1 < labels.size(); ++i) {
        std::smatch match;
        if (!std::regex_search(labels[i], match, phone_re)) continue;
        std::string phone = match[1].str();
        if (phone == "sil" || phone == "pau") continue;
        if (phone == "cl") phone = "q";
        for (char & c : phone) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        phones.push_back(phone);
    }
    return phones;
}

void handle_long_vowel(std::vector<std::vector<std::string>> & phonemes) {
    for (size_t i = 0; i < phonemes.size(); ++i) {
        if (phonemes[i].empty()) continue;
        if (phonemes[i][0] == "ー") {
            if (i > 0 && !phonemes[i - 1].empty()) {
                phonemes[i][0] = phonemes[i - 1].back();
            }
        }
        for (size_t j = 0; j < phonemes[i].size(); ++j) {
            if (phonemes[i][j] == "ー" && j > 0) {
                phonemes[i][j] = phonemes[i][j - 1];
            }
        }
    }
}

struct SepKata {
    std::vector<std::string> sep;
    std::vector<std::string> kata;
};

SepKata text_to_sep_kata(const std::string & text) {
    const auto parsed = OpenJtalkEngine::instance().run_frontend(text);
    SepKata out;
    static const std::array<const char *, 28> rep_keys = {
        "：", "；", "，", "。", "！", "？", "\n", "．", "…", "···", "・・・", "·", "・", "、",
        "$", "“", "”", "\"", "‘", "’", "（", "）", "(", ")", "《", "》", "【", "】",
    };
    static const std::array<const char *, 28> rep_vals = {
        ",", ",", ",", ".", "!", "?", ".", ".", "...", "...", "...", ",", ",", ",",
        ".", "'", "'", "'", "'", "'", "'", "'", "'", "'", "'", "'", "'", "'",
    };

    for (const auto & parts : parsed) {
        std::string word = parts.string;
        std::string yomi = parts.pron;
        const size_t pos = yomi.find("’");
        if (pos != std::string::npos) yomi.erase(pos, strlen("’"));

        word = replace_jp_punctuation(word);
        if (!yomi.empty()) {
            if (is_mark_yomi(yomi)) {
                const auto chars = internal::utf8_chars(word);
                if (chars.size() > 1) {
                    for (const auto & ch : chars) {
                        const std::string piece = replace_jp_punctuation(ch);
                        out.kata.push_back(piece);
                        out.sep.push_back(piece);
                    }
                    continue;
                }
                if (word != "," && word != ".") word = ",";
                yomi = word;
            }
            out.kata.push_back(yomi);
        } else {
            if (is_symbol_token(word)) out.kata.push_back(word);
            else if (word == "っ" || word == "ッ") out.kata.push_back("ッ");
            else if (!is_no_yomi_token(word)) out.kata.push_back(word);
        }
        out.sep.push_back(word);
    }

    for (auto & item : out.kata) item = hira_to_kata(std::move(item));
    return out;
}

std::vector<std::pair<std::string, int>> get_accent(const std::vector<JpNjdNode> & parsed) {
    const auto labels = OpenJtalkEngine::instance().make_label(parsed);
    std::vector<std::pair<std::string, int>> accents;
    static const std::regex phone_re(R"(\-([^\+]*)\+)");
    static const std::regex a1_re(R"(/A:(\-?[0-9]+)\+)");
    static const std::regex a2_re(R"(\+(\d+)\+)");

    for (size_t n = 0; n < labels.size(); ++n) {
        std::smatch phone_match;
        if (!std::regex_search(labels[n], phone_match, phone_re)) continue;
        std::string phoneme = phone_match[1].str();
        if (phoneme == "sil" || phoneme == "pau") continue;
        if (phoneme == "cl") phoneme = "q";
        for (char & c : phoneme) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        std::smatch a1_match;
        std::smatch a2_match;
        if (!std::regex_search(labels[n], a1_match, a1_re)) continue;
        if (!std::regex_search(labels[n], a2_match, a2_re)) continue;
        const int a1 = std::stoi(a1_match[1].str());
        const int a2 = std::stoi(a2_match[1].str());

        int a2_next = -1;
        if (n + 1 < labels.size()) {
            std::smatch next_phone_match;
            if (std::regex_search(labels[n + 1], next_phone_match, phone_re)) {
                const std::string next_phone = next_phone_match[1].str();
                if (next_phone != "sil" && next_phone != "pau") {
                    std::smatch next_a2_match;
                    if (std::regex_search(labels[n + 1], next_a2_match, a2_re)) {
                        a2_next = std::stoi(next_a2_match[1].str());
                    }
                }
            }
        }

        int accent = 0;
        if (a1 == 0 && a2_next == a2 + 1) accent = -1;
        else if (a2 == 1 && a2_next == 2) accent = 1;
        accents.emplace_back(phoneme, accent);
    }
    return accents;
}

std::vector<int64_t> align_tones(
    const std::vector<std::vector<std::string>> & phones,
    std::vector<std::pair<std::string, int>> tones) {
    std::vector<int64_t> result;
    for (const auto & pho : phones) {
        std::vector<int64_t> temp(pho.size(), 0);
        for (size_t idx = 0; idx < pho.size(); ++idx) {
            if (tones.empty()) break;
            if (pho[idx] == tones.front().first) {
                temp[idx] = tones.front().second;
                if (idx > 0) temp[idx] += temp[idx - 1];
                tones.erase(tones.begin());
            }
        }
        temp.insert(temp.begin(), 0);
        if (!temp.empty()) temp.pop_back();
        const bool has_negative = std::any_of(temp.begin(), temp.end(), [](int64_t v) { return v < 0; });
        if (has_negative) {
            for (int64_t & v : temp) v += 1;
        }
        result.insert(result.end(), temp.begin(), temp.end());
    }
    return result;
}

std::vector<std::string> char_tokens(const std::string & word) {
    if (is_jp_punctuation_token(word)) return {word};
    return internal::utf8_chars(word);
}

#endif // BV2_WITH_OPENJTALK

} // namespace

bool openjtalk_available() {
#ifdef BV2_WITH_OPENJTALK
    return OpenJtalkEngine::instance().ready();
#else
    return false;
#endif
}

TextFeatures jp_text_to_sequence_openjtalk(const std::string & text) {
#ifdef BV2_WITH_OPENJTALK
    const std::string norm = normalize_jp_text(text);
    const SepKata sep_kata = text_to_sep_kata(norm);

    std::vector<std::vector<std::string>> sep_phonemes;
    sep_phonemes.reserve(sep_kata.kata.size());
    for (const auto & kata : sep_kata.kata) {
        sep_phonemes.push_back(kata_to_phonemes(kata));
    }
    handle_long_vowel(sep_phonemes);

    const auto parsed = OpenJtalkEngine::instance().run_frontend(norm);
    auto tone_pairs = get_accent(parsed);
    const std::vector<int64_t> tone_values = align_tones(sep_phonemes, std::move(tone_pairs));

    std::vector<int64_t> word2ph;
    std::vector<std::string> bert_tokens;
    for (size_t i = 0; i < sep_kata.sep.size(); ++i) {
        const auto tokens = char_tokens(sep_kata.sep[i]);
        const auto distributed = internal::distribute_phone(
            static_cast<int64_t>(sep_phonemes[i].size()),
            static_cast<int64_t>(tokens.size()));
        word2ph.insert(word2ph.end(), distributed.begin(), distributed.end());
        for (const auto & token : tokens) bert_tokens.push_back(token);
    }

    TextFeatures raw;
    raw.norm_text = norm;
    internal::add_phone(raw, "_", 0, "JP");
    raw.word2ph.push_back(1);
    size_t tone_index = 0;
    for (const auto & phoneme_group : sep_phonemes) {
        for (const auto & phone : phoneme_group) {
            const int64_t tone = tone_index < tone_values.size() ? tone_values[tone_index++] : 0;
            internal::add_phone(raw, phone, tone, "JP");
        }
    }
    internal::add_phone(raw, "_", 0, "JP");
    raw.bert_tokens = bert_tokens;
    for (int64_t n : word2ph) raw.word2ph.push_back(n);
    raw.word2ph.push_back(1);
    return internal::intersperse_blank(raw);
#else
    (void)text;
    throw std::runtime_error("this binary was built without OpenJTalk support");
#endif
}

} // namespace bv2

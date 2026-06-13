#include "bv2_tts.h"
#include "bv2_openjtalk.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open fixture: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool json_find_string(const std::string & json, const std::string & key, std::string & value) {
    const std::string pattern = "\"" + key + "\"";
    const size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;
    size_t begin = json.find('"', pos + pattern.size());
    if (begin == std::string::npos) return false;
    ++begin;
    std::string out;
    for (size_t i = begin; i < json.size(); ++i) {
        const char c = json[i];
        if (c == '"') {
            value = out;
            return true;
        }
        if (c == '\\' && i + 1 < json.size()) {
            out.push_back(json[++i]);
        } else {
            out.push_back(c);
        }
    }
    return false;
}

bool json_find_int64(const std::string & json, const std::string & key, int64_t & value) {
    const std::string pattern = "\"" + key + "\"";
    const size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;
    size_t begin = json.find(':', pos + pattern.size());
    if (begin == std::string::npos) return false;
    ++begin;
    while (begin < json.size() && std::isspace(static_cast<unsigned char>(json[begin]))) ++begin;
    size_t end = begin;
    while (end < json.size() && (std::isdigit(static_cast<unsigned char>(json[end])) || json[end] == '-')) ++end;
    if (end == begin) return false;
    value = std::stoll(json.substr(begin, end - begin));
    return true;
}

std::vector<int64_t> json_find_int_array(const std::string & json, const std::string & key) {
    const std::string pattern = "\"" + key + "\"";
    const size_t pos = json.find(pattern);
    if (pos == std::string::npos) throw std::runtime_error("fixture missing array: " + key);
    const size_t begin = json.find('[', pos);
    const size_t end = json.find(']', begin);
    if (begin == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("fixture invalid array: " + key);
    }

    std::vector<int64_t> out;
    size_t i = begin + 1;
    while (i < end) {
        while (i < end && (std::isspace(static_cast<unsigned char>(json[i])) || json[i] == ',')) ++i;
        if (i >= end) break;
        size_t j = i;
        while (j < end && (std::isdigit(static_cast<unsigned char>(json[j])) || json[j] == '-')) ++j;
        if (j == i) break;
        out.push_back(std::stoll(json.substr(i, j - i)));
        i = j;
    }
    return out;
}

void expect_eq(const char * field, const std::vector<int64_t> & actual, const std::vector<int64_t> & expected) {
    if (actual == expected) return;
    std::ostringstream ss;
    ss << field << " mismatch: actual size=" << actual.size() << " expected size=" << expected.size();
    throw std::runtime_error(ss.str());
}

void check_frontend_invariants(const bv2::TextFeatures & features) {
    if (features.phones.size() != features.tones.size()) {
        throw std::runtime_error("phones.size() != tones.size()");
    }
    if (features.phones.size() != features.languages.size()) {
        throw std::runtime_error("phones.size() != languages.size()");
    }
    if (features.word2ph.empty()) {
        throw std::runtime_error("word2ph is empty");
    }
    int64_t sum = 0;
    for (int64_t n : features.word2ph) sum += n;
    if (sum != static_cast<int64_t>(features.phones.size())) {
        throw std::runtime_error("sum(word2ph) != phones.size()");
    }
    if (features.phones.front() != 0 || features.phones.back() != 0) {
        throw std::runtime_error("phones must start and end with blank id 0");
    }
    if (features.tones.front() != 0 || features.tones.back() != 0) {
        throw std::runtime_error("tones must start and end with 0 after blank insertion");
    }
}

} // namespace

int main() {
    try {
        const std::string fixture_path = "tests/fixtures/jp_greeting.json";
        const std::string fixture = read_file(fixture_path);

        std::string text;
        std::string norm_text;
        int64_t length = 0;
        if (!json_find_string(fixture, "text", text)) throw std::runtime_error("fixture missing text");
        if (!json_find_string(fixture, "norm_text", norm_text)) throw std::runtime_error("fixture missing norm_text");
        if (!json_find_int64(fixture, "length", length)) throw std::runtime_error("fixture missing length");

        const std::vector<int64_t> expected_phones = json_find_int_array(fixture, "phones");
        const std::vector<int64_t> expected_tones = json_find_int_array(fixture, "tones");
        const std::vector<int64_t> expected_languages = json_find_int_array(fixture, "languages");
        const std::vector<int64_t> expected_word2ph = json_find_int_array(fixture, "word2ph");

        if (!bv2::openjtalk_available()) {
            throw std::runtime_error("OpenJTalk dictionary is required for Japanese frontend tests");
        }

        const bv2::TextFeatures features = bv2::text_to_sequence(text, "JP");
        check_frontend_invariants(features);

        if (features.norm_text != norm_text) {
            throw std::runtime_error("norm_text mismatch");
        }
        if (static_cast<int64_t>(features.phones.size()) != length) {
            throw std::runtime_error("length mismatch");
        }
        expect_eq("phones", features.phones, expected_phones);
        expect_eq("tones", features.tones, expected_tones);
        expect_eq("languages", features.languages, expected_languages);
        expect_eq("word2ph", features.word2ph, expected_word2ph);

        std::cout << "jp_frontend ok: text=\"" << text << "\" phones=" << features.phones.size() << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "jp_frontend failed: " << e.what() << "\n";
        return 1;
    }
}

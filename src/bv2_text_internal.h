#pragma once

#include "bv2_tts.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace bv2::internal {

const std::map<std::string, int64_t> & symbol_ids();
int64_t language_id(const std::string & language);
int64_t tone_start(const std::string & language);
void add_phone(TextFeatures & out, const std::string & symbol, int64_t tone, const std::string & language);
std::vector<std::string> utf8_chars(const std::string & text);
TextFeatures intersperse_blank(const TextFeatures & in);
std::vector<int64_t> distribute_phone(int64_t n_phone, int64_t n_word);

} // namespace bv2::internal

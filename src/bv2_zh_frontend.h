#pragma once

#include "bv2_tts.h"

#include <string>

namespace bv2::zh {

std::string text_normalize(const std::string & text);
TextFeatures text_to_sequence(const std::string & text);

} // namespace bv2::zh

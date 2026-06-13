#pragma once

#include "bv2_tts.h"

namespace bv2 {

bool openjtalk_available();
TextFeatures jp_text_to_sequence_openjtalk(const std::string & text);

} // namespace bv2

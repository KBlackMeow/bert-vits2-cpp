#include "bv2_tts.h"
#include "bv2_openjtalk.h"

#include <iostream>
#include <string>

static void print_json_array(const char * name, const std::vector<int64_t> & values) {
    std::cout << "\"" << name << "\": [";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << "]";
}

int main() {
    const std::string text = "こんにちは、世界。";
    const bv2::TextFeatures features = bv2::text_to_sequence(text, "JP");
    std::cout << "{\n";
    std::cout << "\"text\": \"" << text << "\",\n";
    std::cout << "\"language\": \"JP\",\n";
    std::cout << "\"norm_text\": \"" << features.norm_text << "\",\n";
    print_json_array("phones", features.phones);
    std::cout << ",\n";
    print_json_array("tones", features.tones);
    std::cout << ",\n";
    print_json_array("languages", features.languages);
    std::cout << ",\n";
    print_json_array("word2ph", features.word2ph);
    std::cout << ",\n";
    std::cout << "\"length\": " << features.phones.size() << "\n";
    std::cout << "}\n";
    return 0;
}

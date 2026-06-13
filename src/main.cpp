#include "bv2_tts.h"
#include "bv2_openjtalk.h"

#include <chrono>
#include <condition_variable>
#include <cctype>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <mmsystem.h>
#endif

namespace {

struct Args {
    bv2::ModelPaths paths;
    bv2::SynthesisOptions options;
    bool speaker_set = false;
    std::string speaker_name;
    std::string text;
    std::string phones;
    std::string tones;
    std::string languages;
    std::string language = "AUTO";
    std::string prefix;
    std::string config_path;
    std::string out;
    std::string model_dir = "onnx/model";
    std::string bert_zh;
    std::string bert_jp;
    std::string bert_en;
    std::string zh_bert_model = "onnx/bert/chinese-roberta-wwm-ext-large_fp16.onnx";
    std::string zh_bert_vocab = "bert/chinese-roberta-wwm-ext-large/vocab.txt";
    std::string jp_bert_model = "onnx/bert/deberta-v2-large-japanese-char-wwm_fp16.onnx";
    std::string jp_bert_vocab = "bert/deberta-v2-large-japanese-char-wwm/vocab.txt";
    std::string en_bert_model = "onnx/bert/deberta-v3-large_fp16.onnx";
    std::string en_bert_spm = "bert/deberta-v3-large/spm.model";
    std::string emotion;
    bool auto_bert = true;
    bool split_text = true;
    size_t max_chunk_chars = 240;
    int chunk_pause_ms = 120;
    bool server = false;
    bool dump_spans = false;
    std::string host = "127.0.0.1";
    int port = 7860;
    bool device_set = false;
};

std::map<std::string, int64_t> default_speaker_map() {
    return {
        {"nahida_ja", 0},
        {"illya_ja", 1},
        {"sora_ja", 2},
        {"tachibana_ja", 3},
        {"keqing_zh", 4},
        {"keqing_en", 5},
    };
}

void usage(const char * exe) {
    std::cerr
        << "usage: " << exe << " --text TEXT [options]\n"
        << "       " << exe << " --phones IDS [options]\n\n"
        << "options:\n"
        << "  --server               run built-in HTTP server\n"
        << "  --host HOST            HTTP bind host, default 127.0.0.1\n"
        << "  --port N               HTTP port, default 7860\n"
        << "  --device cpu|cuda      execution device, default auto in server mode, cpu otherwise\n"
        << "  --cuda-device N        CUDA device id, default 0\n"
        << "  --model-dir DIR        exported model directory, default onnx/model\n"
        << "  --config FILE          speaker config path, default <model-dir>/config.json\n"
        << "  --prefix NAME          ONNX file prefix, defaults to the last path component of --model-dir\n"
        << "  --text TEXT            text input; ZH/JP/EN use native C++ frontend\n"
        << "  --phones IDS           comma separated phone ids, e.g. 0,97,0,8,0\n"
        << "  --tones IDS            comma separated tone ids from the original frontend\n"
        << "  --languages IDS        comma separated language ids from the original frontend\n"
        << "  --language ZH|JP|EN|MIX|AUTO language id for tones/lang embeddings, default AUTO for --text\n"
        << "  --speaker N            speaker id\n"
        << "  --speaker-name NAME    speaker name: nahida_ja, illya_ja, sora_ja, tachibana_ja, keqing_zh, keqing_en\n"
        << "  --bert-zh FILE         raw float32 [phones,1024] Chinese BERT features\n"
        << "  --bert-jp FILE         raw float32 [phones,1024] Japanese BERT features\n"
        << "  --bert-en FILE         raw float32 [phones,1024] English BERT features\n"
        << "  --zh-bert-model FILE   Chinese BERT ONNX path, default onnx/chinese-roberta-wwm-ext-large.onnx\n"
        << "  --zh-bert-vocab FILE   Chinese BERT vocab path\n"
        << "  --jp-bert-model FILE   Japanese BERT ONNX path, default onnx/deberta-v2-large-japanese-char-wwm.onnx\n"
        << "  --jp-bert-vocab FILE   Japanese BERT vocab path\n"
        << "  --en-bert-model FILE   English BERT ONNX path, default onnx/deberta-v3-large.onnx\n"
        << "  --en-bert-spm FILE     English BERT SentencePiece model path\n"
        << "  --no-bert              disable automatic BERT for --text\n"
        << "  --no-split             disable automatic sentence chunking for --text\n"
        << "  --dump-spans           print language spans for MIX text and exit\n"
        << "  --max-chunk-chars N    max characters per text chunk, default 240\n"
        << "  --chunk-pause-ms N     silence between chunks, default 120\n"
        << "  --emotion FILE         raw float32 [512,1] emotion feature\n"
        << "  --use-emotion         pass emotion input for V2.2-style ONNX exports; project V2.3 does not use it\n"
        << "  --random-aux           use random BERT/emotion placeholders instead of zeros\n"
        << "  --noise-scale F        decoder noise scale, default 0.6\n"
        << "  --noise-scale-w F      duration noise scale, default 0.9\n"
        << "  --length-scale F       speech length scale, default 1.0\n"
        << "  --sdp-ratio F          stochastic duration predictor mix, default 0.0\n"
        << "  --sample-rate N        wav sample rate, default 44100\n"
        << "  --seed N               rng seed, default 114514\n"
        << "  -o, --output FILE      output wav path; if omitted, play audio directly\n";
}

std::string base_name(std::string path) {
    while (!path.empty() && (path.back() == '/' || path.back() == '\\')) path.pop_back();
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string join_path(const std::string & dir, const std::string & file) {
    if (dir.empty()) return file;
    const char last = dir.back();
    if (last == '/' || last == '\\') return dir + file;
    return dir + "/" + file;
}

bool file_exists(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

#ifdef _WIN32
std::string module_directory() {
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    const std::wstring wide(buf, n);
    const int size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) return {};
    std::string exe(static_cast<size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, exe.data(), size, nullptr, nullptr);
    const size_t pos = exe.find_last_of("\\/");
    return pos == std::string::npos ? std::string{} : exe.substr(0, pos);
}
#endif

bool cuda_runtime_available() {
#ifdef _WIN32
    const std::string exe_dir = module_directory();
    const std::string bin_cuda = exe_dir + "/onnxruntime_providers_cuda.dll";
    const std::string bin_shared = exe_dir + "/onnxruntime_providers_shared.dll";
    const std::string root_cuda = "bin/onnxruntime_providers_cuda.dll";
    const std::string root_shared = "bin/onnxruntime_providers_shared.dll";
    const bool provider_ok =
        (file_exists(bin_cuda) && file_exists(bin_shared))
        || (file_exists(root_cuda) && file_exists(root_shared));
    HMODULE driver = LoadLibraryA("nvcuda.dll");
    if (driver) FreeLibrary(driver);
    return provider_ok && driver != nullptr;
#else
    return false;
#endif
}

bv2::BertPaths bert_paths_from(const Args & args) {
    bv2::BertPaths paths;
    paths.zh_model = args.zh_bert_model;
    paths.zh_vocab = args.zh_bert_vocab;
    paths.jp_model = args.jp_bert_model;
    paths.jp_vocab = args.jp_bert_vocab;
    paths.en_model = args.en_bert_model;
    paths.en_spm = args.en_bert_spm;
    return paths;
}

uint32_t utf8_codepoint_at(const std::string & text, size_t pos, size_t & next) {
    const unsigned char c0 = static_cast<unsigned char>(text[pos]);
    if (c0 < 0x80) {
        next = pos + 1;
        return c0;
    }
    if ((c0 & 0xE0) == 0xC0 && pos + 1 < text.size()) {
        next = pos + 2;
        return ((c0 & 0x1F) << 6) | (static_cast<unsigned char>(text[pos + 1]) & 0x3F);
    }
    if ((c0 & 0xF0) == 0xE0 && pos + 2 < text.size()) {
        next = pos + 3;
        return ((c0 & 0x0F) << 12)
            | ((static_cast<unsigned char>(text[pos + 1]) & 0x3F) << 6)
            | (static_cast<unsigned char>(text[pos + 2]) & 0x3F);
    }
    if ((c0 & 0xF8) == 0xF0 && pos + 3 < text.size()) {
        next = pos + 4;
        return ((c0 & 0x07) << 18)
            | ((static_cast<unsigned char>(text[pos + 1]) & 0x3F) << 12)
            | ((static_cast<unsigned char>(text[pos + 2]) & 0x3F) << 6)
            | (static_cast<unsigned char>(text[pos + 3]) & 0x3F);
    }
    next = pos + 1;
    return c0;
}

bool is_sentence_break(uint32_t cp) {
    return cp == '.' || cp == '!' || cp == '?' || cp == ';'
        || cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F || cp == 0xFF1B;
}

bool is_ascii_digit_before(const std::string & text, size_t pos) {
    if (pos == 0) return false;
    const unsigned char c = static_cast<unsigned char>(text[pos - 1]);
    return c >= '0' && c <= '9';
}

bool is_ascii_digit_after(const std::string & text, size_t pos) {
    if (pos >= text.size()) return false;
    const unsigned char c = static_cast<unsigned char>(text[pos]);
    return c >= '0' && c <= '9';
}

bool is_sentence_break_at(const std::string & text, size_t pos, size_t next, uint32_t cp) {
    if (cp == '.') {
        return !(is_ascii_digit_before(text, pos) && is_ascii_digit_after(text, next));
    }
    return is_sentence_break(cp);
}

// First meaningful char language — for primary number reading
std::string first_char_language(const std::string & text) {
    for (size_t pos = 0; pos < text.size();) {
        size_t next = pos + 1;
        const uint32_t cp = utf8_codepoint_at(text, pos, next);
        if ((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF)) return "JP";
        if (cp >= 0x4E00 && cp <= 0x9FFF) return "ZH";
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return "EN";
        pos = next;
    }
    return "ZH";
}

std::string detect_language(const std::string & text) {
    bool has_kana = false, has_cjk = false, has_en = false;
    for (size_t pos = 0; pos < text.size();) {
        size_t next = pos + 1;
        const uint32_t cp = utf8_codepoint_at(text, pos, next);
        if ((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF)) {
            has_kana = true;
        } else if (cp >= 0x4E00 && cp <= 0x9FFF) {
            has_cjk = true;
        } else if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) {
            has_en = true;
        }
        pos = next;
    }

    if (has_en && (has_cjk || has_kana)) return "MIX";
    if (has_cjk && has_kana) {
        const auto spans = bv2::split_text_by_language(text, first_char_language(text));
        if (spans.size() > 1) return "MIX";
        return "JP";
    }
    if (has_kana) return "JP";
    if (has_cjk) return "ZH";
    if (has_en) return "EN";
    return "ZH";
}

std::string trim_copy(const std::string & text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return text.substr(begin, end - begin);
}

std::vector<std::string> split_text_chunks(const std::string & text, size_t max_chars) {
    if (max_chars == 0) return {text};

    std::vector<std::string> chunks;
    size_t chunk_start = 0;
    size_t last_break = std::string::npos;
    size_t last_break_next = 0;
    size_t chars = 0;
    for (size_t pos = 0; pos < text.size();) {
        size_t next = pos + 1;
        const uint32_t cp = utf8_codepoint_at(text, pos, next);
        ++chars;
        if (is_sentence_break_at(text, pos, next, cp) || cp == '\n') {
            last_break = pos;
            last_break_next = next;
        }
        if (chars >= max_chars) {
            const size_t cut = last_break != std::string::npos && last_break >= chunk_start
                ? last_break_next
                : next;
            const std::string chunk = trim_copy(text.substr(chunk_start, cut - chunk_start));
            if (!chunk.empty()) chunks.push_back(chunk);
            chunk_start = cut;
            last_break = std::string::npos;
            chars = 0;
        }
        pos = next;
    }
    const std::string tail = trim_copy(text.substr(chunk_start));
    if (!tail.empty()) chunks.push_back(tail);
    if (chunks.empty()) chunks.push_back(text);
    return chunks;
}

std::string infer_model_prefix(const std::string & model_dir) {
    const std::string dirname = base_name(model_dir);
    if (file_exists(join_path(model_dir, dirname + "_enc_p.onnx"))) return dirname;

    const std::filesystem::path dir(model_dir);
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec || !entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        const std::string suffix = "_enc_p.onnx";
        if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return name.substr(0, name.size() - suffix.size());
        }
    }
    if (ec) throw std::runtime_error("failed to scan model directory: " + model_dir);
    return dirname;
}

std::string read_text_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("failed to open config: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void skip_ws(const std::string & text, size_t & pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
}

std::string parse_json_string(const std::string & text, size_t & pos) {
    skip_ws(text, pos);
    if (pos >= text.size() || text[pos] != '"') throw std::runtime_error("expected JSON string");
    ++pos;
    std::string out;
    while (pos < text.size()) {
        const char c = text[pos++];
        if (c == '"') return out;
        if (c == '\\') {
            if (pos >= text.size()) throw std::runtime_error("unterminated JSON escape");
            const char e = text[pos++];
            if (e == '"' || e == '\\' || e == '/') out.push_back(e);
            else if (e == 'n') out.push_back('\n');
            else if (e == 'r') out.push_back('\r');
            else if (e == 't') out.push_back('\t');
            else throw std::runtime_error("unsupported JSON escape in speaker config");
        } else {
            out.push_back(c);
        }
    }
    throw std::runtime_error("unterminated JSON string");
}

int64_t parse_json_int(const std::string & text, size_t & pos) {
    skip_ws(text, pos);
    const size_t start = pos;
    if (pos < text.size() && text[pos] == '-') ++pos;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) ++pos;
    if (start == pos || (text[start] == '-' && start + 1 == pos)) throw std::runtime_error("expected speaker id");
    return std::stoll(text.substr(start, pos - start));
}

size_t find_matching_brace(const std::string & text, size_t open_pos) {
    bool in_string = false;
    bool escaped = false;
    int depth = 0;
    for (size_t i = open_pos; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
        } else {
            if (c == '"') in_string = true;
            else if (c == '{') ++depth;
            else if (c == '}') {
                --depth;
                if (depth == 0) return i;
            }
        }
    }
    throw std::runtime_error("unterminated spk2id object");
}

std::map<std::string, int64_t> load_speaker_map_from_config(const std::string & path) {
    const std::string text = read_text_file(path);
    const size_t key = text.find("\"spk2id\"");
    if (key == std::string::npos) throw std::runtime_error("config missing data.spk2id: " + path);
    const size_t open = text.find('{', key);
    if (open == std::string::npos) throw std::runtime_error("config has invalid spk2id object: " + path);
    const size_t close = find_matching_brace(text, open);

    std::map<std::string, int64_t> speakers;
    size_t pos = open + 1;
    while (true) {
        skip_ws(text, pos);
        if (pos >= close) break;
        const std::string name = parse_json_string(text, pos);
        skip_ws(text, pos);
        if (pos >= close || text[pos] != ':') throw std::runtime_error("expected ':' in spk2id");
        ++pos;
        speakers[name] = parse_json_int(text, pos);
        skip_ws(text, pos);
        if (pos < close && text[pos] == ',') ++pos;
        else if (pos < close) throw std::runtime_error("expected ',' in spk2id");
    }
    if (speakers.empty()) throw std::runtime_error("config spk2id is empty: " + path);
    return speakers;
}

int64_t default_speaker_id_for_language(
    const std::string & language,
    const std::map<std::string, int64_t> & speakers) {
    const char * preferred = language == "EN" ? "keqing_en"
        : language == "ZH" ? "keqing_zh"
        : language == "MIX" ? "keqing_zh"
        : "tachibana_ja";
    const auto it = speakers.find(preferred);
    if (it != speakers.end()) return it->second;
    if (language == "JP") {
        const auto legacy = speakers.find("tachibana");
        if (legacy != speakers.end()) return legacy->second;
    }
    throw std::runtime_error(std::string("default speaker not found in config: ") + preferred);
}

std::map<std::string, int64_t> speaker_map_for(const Args & args) {
    static std::mutex mutex;
    static std::string cached_config_path;
    static std::map<std::string, int64_t> cached_speakers;
    const std::string config_path = args.config_path.empty()
        ? join_path(args.model_dir, "config.json")
        : args.config_path;
    std::lock_guard<std::mutex> lock(mutex);
    if (config_path != cached_config_path) {
        cached_config_path = config_path;
        cached_speakers = file_exists(config_path)
            ? load_speaker_map_from_config(config_path)
            : default_speaker_map();
    }
    return cached_speakers;
}

Args parse_args(const std::vector<std::string> & argv) {
    Args args;
    for (size_t i = 1; i < argv.size(); ++i) {
        const std::string key = argv[i];
        auto next = [&](const std::string & name) -> std::string {
            if (++i >= argv.size()) throw std::runtime_error(name + " requires a value");
            return argv[i];
        };

        if (key == "--server") args.server = true;
        else if (key == "--host") args.host = next(key);
        else if (key == "--port") args.port = std::stoi(next(key));
        else if (key == "--model-dir") args.model_dir = next(key);
        else if (key == "--config") args.config_path = next(key);
        else if (key == "--prefix") args.prefix = next(key);
        else if (key == "--text") args.text = next(key);
        else if (key == "--phones") args.phones = next(key);
        else if (key == "--tones") args.tones = next(key);
        else if (key == "--languages") args.languages = next(key);
        else if (key == "--language") args.language = next(key);
        else if (key == "--speaker") {
            args.options.speaker_id = std::stoll(next(key));
            args.speaker_set = true;
        }
        else if (key == "--speaker-name") {
            args.speaker_name = next(key);
            args.speaker_set = true;
        }
        else if (key == "--bert-zh") args.bert_zh = next(key);
        else if (key == "--bert-jp") args.bert_jp = next(key);
        else if (key == "--bert-en") args.bert_en = next(key);
        else if (key == "--zh-bert-model") args.zh_bert_model = next(key);
        else if (key == "--zh-bert-vocab") args.zh_bert_vocab = next(key);
        else if (key == "--jp-bert-model") args.jp_bert_model = next(key);
        else if (key == "--jp-bert-vocab") args.jp_bert_vocab = next(key);
        else if (key == "--en-bert-model") args.en_bert_model = next(key);
        else if (key == "--en-bert-spm") args.en_bert_spm = next(key);
        else if (key == "--no-bert") args.auto_bert = false;
        else if (key == "--dump-spans") args.dump_spans = true;
        else if (key == "--no-split") args.split_text = false;
        else if (key == "--max-chunk-chars") args.max_chunk_chars = static_cast<size_t>(std::stoul(next(key)));
        else if (key == "--chunk-pause-ms") args.chunk_pause_ms = std::stoi(next(key));
        else if (key == "--device") {
            const std::string device = next(key);
            if (device == "cuda") {
                args.options.use_cuda = true;
                args.device_set = true;
            }
            else if (device == "cpu") {
                args.options.use_cuda = false;
                args.device_set = true;
            }
            else throw std::runtime_error("--device must be cpu or cuda");
        }
        else if (key == "--cuda-device") args.options.cuda_device_id = std::stoi(next(key));
        else if (key == "--emotion") args.emotion = next(key);
        else if (key == "--use-emotion") args.options.use_emotion = true;
        else if (key == "--random-aux") args.options.random_aux = true;
        else if (key == "--noise-scale") args.options.noise_scale = std::stof(next(key));
        else if (key == "--noise-scale-w") args.options.noise_scale_w = std::stof(next(key));
        else if (key == "--length-scale") args.options.length_scale = std::stof(next(key));
        else if (key == "--sdp-ratio") args.options.sdp_ratio = std::stof(next(key));
        else if (key == "--sample-rate") args.options.sample_rate = std::stoi(next(key));
        else if (key == "--seed") args.options.seed = static_cast<uint32_t>(std::stoul(next(key)));
        else if (key == "-o" || key == "--output") args.out = next(key);
        else if (key == "-h" || key == "--help") {
            usage(argv[0].c_str());
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }

    if (args.server && !args.device_set) {
        args.options.use_cuda = cuda_runtime_available();
    }
    if (args.server) {
        std::cout << "execution device: "
                  << (args.options.use_cuda ? "cuda" : "cpu");
        if (!args.device_set) std::cout << " (auto)";
        std::cout << "\n";
    }
    if (args.language == "AUTO" && !args.text.empty()) {
        args.language = detect_language(args.text);
    }
    if (args.prefix.empty()) args.prefix = infer_model_prefix(args.model_dir);
    if (!args.server && args.text.empty() == args.phones.empty()) {
        throw std::runtime_error("provide exactly one of --text or --phones");
    }
    if (args.config_path.empty()) args.config_path = join_path(args.model_dir, "config.json");
    const std::map<std::string, int64_t> speakers = speaker_map_for(args);
    if (!args.speaker_name.empty()) {
        const auto it = speakers.find(args.speaker_name);
        if (it == speakers.end()) throw std::runtime_error("unknown speaker name: " + args.speaker_name);
        args.options.speaker_id = it->second;
    }
    if (!args.speaker_set) {
        args.options.speaker_id = default_speaker_id_for_language(args.language, speakers);
    }

    args.paths.enc = join_path(args.model_dir, args.prefix + "_enc_p.onnx");
    args.paths.emb = join_path(args.model_dir, args.prefix + "_emb.onnx");
    args.paths.dp = join_path(args.model_dir, args.prefix + "_dp.onnx");
    args.paths.sdp = join_path(args.model_dir, args.prefix + "_sdp.onnx");
    args.paths.flow = join_path(args.model_dir, args.prefix + "_flow.onnx");
    args.paths.dec = join_path(args.model_dir, args.prefix + "_dec.onnx");
    return args;
}

#ifdef _WIN32
std::wstring utf8_to_wide(const std::string & value) {
    if (value.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
    if (size <= 0) throw std::runtime_error("failed to convert UTF-8 path to wide string");
    std::wstring out(static_cast<size_t>(size - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, out.data(), size);
    return out;
}

std::string wide_to_utf8(const wchar_t * value) {
    if (value == nullptr || *value == L'\0') return {};
    const int size = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) throw std::runtime_error("failed to convert argv to UTF-8");
    std::string out(static_cast<size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, out.data(), size, nullptr, nullptr);
    return out;
}
#endif

std::string playback_temp_wav_path() {
    const auto path = std::filesystem::temp_directory_path()
        / ("bert-vits2-cpp-play-" + std::to_string(
#ifdef _WIN32
            GetCurrentProcessId()
#else
            0
#endif
        ) + ".wav");
    return path.string();
}

void play_wav_file(const std::string & path) {
#ifdef _WIN32
    const std::wstring wide_path = utf8_to_wide(path);
    if (!PlaySoundW(wide_path.c_str(), nullptr, SND_FILENAME | SND_SYNC)) {
        throw std::runtime_error("failed to play wav: " + path);
    }
#else
    throw std::runtime_error("direct playback is only implemented on Windows; pass -o to write a wav file");
#endif
}

struct BertBundle {
    bv2::Tensor zh;
    bv2::Tensor jp;
    bv2::Tensor en;
    bv2::Tensor emo;
};

BertBundle build_single_language_bert(
    const Args & args,
    const bv2::TextFeatures & features,
    const std::string & norm_text,
    const std::string & language) {
    const int64_t n = static_cast<int64_t>(features.phones.size());
    BertBundle out;
    out.zh = args.options.random_aux ? bv2::random_bert(n, args.options.seed + 1) : bv2::zeros_bert(n);
    out.jp = args.options.random_aux ? bv2::random_bert(n, args.options.seed + 2) : bv2::zeros_bert(n);
    out.en = args.options.random_aux ? bv2::random_bert(n, args.options.seed + 3) : bv2::zeros_bert(n);
    out.emo = args.options.random_aux ? bv2::random_emotion(args.options.seed + 4) : bv2::zeros_emotion();

    if (!args.bert_zh.empty()) out.zh = bv2::load_float_bin(args.bert_zh, {n, 1024});
    else if (args.auto_bert && args.phones.empty() && language == "ZH") {
        out.zh = bv2::chinese_bert(args.zh_bert_model, args.zh_bert_vocab, features, args.options);
    }

    if (!args.bert_jp.empty()) out.jp = bv2::load_float_bin(args.bert_jp, {n, 1024});
    else if (args.auto_bert && args.phones.empty() && language == "JP") {
        out.jp = bv2::japanese_bert(args.jp_bert_model, args.jp_bert_vocab, features, args.options);
    }

    if (!args.bert_en.empty()) out.en = bv2::load_float_bin(args.bert_en, {n, 1024});
    else if (args.auto_bert && args.phones.empty() && language == "EN") {
        out.en = bv2::english_bert(args.en_bert_model, args.en_bert_spm, norm_text, features, args.options);
    }

    if (args.auto_bert && args.phones.empty()) {
        if (language == "ZH") {
            out.jp = bv2::random_bert(n, args.options.seed + 2);
            out.en = bv2::random_bert(n, args.options.seed + 3);
        } else if (language == "JP") {
            out.zh = bv2::random_bert(n, args.options.seed + 1);
            out.en = bv2::random_bert(n, args.options.seed + 3);
        } else if (language == "EN") {
            out.zh = bv2::random_bert(n, args.options.seed + 1);
            out.jp = bv2::random_bert(n, args.options.seed + 2);
        }
    }
    if (!args.emotion.empty()) out.emo = bv2::load_float_bin(args.emotion, {512, 1});
    return out;
}

BertBundle build_mixed_bert(const Args & args, const bv2::MixedSequence & mixed) {
    const int64_t n = static_cast<int64_t>(mixed.features.phones.size());
    BertBundle out;
    out.zh = bv2::random_bert(n, args.options.seed + 1);
    out.jp = bv2::random_bert(n, args.options.seed + 2);
    out.en = bv2::random_bert(n, args.options.seed + 3);
    out.emo = args.options.random_aux ? bv2::random_emotion(args.options.seed + 4) : bv2::zeros_emotion();

    if (!args.bert_zh.empty()) out.zh = bv2::load_float_bin(args.bert_zh, {n, 1024});
    if (!args.bert_jp.empty()) out.jp = bv2::load_float_bin(args.bert_jp, {n, 1024});
    if (!args.bert_en.empty()) out.en = bv2::load_float_bin(args.bert_en, {n, 1024});
    if (!args.emotion.empty()) out.emo = bv2::load_float_bin(args.emotion, {512, 1});

    if (!args.auto_bert || !args.phones.empty()) return out;

    for (const bv2::MixedSpanInfo & span : mixed.spans) {
        const int64_t src_begin = span.phone_begin_in_full;
        const int64_t src_end = src_begin + span.phone_count;
        const std::string & norm_text = span.full_features.norm_text.empty()
            ? span.text
            : span.full_features.norm_text;

        if (span.lang == "ZH") {
            const bv2::Tensor active = bv2::chinese_bert(
                args.zh_bert_model, args.zh_bert_vocab, span.full_features, args.options);
            bv2::copy_bert_slice(out.zh, active, src_begin, src_end, span.phone_offset);
        } else if (span.lang == "JP") {
            const bv2::Tensor active = bv2::japanese_bert(
                args.jp_bert_model, args.jp_bert_vocab, span.full_features, args.options);
            bv2::copy_bert_slice(out.jp, active, src_begin, src_end, span.phone_offset);
        } else if (span.lang == "EN") {
            const bv2::Tensor active = bv2::english_bert(
                args.en_bert_model, args.en_bert_spm, norm_text, span.full_features, args.options);
            bv2::copy_bert_slice(out.en, active, src_begin, src_end, span.phone_offset);
        }
    }
    return out;
}

std::vector<float> synthesize_features(
    const Args & args,
    const bv2::TextFeatures & features,
    const std::string & text_for_bert,
    const std::string & language_override = "") {
    const std::string language = language_override.empty() ? args.language : language_override;
    const std::string norm_text = features.norm_text.empty() ? text_for_bert : features.norm_text;
    const BertBundle bert = build_single_language_bert(args, features, norm_text, language);
    return bv2::synthesize(
        args.paths, features, args.options, &bert.zh, &bert.jp, &bert.en, &bert.emo);
}

std::vector<float> synthesize_single_language_text(const Args & args) {
    const bv2::TextFeatures features = bv2::text_to_sequence(args.text, args.language);
    return synthesize_features(args, features, args.text);
}

std::vector<float> synthesize_mixed_spans(const Args & args) {
    const std::string primary = first_char_language(args.text);
    const bv2::MixedSequence mixed = bv2::prepare_mixed_sequence(args.text, primary);
    if (mixed.spans.empty()) {
        Args fallback = args;
        fallback.language = "ZH";
        return synthesize_single_language_text(fallback);
    }
    if (mixed.spans.size() == 1) {
        Args single = args;
        single.text = mixed.spans[0].text;
        single.language = mixed.spans[0].lang;
        return synthesize_single_language_text(single);
    }

    std::cout << "mix single-pass spans=" << mixed.spans.size()
              << " speaker_id=" << args.options.speaker_id << "\n";
    for (size_t i = 0; i < mixed.spans.size(); ++i) {
        std::cout << "  span " << mixed.spans[i].lang
                  << " chars=" << mixed.spans[i].text.size()
                  << " phones=" << mixed.spans[i].phone_count << "\n";
    }

    const BertBundle bert = build_mixed_bert(args, mixed);
    return bv2::synthesize(
        args.paths, mixed.features, args.options, &bert.zh, &bert.jp, &bert.en, &bert.emo);
}

std::vector<float> synthesize_text_chunks(const Args & args) {
    if (args.language == "MIX") {
        return synthesize_mixed_spans(args);
    }

    const std::vector<std::string> chunks = args.split_text
        ? split_text_chunks(args.text, args.max_chunk_chars)
        : std::vector<std::string>{args.text};
    if (chunks.size() > 1) {
        std::cout << "split text into " << chunks.size() << " chunks\n";
    }

    std::vector<float> audio;
    const size_t pause_samples = static_cast<size_t>(
        static_cast<int64_t>(args.options.sample_rate) * args.chunk_pause_ms / 1000);
    for (size_t i = 0; i < chunks.size(); ++i) {
        bv2::TextFeatures features = bv2::text_to_sequence(chunks[i], args.language);
        std::vector<float> chunk_audio = synthesize_features(args, features, chunks[i]);
        audio.insert(audio.end(), chunk_audio.begin(), chunk_audio.end());
        if (i + 1 < chunks.size() && pause_samples > 0) {
            audio.insert(audio.end(), pause_samples, 0.0f);
        }
    }
    return audio;
}

std::vector<float> synthesize_args(const Args & args) {
    if (args.phones.empty()) {
        return synthesize_text_chunks(args);
    }
    const bv2::TextFeatures features = bv2::parse_phone_ids(args.phones, args.tones, args.languages, args.language);
    return synthesize_features(args, features, args.text);
}

using AudioChunkCallback = std::function<void(const std::vector<float> &)>;

void synthesize_args_streaming(const Args & args, const AudioChunkCallback & emit_chunk) {
    if (!args.phones.empty()) {
        const bv2::TextFeatures features = bv2::parse_phone_ids(args.phones, args.tones, args.languages, args.language);
        emit_chunk(synthesize_features(args, features, args.text));
        return;
    }

    const size_t pause_samples = static_cast<size_t>(
        static_cast<int64_t>(args.options.sample_rate) * args.chunk_pause_ms / 1000);
    const auto emit_pause = [&]() {
        if (pause_samples > 0) {
            emit_chunk(std::vector<float>(pause_samples, 0.0f));
        }
    };

    if (args.language == "MIX") {
        const std::vector<std::string> chunks = args.split_text
            ? split_text_chunks(args.text, args.max_chunk_chars)
            : std::vector<std::string>{args.text};
        if (chunks.size() > 1) {
            std::cout << "split text into " << chunks.size() << " chunks (stream)\n";
        }
        for (size_t i = 0; i < chunks.size(); ++i) {
            Args chunk_args = args;
            chunk_args.text = chunks[i];
            chunk_args.language = detect_language(chunks[i]);
            if (chunk_args.language == "MIX") {
                emit_chunk(synthesize_mixed_spans(chunk_args));
            } else {
                const bv2::TextFeatures features = bv2::text_to_sequence(chunks[i], chunk_args.language);
                emit_chunk(synthesize_features(chunk_args, features, chunks[i]));
            }
            if (i + 1 < chunks.size()) emit_pause();
        }
        return;
    }

    const std::vector<std::string> chunks = args.split_text
        ? split_text_chunks(args.text, args.max_chunk_chars)
        : std::vector<std::string>{args.text};
    if (chunks.size() > 1) {
        std::cout << "split text into " << chunks.size() << " chunks (stream)\n";
    }
    for (size_t i = 0; i < chunks.size(); ++i) {
        const bv2::TextFeatures features = bv2::text_to_sequence(chunks[i], args.language);
        emit_chunk(synthesize_features(args, features, chunks[i]));
        if (i + 1 < chunks.size()) emit_pause();
    }
}

void write_or_play_audio(const Args & args, const std::vector<float> & audio) {
    if (args.out.empty()) {
        const std::string temp_wav = playback_temp_wav_path();
        bv2::write_wav(temp_wav, audio, args.options.sample_rate);
        std::cout << "playing audio (" << audio.size() << " samples, " << args.options.sample_rate << " Hz)\n";
        play_wav_file(temp_wav);
        std::error_code ec;
        std::filesystem::remove(temp_wav, ec);
    } else {
        bv2::write_wav(args.out, audio, args.options.sample_rate);
        std::cout << "wrote " << args.out << " (" << audio.size() << " samples, " << args.options.sample_rate << " Hz)\n";
    }
}

std::string json_escape(const std::string & s) {
    std::string out;
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out.push_back(c);
    }
    return out;
}

bool json_value_pos(const std::string & body, const std::string & key, size_t & pos) {
    size_t key_pos = body.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        key_pos = body.find(key);
        while (key_pos != std::string::npos) {
            const auto is_key_char = [](char c) {
                return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
            };
            const bool left_ok = key_pos == 0 || !is_key_char(body[key_pos - 1]);
            const size_t key_end = key_pos + key.size();
            const bool right_ok = key_end >= body.size() || !is_key_char(body[key_end]);
            if (left_ok && right_ok) break;
            key_pos = body.find(key, key_pos + 1);
        }
        if (key_pos == std::string::npos) return false;
    }
    const size_t colon = body.find(':', key_pos);
    if (colon == std::string::npos) return false;
    pos = colon + 1;
    skip_ws(body, pos);
    return true;
}

std::string unescape_common_json_payload(std::string body) {
    body = trim_copy(body);
    if (body.size() >= 2 && body.front() == '\'' && body.back() == '\'') {
        body = body.substr(1, body.size() - 2);
    }
    std::string out;
    out.reserve(body.size());
    for (size_t i = 0; i < body.size(); ++i) {
        if (body[i] == '\\' && i + 1 < body.size() && body[i + 1] == '"') {
            out.push_back('"');
            ++i;
        } else {
            out.push_back(body[i]);
        }
    }
    return out;
}

bool json_get_string(const std::string & body, const std::string & key, std::string & value) {
    size_t pos = 0;
    if (!json_value_pos(body, key, pos) || pos >= body.size()) return false;
    if (body[pos] == '"') {
        value = parse_json_string(body, pos);
    } else {
        const size_t start = pos;
        while (pos < body.size() && body[pos] != ',' && body[pos] != '}') ++pos;
        value = trim_copy(body.substr(start, pos - start));
    }
    return true;
}

bool json_get_raw(const std::string & body, const std::string & key, std::string & value) {
    size_t pos = 0;
    if (!json_value_pos(body, key, pos)) return false;
    const size_t start = pos;
    while (pos < body.size() && body[pos] != ',' && body[pos] != '}') ++pos;
    value = trim_copy(body.substr(start, pos - start));
    return !value.empty();
}

bool json_get_bool(const std::string & body, const std::string & key, bool default_value = false) {
    std::string raw;
    if (!json_get_raw(body, key, raw)) return default_value;
    return raw == "true" || raw == "1";
}

void add_json_string_arg(std::vector<std::string> & argv, const std::string & body, const std::string & key, const std::string & cli) {
    std::string value;
    if (json_get_string(body, key, value) && !value.empty()) {
        argv.push_back(cli);
        argv.push_back(value);
    }
}

void add_json_raw_arg(std::vector<std::string> & argv, const std::string & body, const std::string & key, const std::string & cli) {
    std::string value;
    if (json_get_raw(body, key, value) && !value.empty()) {
        argv.push_back(cli);
        argv.push_back(value);
    }
}

std::string output_api_wav_path() {
    std::filesystem::create_directories("output");
    const auto ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return join_path("output", "api_" + std::to_string(ticks) + ".wav");
}

std::string temp_api_wav_path() {
    const auto ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path()
        / ("bert-vits2-cpp-api-" + std::to_string(ticks) + ".wav");
    return path.string();
}

class SynthesisGate {
public:
    explicit SynthesisGate(int max_concurrent) : max_concurrent_(max_concurrent) {}

    class Guard {
    public:
        explicit Guard(SynthesisGate & gate) : gate_(gate) {
            std::unique_lock<std::mutex> lock(gate_.mutex_);
            gate_.cv_.wait(lock, [&] { return gate_.active_ < gate_.max_concurrent_; });
            ++gate_.active_;
        }
        ~Guard() {
            {
                std::lock_guard<std::mutex> lock(gate_.mutex_);
                --gate_.active_;
            }
            gate_.cv_.notify_one();
        }

    private:
        SynthesisGate & gate_;
    };

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int active_ = 0;
    int max_concurrent_;
};

void apply_optional_float(const std::string & body, const std::string & key, float & value) {
    std::string raw;
    if (json_get_raw(body, key, raw)) value = std::stof(raw);
}

void apply_optional_uint32(const std::string & body, const std::string & key, uint32_t & value) {
    std::string raw;
    if (json_get_raw(body, key, raw)) value = static_cast<uint32_t>(std::stoul(raw));
}

void apply_optional_int64(const std::string & body, const std::string & key, int64_t & value) {
    std::string raw;
    if (json_get_raw(body, key, raw)) value = std::stoll(raw);
}

void apply_optional_size_t(const std::string & body, const std::string & key, size_t & value) {
    std::string raw;
    if (json_get_raw(body, key, raw)) value = static_cast<size_t>(std::stoull(raw));
}

void apply_optional_int(const std::string & body, const std::string & key, int & value) {
    std::string raw;
    if (json_get_raw(body, key, raw)) value = std::stoi(raw);
}

Args args_from_http_tts(const Args & base, const std::string & body, const std::string & out_path) {
    const std::string normalized_body = unescape_common_json_payload(body);
    std::string text;
    if (!json_get_string(normalized_body, "text", text) || text.empty()) {
        throw std::runtime_error("JSON field text is required; received body: " + normalized_body.substr(0, 200));
    }

    Args args = base;
    args.text = text;
    args.out = out_path;
    args.phones.clear();
    args.tones.clear();
    args.languages.clear();
    args.speaker_name.clear();
    args.speaker_set = false;

    std::string language;
    if (json_get_string(normalized_body, "language", language) && !language.empty()) {
        args.language = language;
    }
    json_get_string(normalized_body, "speaker_name", args.speaker_name);
    apply_optional_float(normalized_body, "length_scale", args.options.length_scale);
    apply_optional_float(normalized_body, "noise_scale", args.options.noise_scale);
    apply_optional_float(normalized_body, "noise_scale_w", args.options.noise_scale_w);
    apply_optional_float(normalized_body, "sdp_ratio", args.options.sdp_ratio);
    apply_optional_uint32(normalized_body, "seed", args.options.seed);
    apply_optional_size_t(normalized_body, "max_chunk_chars", args.max_chunk_chars);
    apply_optional_int(normalized_body, "chunk_pause_ms", args.chunk_pause_ms);
    if (json_get_bool(normalized_body, "no_bert")) args.auto_bert = false;
    if (json_get_bool(normalized_body, "no_split")) args.split_text = false;

    std::string speaker_raw;
    if (json_get_raw(normalized_body, "speaker", speaker_raw) && !speaker_raw.empty()) {
        args.options.speaker_id = std::stoll(speaker_raw);
        args.speaker_set = true;
    }

    if (args.language == "AUTO") {
        args.language = detect_language(args.text);
    }

    const std::map<std::string, int64_t> speakers = speaker_map_for(args);
    if (!args.speaker_name.empty()) {
        const auto it = speakers.find(args.speaker_name);
        if (it == speakers.end()) throw std::runtime_error("unknown speaker name: " + args.speaker_name);
        args.options.speaker_id = it->second;
        args.speaker_set = true;
    }
    if (!args.speaker_set) {
        args.options.speaker_id = default_speaker_id_for_language(args.language, speakers);
    }
    return args;
}

std::vector<std::string> build_argv_from_json(const std::string & body, const std::string & out, const Args & base_args) {
    const std::string normalized_body = unescape_common_json_payload(body);
    std::string text;
    if (!json_get_string(normalized_body, "text", text) || text.empty()) {
        throw std::runtime_error("JSON field text is required; received body: " + normalized_body.substr(0, 200));
    }

    std::vector<std::string> argv = {"api", "--model-dir", base_args.model_dir, "--prefix", base_args.prefix};
    if (!base_args.config_path.empty()) {
        argv.push_back("--config");
        argv.push_back(base_args.config_path);
    }
    argv.push_back("--device");
    argv.push_back(base_args.options.use_cuda ? "cuda" : "cpu");
    argv.push_back("--cuda-device");
    argv.push_back(std::to_string(base_args.options.cuda_device_id));
    argv.push_back("--text");
    argv.push_back(text);
    argv.push_back("-o");
    argv.push_back(out);

    std::string language;
    if (json_get_string(normalized_body, "language", language) && !language.empty()) {
        argv.push_back("--language");
        argv.push_back(language);
    }
    add_json_string_arg(argv, normalized_body, "speaker_name", "--speaker-name");
    add_json_raw_arg(argv, normalized_body, "speaker", "--speaker");
    add_json_raw_arg(argv, normalized_body, "length_scale", "--length-scale");
    add_json_raw_arg(argv, normalized_body, "noise_scale", "--noise-scale");
    add_json_raw_arg(argv, normalized_body, "noise_scale_w", "--noise-scale-w");
    add_json_raw_arg(argv, normalized_body, "sdp_ratio", "--sdp-ratio");
    add_json_raw_arg(argv, normalized_body, "seed", "--seed");
    add_json_raw_arg(argv, normalized_body, "max_chunk_chars", "--max-chunk-chars");
    add_json_raw_arg(argv, normalized_body, "chunk_pause_ms", "--chunk-pause-ms");
    if (json_get_bool(normalized_body, "no_bert")) argv.push_back("--no-bert");
    if (json_get_bool(normalized_body, "no_split")) argv.push_back("--no-split");
    return argv;
}

std::vector<char> read_binary_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("failed to read wav: " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

#ifdef _WIN32
void socket_send_all(SOCKET s, const std::string & data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const int n = send(s, data.data() + sent, static_cast<int>(data.size() - sent), 0);
        if (n <= 0) throw std::runtime_error("socket send failed");
        sent += static_cast<size_t>(n);
    }
}

void socket_send_all(SOCKET s, const std::vector<char> & data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const int n = send(s, data.data() + sent, static_cast<int>(data.size() - sent), 0);
        if (n <= 0) throw std::runtime_error("socket send failed");
        sent += static_cast<size_t>(n);
    }
}

void socket_send_http_chunk(SOCKET s, const std::vector<char> & data) {
    if (data.empty()) return;
    std::ostringstream header;
    header << std::hex << std::uppercase << data.size() << "\r\n";
    socket_send_all(s, header.str());
    socket_send_all(s, data);
    socket_send_all(s, "\r\n");
}

void socket_finish_http_chunks(SOCKET s) {
    socket_send_all(s, "0\r\n\r\n");
}

std::string http_streaming_pcm_header(int sample_rate) {
    std::ostringstream ss;
    ss << "HTTP/1.1 200 OK\r\n"
       << "Content-Type: audio/L16;rate=" << sample_rate << ";channels=1\r\n"
       << "X-Sample-Rate: " << sample_rate << "\r\n"
       << "X-Audio-Format: pcm_s16le\r\n"
       << "Transfer-Encoding: chunked\r\n"
       << "Access-Control-Allow-Origin: *\r\n"
       << "Connection: close\r\n\r\n";
    return ss.str();
}

Args g_http_base_args;
SynthesisGate g_synthesis_gate(4);

std::string http_response_json(int status, const std::string & body) {
    std::ostringstream ss;
    ss << "HTTP/1.1 " << status << (status == 200 ? " OK" : " Error") << "\r\n"
       << "Content-Type: application/json; charset=utf-8\r\n"
       << "Access-Control-Allow-Origin: *\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Connection: close\r\n\r\n"
       << body;
    return ss.str();
}

std::string now_str() {
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << std::setfill('0')
       << std::setw(2) << tm.tm_hour << ':'
       << std::setw(2) << tm.tm_min << ':'
       << std::setw(2) << tm.tm_sec << '.'
       << std::setw(3) << ms.count();
    return ss.str();
}

size_t content_length_from_headers(const std::string & request) {
    const std::string key = "Content-Length:";
    const size_t pos = request.find(key);
    if (pos == std::string::npos) return 0;
    size_t begin = pos + key.size();
    while (begin < request.size() && std::isspace(static_cast<unsigned char>(request[begin]))) ++begin;
    size_t end = begin;
    while (end < request.size() && std::isdigit(static_cast<unsigned char>(request[end]))) ++end;
    return static_cast<size_t>(std::stoul(request.substr(begin, end - begin)));
}

void handle_http_client(SOCKET client) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        std::string request;
        char buffer[8192];
        size_t header_end = std::string::npos;
        while ((header_end = request.find("\r\n\r\n")) == std::string::npos) {
            const int n = recv(client, buffer, sizeof(buffer), 0);
            if (n <= 0) return;
            request.append(buffer, buffer + n);
        }

        const size_t content_length = content_length_from_headers(request);
        const size_t body_start = header_end + 4;
        while (request.size() < body_start + content_length) {
            const int n = recv(client, buffer, sizeof(buffer), 0);
            if (n <= 0) break;
            request.append(buffer, buffer + n);
        }

        const std::string first_line = request.substr(0, request.find("\r\n"));
        if (first_line.rfind("OPTIONS ", 0) == 0) {
            std::cout << "[" << now_str() << "] OPTIONS 204" << std::endl;
            socket_send_all(client,
                "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Headers: Content-Type\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Connection: close\r\n\r\n");
            return;
        }
        if (first_line.rfind("GET /health ", 0) == 0) {
            std::cout << "[" << now_str() << "] GET /health 200" << std::endl;
            socket_send_all(client, http_response_json(200, "{\"ok\":true}"));
            return;
        }
        if (first_line.rfind("POST /tts ", 0) != 0 && first_line.rfind("POST /tts/stream ", 0) != 0) {
            std::cout << "[" << now_str() << "] " << first_line << " 404" << std::endl;
            socket_send_all(client, http_response_json(404, "{\"error\":\"not found\"}"));
            return;
        }

        const std::string body = request.substr(body_start, content_length);
        std::string text;
        json_get_string(body, "text", text);
        std::string language;
        if (!json_get_string(body, "language", language)) language = "AUTO";
        const bool explicit_stream = first_line.rfind("POST /tts/stream ", 0) == 0;
        const bool stream = explicit_stream || json_get_bool(body, "stream");
        std::cout << "[" << now_str() << "] POST " << (explicit_stream ? "/tts/stream" : "/tts")
                  << "  text=\"" << text << "\"  lang=" << language
                  << (stream ? "  stream=1" : "") << std::endl;

        const bool return_json = json_get_bool(body, "return_json");
        if (stream && return_json) {
            socket_send_all(client, http_response_json(
                400, "{\"error\":\"stream cannot be combined with return_json\"}"));
            return;
        }

        const std::string wav_path = return_json ? output_api_wav_path() : std::string();
        Args args = args_from_http_tts(g_http_base_args, body, wav_path);

        if (stream) {
            size_t total_samples = 0;
            size_t chunk_count = 0;
            socket_send_all(client, http_streaming_pcm_header(args.options.sample_rate));
            {
                SynthesisGate::Guard gate(g_synthesis_gate);
                synthesize_args_streaming(args, [&](const std::vector<float> & chunk_audio) {
                    const std::vector<char> pcm = bv2::encode_pcm16(chunk_audio);
                    socket_send_http_chunk(client, pcm);
                    total_samples += chunk_audio.size();
                    ++chunk_count;
                });
            }
            socket_finish_http_chunks(client);

            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            const float duration_sec = static_cast<float>(total_samples) / static_cast<float>(args.options.sample_rate);
            std::cout << "[" << now_str() << "] 200 stream  chunks=" << chunk_count
                      << "  samples=" << total_samples
                      << "  dur=" << std::fixed << std::setprecision(1) << duration_sec << "s"
                      << "  rt=" << elapsed << "ms" << std::endl;
            return;
        }

        std::vector<float> audio;
        {
            SynthesisGate::Guard gate(g_synthesis_gate);
            audio = synthesize_args(args);
        }

        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        const float duration_sec = static_cast<float>(audio.size()) / static_cast<float>(args.options.sample_rate);
        std::cout << "[" << now_str() << "] 200  samples=" << audio.size()
                  << "  dur=" << std::fixed << std::setprecision(1) << duration_sec << "s"
                  << "  rt=" << elapsed << "ms" << std::endl;

        if (return_json) {
            bv2::write_wav(wav_path, audio, args.options.sample_rate);
            std::ostringstream json;
            json << "{\"ok\":true,\"file\":\"" << json_escape(wav_path) << "\",\"samples\":" << audio.size() << "}";
            socket_send_all(client, http_response_json(200, json.str()));
        } else {
            const std::vector<char> wav = bv2::encode_wav(audio, args.options.sample_rate);
            std::ostringstream header;
            header << "HTTP/1.1 200 OK\r\n"
                   << "Content-Type: audio/wav\r\n"
                   << "Content-Length: " << wav.size() << "\r\n"
                   << "Access-Control-Allow-Origin: *\r\n"
                   << "Connection: close\r\n\r\n";
            socket_send_all(client, header.str());
            socket_send_all(client, wav);
        }
    } catch (const std::exception & e) {
        std::cerr << "[" << now_str() << "] ERROR 500  " << e.what() << std::endl;
        const std::string body = std::string("{\"error\":\"") + json_escape(e.what()) + "\"}";
        try {
            socket_send_all(client, http_response_json(500, body));
        } catch (...) {
        }
    }
}

int run_http_server(const Args & args) {
    g_http_base_args = args;
    std::cout << "preloading VITS ONNX sessions";
    if (args.options.use_cuda) std::cout << " (cuda:" << args.options.cuda_device_id << ")";
    else std::cout << " (cpu)";
    std::cout << "...\n";
    bv2::preload_synthesis_model(args.paths, args.options);
    std::cout << "preloading BERT ONNX sessions...\n";
    bv2::preload_bert_models(bert_paths_from(args), args.options);
    if (bv2::openjtalk_available()) {
        std::cout << "openjtalk dictionary ready\n";
    }
    std::cout << "preload complete\n";
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) throw std::runtime_error("WSAStartup failed");

    SOCKET server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server == INVALID_SOCKET) throw std::runtime_error("socket creation failed");

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(args.port));
    if (inet_pton(AF_INET, args.host.c_str(), &addr.sin_addr) != 1) {
        closesocket(server);
        throw std::runtime_error("invalid --host: " + args.host);
    }
    if (bind(server, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        closesocket(server);
        throw std::runtime_error("bind failed");
    }
    if (listen(server, SOMAXCONN) == SOCKET_ERROR) {
        closesocket(server);
        throw std::runtime_error("listen failed");
    }

    std::cout << "listening on http://" << args.host << ":" << args.port << "\n";
    std::cout << "device: " << (args.options.use_cuda ? "cuda" : "cpu");
    if (!args.device_set) std::cout << " (auto)";
    std::cout << "\n";
    std::cout << "example request: POST /tts with JSON {\"text\":\"hello\",\"language\":\"EN\"}\n";
    std::cout << "stream request: POST /tts/stream or POST /tts with {\"stream\":true}\n";
    std::cout << "curl example: curl -X POST http://" << args.host << ":" << args.port
              << "/tts -H \"Content-Type: application/json\" --data-raw \"{\\\"text\\\":\\\"hello\\\",\\\"language\\\":\\\"EN\\\"}\"\n";
    std::cout << "curl stream: curl -N -X POST http://" << args.host << ":" << args.port
              << "/tts/stream -H \"Content-Type: application/json\" --data-raw \"{\\\"text\\\":\\\"你好，这是流式测试。\\\",\\\"language\\\":\\\"ZH\\\"}\" --output output\\stream.pcm\n";
    while (true) {
        SOCKET client = accept(server, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;
        std::thread([client]() {
            handle_http_client(client);
            closesocket(client);
        }).detach();
    }
}
#else
int run_http_server(const Args &) {
    throw std::runtime_error("HTTP server is only implemented on Windows");
}
#endif

int app_main(const std::vector<std::string> & argv) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    try {
        Args args = parse_args(argv);
        if (args.server) return run_http_server(args);
        if (args.dump_spans) {
            const std::string primary = first_char_language(args.text);
            const auto spans = bv2::split_text_by_language(args.text, primary);
            std::cout << "primary=" << primary << " spans=" << spans.size() << "\n";
            for (size_t i = 0; i < spans.size(); ++i) {
                const auto & [seg, lang] = spans[i];
                std::string preview = seg.substr(0, std::min(seg.size(), size_t{48}));
                for (char & c : preview) {
                    if (c == '\n' || c == '\r') c = ' ';
                }
                std::cout << i << " " << lang << " len=" << seg.size() << " \"" << preview << "\"\n";
            }
            return 0;
        }
        write_or_play_audio(args, synthesize_args(args));
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n\n";
        usage(argv.empty() ? "bert-vits2-project" : argv[0].c_str());
        return 1;
    }
}

} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t ** wargv) {
    std::vector<std::string> argv;
    argv.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        argv.push_back(wide_to_utf8(wargv[i]));
    }
    return app_main(argv);
}
#else
int main(int argc, char ** argv) {
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return app_main(args);
}
#endif

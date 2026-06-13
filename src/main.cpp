#include "bv2_tts.h"

#include <chrono>
#include <cctype>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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
        << "  --language ZH|JP|EN|AUTO language id for tones/lang embeddings, default AUTO for --text\n"
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
        << "  --max-chunk-chars N    max characters per text chunk, default 240\n"
        << "  --chunk-pause-ms N     silence between chunks, default 120\n"
        << "  --emotion FILE         raw float32 [512,1] emotion feature\n"
        << "  --use-emotion         pass emotion input for V2.2-style ONNX exports; project V2.3 does not use it\n"
        << "  --random-aux           use random BERT/emotion placeholders instead of zeros\n"
        << "  --noise-scale F        decoder noise scale, default 0.8\n"
        << "  --noise-scale-w F      duration noise scale, default 0.6\n"
        << "  --length-scale F       speech length scale, default 1.1\n"
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

bool cuda_runtime_available() {
#ifdef _WIN32
    const bool provider_ok =
        file_exists("bin/onnxruntime_providers_cuda.dll")
        && file_exists("bin/onnxruntime_providers_shared.dll");
    HMODULE driver = LoadLibraryA("nvcuda.dll");
    if (driver) FreeLibrary(driver);
    return provider_ok && driver != nullptr;
#else
    return false;
#endif
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

    // Kana present — Japanese context; if also EN → MIX
    if (has_kana) {
        if (has_en) return "MIX";
        return "JP";
    }

    // CJK + EN → MIX
    if (has_cjk && has_en) return "MIX";
    if (has_cjk) return "ZH";
    if (has_en) return "EN";
    return "ZH";
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
    const char * preferred = language == "EN" ? "keqing_en" : language == "ZH" ? "keqing_zh" : "tachibana_ja";
    const auto it = speakers.find(preferred);
    if (it != speakers.end()) return it->second;
    if (language == "JP") {
        const auto legacy = speakers.find("tachibana");
        if (legacy != speakers.end()) return legacy->second;
    }
    throw std::runtime_error(std::string("default speaker not found in config: ") + preferred);
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
    if (args.language == "AUTO" && !args.text.empty()) {
        args.language = detect_language(args.text);
    }
    if (args.prefix.empty()) args.prefix = infer_model_prefix(args.model_dir);
    if (!args.server && args.text.empty() == args.phones.empty()) {
        throw std::runtime_error("provide exactly one of --text or --phones");
    }
    if (args.config_path.empty()) args.config_path = join_path(args.model_dir, "config.json");
    const std::map<std::string, int64_t> speakers =
        file_exists(args.config_path) ? load_speaker_map_from_config(args.config_path) : default_speaker_map();
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

std::vector<float> synthesize_features(const Args & args, const bv2::TextFeatures & features, const std::string & text_for_bert) {
    const std::string norm_text = features.norm_text.empty() ? text_for_bert : features.norm_text;
    const bool is_mixed = (args.auto_bert && args.phones.empty() && args.language == "MIX");

    if (is_mixed) {
        // ── Phone-level mixed: merge all spans, scatter BERT, single synthesis ──
        std::string primary = first_char_language(text_for_bert);
        auto spans = bv2::split_text_by_language(text_for_bert, primary);
        const int64_t n = static_cast<int64_t>(features.phones.size());

        bv2::Tensor zh_bert = bv2::zeros_bert(n);
        bv2::Tensor jp_bert = bv2::zeros_bert(n);
        bv2::Tensor en_bert = bv2::zeros_bert(n);

        // Track per-language BERT data and phone positions
        struct LangSpan { std::vector<std::string> tokens; std::vector<int64_t> word2ph; int64_t phone_start; int64_t phone_count; std::string raw_text; };
        std::vector<LangSpan> zh_data, jp_data, en_data;
        int64_t global_phone = 0;
        for (const auto & [span_text, lang] : spans) {
            bv2::TextFeatures sf = bv2::text_to_sequence(span_text, lang);
            LangSpan ls;
            ls.tokens = sf.bert_tokens;
            ls.word2ph = sf.word2ph;
            ls.phone_start = global_phone;
            ls.phone_count = static_cast<int64_t>(sf.phones.size());
            ls.raw_text = span_text;
            if (lang == "ZH") zh_data.push_back(ls);
            else if (lang == "JP") jp_data.push_back(ls);
            else if (lang == "EN") en_data.push_back(ls);
            global_phone += ls.phone_count;
        }

        // Process each span's BERT individually and scatter to global positions.
        // This avoids word2ph merging issues and SentencePiece re-tokenization mismatches.
        for (auto & d : zh_data) {
            bv2::TextFeatures tf;
            tf.bert_tokens = d.tokens;
            tf.word2ph = d.word2ph;
            tf.phones.resize(static_cast<size_t>(d.phone_count));
            bv2::Tensor bert = bv2::chinese_bert(args.zh_bert_model, args.zh_bert_vocab, tf, args.options);
            for (int64_t p = 0; p < d.phone_count; ++p)
                std::copy_n(bert.data.begin() + p * 1024, 1024, zh_bert.data.begin() + (d.phone_start + p) * 1024);
        }
        for (auto & d : jp_data) {
            bv2::TextFeatures tf;
            tf.bert_tokens = d.tokens;
            tf.word2ph = d.word2ph;
            tf.phones.resize(static_cast<size_t>(d.phone_count));
            bv2::Tensor bert = bv2::japanese_bert(args.jp_bert_model, args.jp_bert_vocab, tf, args.options);
            for (int64_t p = 0; p < d.phone_count; ++p)
                std::copy_n(bert.data.begin() + p * 1024, 1024, jp_bert.data.begin() + (d.phone_start + p) * 1024);
        }
        for (auto & d : en_data) {
            bv2::TextFeatures tf;
            tf.bert_tokens = d.tokens;
            tf.word2ph = d.word2ph;
            tf.phones.resize(static_cast<size_t>(d.phone_count));
            bv2::Tensor bert = bv2::english_bert(args.en_bert_model, args.en_bert_spm,
                d.raw_text, tf, args.options);
            for (int64_t p = 0; p < d.phone_count; ++p)
                std::copy_n(bert.data.begin() + p * 1024, 1024, en_bert.data.begin() + (d.phone_start + p) * 1024);
        }

        bv2::Tensor emo = args.options.random_aux ? bv2::random_emotion(args.options.seed + 4) : bv2::zeros_emotion();
        if (!args.emotion.empty()) emo = bv2::load_float_bin(args.emotion, {512, 1});
        return bv2::synthesize(args.paths, features, args.options, &zh_bert, &jp_bert, &en_bert, &emo);
    }

    // ── Single-language path ──
    const int64_t n = static_cast<int64_t>(features.phones.size());
    bv2::Tensor zh = args.options.random_aux ? bv2::random_bert(n, args.options.seed + 1) : bv2::zeros_bert(n);
    bv2::Tensor jp = args.options.random_aux ? bv2::random_bert(n, args.options.seed + 2) : bv2::zeros_bert(n);
    bv2::Tensor en = args.options.random_aux ? bv2::random_bert(n, args.options.seed + 3) : bv2::zeros_bert(n);
    bv2::Tensor emo = args.options.random_aux ? bv2::random_emotion(args.options.seed + 4) : bv2::zeros_emotion();

    if (!args.bert_zh.empty()) zh = bv2::load_float_bin(args.bert_zh, {n, 1024});
    else if (args.auto_bert && args.phones.empty() && args.language == "ZH") {
        zh = bv2::chinese_bert(args.zh_bert_model, args.zh_bert_vocab, features, args.options);
    }

    if (!args.bert_jp.empty()) jp = bv2::load_float_bin(args.bert_jp, {n, 1024});
    else if (args.auto_bert && args.phones.empty() && args.language == "JP") {
        jp = bv2::japanese_bert(args.jp_bert_model, args.jp_bert_vocab, features, args.options);
    }

    if (!args.bert_en.empty()) en = bv2::load_float_bin(args.bert_en, {n, 1024});
    else if (args.auto_bert && args.phones.empty() && args.language == "EN") {
        en = bv2::english_bert(args.en_bert_model, args.en_bert_spm, norm_text, features, args.options);
    }

    if (args.auto_bert && args.phones.empty()) {
        if (args.language == "ZH") {
            jp = bv2::random_bert(n, args.options.seed + 2);
            en = bv2::random_bert(n, args.options.seed + 3);
        } else if (args.language == "JP") {
            zh = bv2::random_bert(n, args.options.seed + 1);
            en = bv2::random_bert(n, args.options.seed + 3);
        } else if (args.language == "EN") {
            zh = bv2::random_bert(n, args.options.seed + 1);
            jp = bv2::random_bert(n, args.options.seed + 2);
        }
    }
    if (!args.emotion.empty()) emo = bv2::load_float_bin(args.emotion, {512, 1});

    return bv2::synthesize(args.paths, features, args.options, &zh, &jp, &en, &emo);
}

std::vector<float> synthesize_text_chunks(const Args & args) {
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

Args g_http_base_args;

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
        if (first_line.rfind("POST /tts ", 0) != 0) {
            std::cout << "[" << now_str() << "] " << first_line << " 404" << std::endl;
            socket_send_all(client, http_response_json(404, "{\"error\":\"not found\"}"));
            return;
        }

        const std::string body = request.substr(body_start, content_length);
        std::string text;
        json_get_string(body, "text", text);
        std::string language;
        if (!json_get_string(body, "language", language)) language = "AUTO";
        std::cout << "[" << now_str() << "] POST /tts  text=\"" << text << "\"  lang=" << language << std::endl;

        const bool return_json = json_get_bool(body, "return_json");
        const std::string wav_path = return_json ? output_api_wav_path() : temp_api_wav_path();
        Args args = parse_args(build_argv_from_json(body, wav_path, g_http_base_args));
        const std::vector<float> audio = synthesize_args(args);
        bv2::write_wav(wav_path, audio, args.options.sample_rate);

        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        const float duration_sec = static_cast<float>(audio.size()) / static_cast<float>(args.options.sample_rate);
        std::cout << "[" << now_str() << "] 200  samples=" << audio.size()
                  << "  dur=" << std::fixed << std::setprecision(1) << duration_sec << "s"
                  << "  rt=" << elapsed << "ms" << std::endl;

        if (return_json) {
            std::ostringstream json;
            json << "{\"ok\":true,\"file\":\"" << json_escape(wav_path) << "\",\"samples\":" << audio.size() << "}";
            socket_send_all(client, http_response_json(200, json.str()));
        } else {
            const std::vector<char> wav = read_binary_file(wav_path);
            std::ostringstream header;
            header << "HTTP/1.1 200 OK\r\n"
                   << "Content-Type: audio/wav\r\n"
                   << "Content-Length: " << wav.size() << "\r\n"
                   << "Access-Control-Allow-Origin: *\r\n"
                   << "Connection: close\r\n\r\n";
            socket_send_all(client, header.str());
            socket_send_all(client, wav);
            std::error_code ec;
            std::filesystem::remove(wav_path, ec);
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
    std::cout << "preloading VITS ONNX sessions...\n";
    bv2::preload_synthesis_model(args.paths, args.options);
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
    std::cout << "curl example: curl -X POST http://" << args.host << ":" << args.port
              << "/tts -H \"Content-Type: application/json\" --data-raw \"{\\\"text\\\":\\\"hello\\\",\\\"language\\\":\\\"EN\\\"}\"\n";
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

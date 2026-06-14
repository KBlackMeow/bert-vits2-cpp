#include "bv2_mlx_runtime.h"
#include "bv2_mlx_models.h"
#include "bv2_mlx_modules.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

// We use a tiny header-only JSON reader for `config.json` to avoid pulling
// in another dependency. Keep it minimal: we only need top-level keys
// `model`, `data`, `version` and a handful of nested ints / lists.
#include <cctype>

namespace mc = ::mlx::core;

namespace bv2 {
namespace mlx_rt {

namespace {

// ---------------- minimal JSON reader (good enough for our config.json) ----

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type t = Null;
    double num = 0;
    bool boolean = false;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    bool has(const std::string& k) const {
        for (auto& kv : obj) if (kv.first == k) return true;
        return false;
    }
    const Json& at(const std::string& k) const {
        for (auto& kv : obj) if (kv.first == k) return kv.second;
        throw std::runtime_error("config.json missing key: " + k);
    }
};

struct JsonParser {
    // Store the source by value (not by reference) so callers can pass
    // temporaries like `ss.str()` without dangling-reference UB.
    std::string s;
    size_t i = 0;
    explicit JsonParser(std::string src) : s(std::move(src)) {}

    void skip_ws() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
    char peek() { skip_ws(); return i < s.size() ? s[i] : '\0'; }
    char get()  { skip_ws(); return i < s.size() ? s[i++] : '\0'; }

    Json parse() {
        skip_ws();
        char c = s[i];
        if (c == '{') return parse_obj();
        if (c == '[') return parse_arr();
        if (c == '"') return parse_str();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_num();
    }
    Json parse_obj() {
        Json j; j.t = Json::Object;
        ++i;   // {
        skip_ws();
        if (peek() == '}') { ++i; return j; }
        while (true) {
            skip_ws();
            Json key = parse_str();
            skip_ws();
            if (get() != ':') throw std::runtime_error("expected :");
            Json val = parse();
            j.obj.emplace_back(key.str, std::move(val));
            skip_ws();
            char c = get();
            if (c == ',') continue;
            if (c == '}') break;
            throw std::runtime_error("expected , or } in object");
        }
        return j;
    }
    Json parse_arr() {
        Json j; j.t = Json::Array;
        ++i;
        skip_ws();
        if (peek() == ']') { ++i; return j; }
        while (true) {
            j.arr.push_back(parse());
            skip_ws();
            char c = get();
            if (c == ',') continue;
            if (c == ']') break;
            throw std::runtime_error("expected , or ] in array");
        }
        return j;
    }
    Json parse_str() {
        Json j; j.t = Json::String;
        if (get() != '"') throw std::runtime_error("expected string");
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char e = s[i + 1];
                if (e == 'n')      out.push_back('\n');
                else if (e == 't') out.push_back('\t');
                else if (e == 'r') out.push_back('\r');
                else                out.push_back(e);
                i += 2;
            } else {
                out.push_back(s[i++]);
            }
        }
        ++i;   // closing quote
        j.str = std::move(out);
        return j;
    }
    Json parse_num() {
        size_t start = i;
        while (i < s.size() &&
               (std::isdigit(static_cast<unsigned char>(s[i])) ||
                s[i] == '-' || s[i] == '+' || s[i] == '.' ||
                s[i] == 'e' || s[i] == 'E')) ++i;
        Json j; j.t = Json::Number;
        std::string token = s.substr(start, i - start);
        try {
            j.num = std::stod(token);
        } catch (const std::exception& e) {
            // Decorate with byte offset and surrounding context so config.json
            // problems are diagnosable instead of just emitting `stod: no
            // conversion`.
            size_t ctx_lo = start > 32 ? start - 32 : 0;
            size_t ctx_hi = std::min(s.size(), i + 32);
            std::string ctx = s.substr(ctx_lo, ctx_hi - ctx_lo);
            for (auto& ch : ctx) if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
            throw std::runtime_error(
                "JSON number parse failed at byte " + std::to_string(start)
                + ": token=\"" + token + "\" (\"" + ctx + "\")");
        }
        return j;
    }
    Json parse_bool() {
        Json j; j.t = Json::Bool;
        if (s.compare(i, 4, "true") == 0)  { i += 4; j.boolean = true; }
        else if (s.compare(i, 5, "false") == 0) { i += 5; j.boolean = false; }
        else throw std::runtime_error("invalid bool");
        return j;
    }
    Json parse_null() {
        Json j; j.t = Json::Null;
        if (s.compare(i, 4, "null") == 0) { i += 4; }
        else throw std::runtime_error("invalid null");
        return j;
    }
};

Json read_json(const std::string& path) {
    std::ifstream fp(path);
    if (!fp) throw std::runtime_error("cannot open: " + path);
    std::stringstream ss;
    ss << fp.rdbuf();
    JsonParser jp(ss.str());
    try {
        return jp.parse();
    } catch (const std::exception& e) {
        throw std::runtime_error("parsing " + path + ": " + e.what());
    }
}

MlxConfig parse_config(const Json& j) {
    MlxConfig c;
    if (j.has("model")) {
        const auto& m = j.at("model");
        auto get_int = [&](const std::string& k, int& dst) {
            if (m.has(k)) dst = static_cast<int>(m.at(k).num);
        };
        auto get_int_list = [&](const std::string& k, std::vector<int>& dst) {
            if (!m.has(k)) return;
            dst.clear();
            for (auto& e : m.at(k).arr) dst.push_back(static_cast<int>(e.num));
        };
        get_int("inter_channels",       c.inter_channels);
        get_int("hidden_channels",      c.hidden_channels);
        get_int("filter_channels",      c.filter_channels);
        get_int("n_heads",              c.n_heads);
        get_int("n_layers",             c.n_layers);
        get_int("kernel_size",          c.kernel_size);
        get_int("gin_channels",         c.gin_channels);
        get_int("upsample_initial_channel", c.upsample_initial_channel);
        get_int_list("upsample_rates",        c.upsample_rates);
        get_int_list("upsample_kernel_sizes", c.upsample_kernel_sizes);
        get_int_list("resblock_kernel_sizes", c.resblock_kernel_sizes);
        if (m.has("resblock")) {
            const auto& rb = m.at("resblock");
            c.resblock_type = (rb.t == Json::String) ? rb.str
                                                     : std::to_string(static_cast<int>(rb.num));
        }
        if (m.has("resblock_dilation_sizes")) {
            c.resblock_dilation_sizes.clear();
            for (auto& outer : m.at("resblock_dilation_sizes").arr) {
                std::vector<int> inner;
                for (auto& e : outer.arr) inner.push_back(static_cast<int>(e.num));
                c.resblock_dilation_sizes.push_back(std::move(inner));
            }
        }
        // Optional v2.3 flow knobs (defaults from upstream V230 SynthesizerTrn).
        get_int("n_flow_layer", c.n_flow_layer);
        get_int("n_layers_trans_flow", c.n_layers_trans_flow);
        if (m.has("use_transformer_flow")) {
            c.use_transformer_flow = m.at("use_transformer_flow").boolean;
        }
    }
    if (j.has("data")) {
        const auto& d = j.at("data");
        if (d.has("sampling_rate")) c.sampling_rate = static_cast<int>(d.at("sampling_rate").num);
        if (d.has("n_speakers"))    c.n_speakers    = static_cast<int>(d.at("n_speakers").num);
        if (d.has("filter_length")) c.spec_channels = static_cast<int>(d.at("filter_length").num) / 2 + 1;
    }
    return c;
}

} // namespace

// ---------------------------------------------------------------------------
// MlxVitsRuntime::Impl
// ---------------------------------------------------------------------------

struct MlxVitsRuntime::Impl {
    bool loaded = false;
    MlxConfig cfg;
    ParamStore params;
};

MlxVitsRuntime::MlxVitsRuntime() : impl_(std::make_unique<Impl>()) {}
MlxVitsRuntime::~MlxVitsRuntime() = default;
MlxVitsRuntime::MlxVitsRuntime(MlxVitsRuntime&&) noexcept = default;
MlxVitsRuntime& MlxVitsRuntime::operator=(MlxVitsRuntime&&) noexcept = default;

bool MlxVitsRuntime::is_loaded() const noexcept { return impl_ && impl_->loaded; }

int MlxVitsRuntime::sampling_rate() const noexcept {
    return impl_ ? impl_->cfg.sampling_rate : 0;
}

bool MlxVitsRuntime::load(const std::string& model_dir, std::string* error_out) {
    return load_files(model_dir + "/G_mlx.safetensors",
                      model_dir + "/config.json", error_out);
}

bool MlxVitsRuntime::load_files(const std::string& weights_path,
                                const std::string& config_path,
                                std::string* error_out) {
    auto set_err = [&](const std::string& s) {
        if (error_out) *error_out = s;
    };
    try {
        Json j = read_json(config_path);
        impl_->cfg    = parse_config(j);
        impl_->params = load_safetensors(weights_path);

        if (!impl_->cfg.use_transformer_flow) {
            set_err("MLX runtime currently requires use_transformer_flow=true");
            return false;
        }
        impl_->loaded = true;
        return true;
    } catch (const std::exception& e) {
        set_err(std::string("MLX load failed: ") + e.what());
        return false;
    }
}

std::vector<float> MlxVitsRuntime::infer(const MlxInferInputs& in,
                                         std::string* error_out) {
    if (!is_loaded()) {
        if (error_out) *error_out = "MLX runtime not loaded";
        return {};
    }
    try {
        array audio = synthesizer_infer(impl_->params, in, impl_->cfg);
        // Materialise on host and copy out as plain float vector.
        mc::eval(audio);
        auto sh = audio.shape();   // [1, 1, T]
        if (sh.size() != 3) {
            throw std::runtime_error("synthesizer output rank != 3");
        }
        size_t total = static_cast<size_t>(sh[0]) * sh[1] * sh[2];
        std::vector<float> out(total);
        const float* src = audio.data<float>();
        std::copy(src, src + total, out.begin());
        return out;
    } catch (const std::exception& e) {
        if (error_out) *error_out = std::string("MLX infer failed: ") + e.what();
        return {};
    }
}

} // namespace mlx_rt
} // namespace bv2

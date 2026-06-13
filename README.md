# Bert-VITS2 C++ runtime package

This package contains the C++ runtime, exported Bert-VITS2 model, exported
multilingual BERT models, tokenizer assets, and ONNX Runtime DLLs.

Run and play audio directly:

```powershell
.\bin\bert-vits2-project.exe --text "你好，这是测试。" --language ZH --speaker-name keqing_zh
```

Mixed Chinese + Japanese + English in one sentence (`--language AUTO` detects `MIX`).
Each MIX span keeps its own language frontend (`ZH` / `EN` / `JP` phones, tones, BERT). `--speaker-name` applies to the whole utterance and only changes voice style:

```powershell
.\bin\bert-vits2-project.exe --text "你好，Hello，こんにちは。" --language AUTO --speaker-name keqing_zh -o output\mix.wav
```

Use `--dump-spans` to inspect how the text is split before synthesis.

Write a wav file instead:

```powershell
.\bin\bert-vits2-project.exe --text "你好，这是测试。" --language ZH --speaker-name keqing_zh -o output\out.wav
.\bin\bert-vits2-project.exe --text "你好，这是测试。" --language ZH --speaker-name keqing_zh --device cuda -o output\out_cuda.wav
```

When `-o` or `--output` is omitted, the program writes a temporary wav, plays it
with the Windows audio API, and then removes the temporary file.

Multilingual BERT assets:

- Chinese: `bert/chinese-roberta-wwm-ext-large` + `onnx/chinese-roberta-wwm-ext-large.onnx`
- Japanese: `bert/deberta-v2-large-japanese-char-wwm` + `onnx/deberta-v2-large-japanese-char-wwm.onnx`
- English: `bert/deberta-v3-large` + `onnx/deberta-v3-large.onnx`

Frontend data files (required for `--text`, run from project root):

```text
text/opencpop-strict.txt
text/cmudict.rep
text/jieba/
src/zh_pinyin.tsv
src/zh_word_pinyin.tsv
src/zh_neural_tone_words.txt
src/zh_not_neural_tone_words.txt
bert/chinese-roberta-wwm-ext-large/vocab.txt
bert/deberta-v2-large-japanese-char-wwm/vocab.txt
bert/deberta-v3-large/spm.model
```

Regenerate Chinese frontend data files from the Python reference:

```powershell
E:\AI\vits2_env\python.exe src\make_zh_data.py
```

One-time setup if you see `required frontend asset was not found`:

```powershell
# Chinese pinyin table (requires: pip install pypinyin)
python src\make_zh_pinyin_table.py

# G2P tables from Bert-VITS2
New-Item -ItemType Directory -Force text | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/fishaudio/Bert-VITS2/master/text/opencpop-strict.txt" -OutFile text\opencpop-strict.txt
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/fishaudio/Bert-VITS2/master/text/cmudict.rep" -OutFile text\cmudict.rep

# BERT tokenizers (ONNX models stay under onnx/bert/)
New-Item -ItemType Directory -Force bert\deberta-v2-large-japanese-char-wwm,bert\deberta-v3-large | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm/resolve/main/vocab.txt" -OutFile bert\deberta-v2-large-japanese-char-wwm\vocab.txt
Invoke-WebRequest -Uri "https://huggingface.co/microsoft/deberta-v3-large/resolve/main/spm.model" -OutFile bert\deberta-v3-large\spm.model
```

Examples:

```powershell
.\bin\bert-vits2-project.exe --text "Hello, world." --language EN --speaker-name keqing_en
.\bin\bert-vits2-project.exe --text "こんにちは、世界。" --language JP --speaker-name tachibana_ja
.\bin\bert-vits2-project.exe --text "你好，这是测试。" --language ZH --speaker-name keqing_zh
```

Built-in C++ HTTP API:

```powershell
.\bin\bert-vits2-project.exe --server --host 127.0.0.1 --port 7860
```

The server auto-selects CUDA when CUDA Runtime and ONNX Runtime CUDA provider
are available. Override the server-wide device at startup:

```powershell
.\bin\bert-vits2-project.exe --server --device cuda --cuda-device 0
.\bin\bert-vits2-project.exe --server --device cpu
```

In server mode, the VITS and language BERT ONNX sessions are preloaded at startup
and reused for requests.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/tts` | Synthesize speech (WAV by default) |
| `POST` | `/tts/stream` | Synthesize speech with chunked PCM streaming |

All `POST` bodies use `Content-Type: application/json`. CORS is enabled
(`Access-Control-Allow-Origin: *`).

### Request body (`POST /tts`, `POST /tts/stream`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | yes | — | Input text to synthesize |
| `language` | string | no | `AUTO` | `ZH`, `JP`, `EN`, `MIX`, or `AUTO` (auto-detect) |
| `speaker_name` | string | no | language default | Speaker name from `config.json` (see below) |
| `speaker` | integer | no | language default | Speaker id; overrides `speaker_name` when set |
| `length_scale` | float | no | `1.0` | Speech length scale |
| `noise_scale` | float | no | `0.6` | Decoder noise scale |
| `noise_scale_w` | float | no | `0.9` | Duration noise scale |
| `sdp_ratio` | float | no | `0.0` | Stochastic duration predictor mix |
| `seed` | integer | no | `114514` | RNG seed |
| `max_chunk_chars` | integer | no | `240` | Max characters per text chunk (long-text splitting) |
| `chunk_pause_ms` | integer | no | `120` | Silence between chunks, in milliseconds |
| `no_bert` | boolean | no | `false` | Disable automatic BERT inference |
| `no_split` | boolean | no | `false` | Disable automatic sentence chunking |
| `return_json` | boolean | no | `false` | Return JSON with output file path instead of WAV (`/tts` only) |
| `stream` | boolean | no | `false` | Stream PCM chunks (`/tts` only; use `/tts/stream` as an alternative) |

`stream` and `return_json` cannot be combined.

Example body:

```json
{
  "text": "你好，这是接口测试。",
  "language": "ZH",
  "speaker_name": "keqing_zh",
  "length_scale": 1.0,
  "noise_scale": 0.6
}
```

### `GET /health`

Response: `200 OK`, JSON body:

```json
{"ok": true}
```

### `POST /tts` (non-streaming)

Default response: `200 OK`, complete WAV file.

| Header | Value |
|--------|-------|
| `Content-Type` | `audio/wav` |
| `Content-Length` | WAV byte length |

```powershell
curl.exe -X POST http://127.0.0.1:7860/tts `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"你好，这是接口测试。\",\"language\":\"ZH\",\"speaker_name\":\"keqing_zh\"}" `
  --output output\api_test.wav
```

With `return_json: true`, response is JSON instead of WAV:

```json
{"ok": true, "file": "C:\\...\\bert-vits2-cpp-api-....wav", "samples": 123456}
```

```powershell
curl.exe -X POST http://127.0.0.1:7860/tts `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"你好，这是接口测试。\",\"language\":\"ZH\",\"return_json\":true}"
```

### `POST /tts/stream` (streaming)

Synthesizes long text in sentence chunks and sends each chunk as soon as it is
ready. Use this endpoint, or pass `"stream": true` on `/tts`.

Response: `200 OK`, chunked transfer encoding, raw PCM audio.

| Header | Value |
|--------|-------|
| `Content-Type` | `audio/L16;rate=44100;channels=1` |
| `Transfer-Encoding` | `chunked` |
| `X-Sample-Rate` | `44100` (matches server sample rate) |
| `X-Audio-Format` | `pcm_s16le` |

PCM format: signed 16-bit little-endian, mono, sample rate from `X-Sample-Rate`.
Between text chunks, `chunk_pause_ms` silence is inserted (default 120 ms).

```powershell
curl.exe -N -X POST http://127.0.0.1:7860/tts/stream `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"你好，这是流式测试。\",\"language\":\"ZH\",\"speaker_name\":\"keqing_zh\"}" `
  --output output\stream.pcm
```

Equivalent using `stream` on `/tts`:

```powershell
curl.exe -N -X POST http://127.0.0.1:7860/tts `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"你好，这是流式测试。\",\"language\":\"ZH\",\"stream\":true}" `
  --output output\stream.pcm
```

Use `-N` (`--no-buffer`) with curl so chunks are not buffered client-side.

Play or convert the PCM output:

```powershell
ffmpeg -f s16le -ar 44100 -ac 1 -i output\stream.pcm output\stream.wav
```

Notes for streaming:

- Long text is split at sentence boundaries (`max_chunk_chars`, default 240).
- For `MIX` language, each chunk is language-detected independently (differs from
  non-streaming `MIX`, which uses a single-pass multi-span synthesis).
- `return_json` is not supported in streaming mode (`400` error).

### Errors

| Status | Body | Typical cause |
|--------|------|---------------|
| `400` | `{"error":"..."}` | Invalid JSON, missing `text`, `stream` + `return_json` |
| `404` | `{"error":"not found"}` | Unknown path |
| `500` | `{"error":"..."}` | Synthesis or frontend failure |

Model speaker names are read from `<model-dir>\config.json` by default. For the
included model, the config is next to the ONNX files:

```text
onnx/model/config.json
```

Available speakers:

```text
nahida_ja=0
illya_ja=1
sora_ja=2
tachibana_ja=3
keqing_zh=4
keqing_en=5
```

Export the VITS model from a PyTorch checkpoint:

```powershell
python src\export_project_model.py ..\Data\models\G_46000.pth
```

Rebuild from source:

```powershell
cmake --build build --config Release
```

The build auto-detects `third_party/onnxruntime-gpu-windows-nuget` or
`third_party/onnxruntime-nuget`, copies ONNX Runtime DLLs, and writes
`bin\bert-vits2-project.exe`.

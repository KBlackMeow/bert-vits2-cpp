# Bert-VITS2 C++ runtime package

This package contains the C++ runtime, exported Bert-VITS2 model, exported
multilingual BERT models, tokenizer assets, and ONNX Runtime DLLs.

Run and play audio directly:

```powershell
.\bin\bert-vits2-project.exe --text "你好，这是测试。" --language ZH --speaker-name keqing_zh
```

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

In server mode, the VITS ONNX sessions are preloaded at startup and reused for
requests. Language BERT ONNX sessions are loaded on first use and then cached.

POST `/tts` returns `audio/wav` by default:

```powershell
curl.exe -X POST http://127.0.0.1:7860/tts `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"你好，这是接口测试。\",\"language\":\"ZH\",\"speaker_name\":\"keqing_zh\"}" `
  --output output\api_test.wav
```

Return JSON with the generated output path:

```powershell
curl.exe -X POST http://127.0.0.1:7860/tts `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"你好，这是接口测试。\",\"language\":\"ZH\",\"return_json\":true}"
```

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

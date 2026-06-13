# Tests

## Japanese frontend (C++, no server)

Validates OpenJTalk Japanese frontend output against `tests/fixtures/jp_greeting.json`.

```powershell
cmake --build build --config Release --target test_jp_frontend
.\build\tests\Release\test_jp_frontend.exe
```

Or via CTest from the project root:

```powershell
ctest --test-dir build -C Release -R jp_frontend --output-on-failure
```

Refresh the fixture after intentional frontend changes:

```powershell
.\build\tests\Release\gen_jp_fixture.exe > tests\fixtures\jp_greeting.json
```

## Japanese HTTP API (Python, server required)

Checks Japanese WAV synthesis and PCM streaming on a running server.

```powershell
.\bin\bert-vits2-project.exe --server --host 127.0.0.1 --port 7860
python tests\test_jp_api.py
```

Optional environment variables: `BERT_VITS2_HOST`, `BERT_VITS2_PORT`.

## Stream and play (Python)

Receive PCM from `/tts/stream` and play while chunks arrive (requires `sounddevice`):

```powershell
pip install sounddevice
python tests\stream_play.py --speaker-name tachibana_ja
```

Japanese example:

```powershell
python tests\stream_play.py `
  --text "こんにちは。これはストリーミング再生テストです。" `
  --language JP `
  --speaker-name tachibana_ja `
  --save output\stream_live.pcm
```

Chinese example:

```powershell
python tests\stream_play.py `
  --text "你好，这是流式播放测试。" `
  --language ZH `
  --speaker-name keqing_zh
```

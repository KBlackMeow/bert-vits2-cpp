#!/usr/bin/env bash
# Smoke-test the local TTS server with a long Chinese poem.
# Requires the server to be running at 127.0.0.1:7860 and an existing
# output/ directory next to where this script is invoked from.
set -euo pipefail

curl -X POST http://127.0.0.1:7860/tts \
  -H "Content-Type: application/json" \
  -o output/test_zh.wav \
  --data-raw '{"text":"我好喜欢你呀","language":"ZH","speaker_name":"keqing_zh","length_scale":1.1}'

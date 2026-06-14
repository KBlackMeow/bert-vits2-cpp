#!/usr/bin/env bash
# 立华奏 · 三语 MIX 自动声线测试
# Requires the server to be running at 127.0.0.1:7860 and an existing
# output/ directory next to where this script is invoked from.
set -euo pipefail

curl -X POST http://127.0.0.1:7860/tts \
  -H "Content-Type: application/json" \
  -o output/test_tachibana_mix.wav \
  --data-raw '{"text":"こんにちは。立華奏です。今日はこのプロジェクトについて、三つの言語で紹介します。まずは日本語から始めます。Bert-VITS2 C++ランタイムは、テキストを音声に変換するための高速な推論エンジンです。中国語、日本語、英語の三言語に対応しており、一文の中で自由に混ぜることができます。システムが自動的に言語を検出し、それぞれの専用フロントエンドで処理します。中国語はピンインと声調、日本語は形態素解析とアクセント、英語は発音辞書を使います。Now let me switch to English. This system supports mixed-language synthesis in real time. You can write Chinese, Japanese, and English together in a single paragraph, and the system will detect each language span automatically. The BERT-based frontend ensures natural prosody and accurate pronunciation across all three languages. It runs on Windows, Linux, and macOS with native GPU acceleration. 次は中国語です。这个项目完全开源。代码公开在代码托管平台上。任何人都可以使用，也可以参与改进。使用方法非常简单，终端里输入一行命令即可。指定文本，选择语言和发音人，按下回车。如果你在做应用集成，直接调用网络接口就能拿到合成结果。支持流式输出，长文本会自动分句处理。总之，这是一个高效可靠的文字转语音工具。如果有什么想传达的话，它会帮你。以上です。","language":"AUTO","length_scale":1.1}'
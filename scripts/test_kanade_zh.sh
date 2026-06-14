#!/usr/bin/env bash
# 立华奏 · 项目介绍 TTS 测试
# Requires the server to be running at 127.0.0.1:7860 and an existing
# output/ directory next to where this script is invoked from.
set -euo pipefail

curl -X POST http://127.0.0.1:7860/tts \
  -H "Content-Type: application/json" \
  -o output/intro_kanade.wav \
  --data-raw '{"text":"初次见面。我是立华奏。……这个项目，叫做 Bert-VITS2 C++ 运行时。它是一个语音合成的工具。可以把写下的文字，变成可以听见的声音。它支持三种语言。中文。日文。还有英文。一句话里，混合不同的语言也没有关系。它会自动识别，然后分别处理。在 Windows 上可以运行。在 Linux 上也可以运行。在 macOS 上面，还有原生的 MLX 推理引擎。用 Apple Silicon 芯片的话，会很快。……特别厉害。项目内置了一个 HTTP 服务。用一行命令启动它，就可以通过接口来合成语音。支持流式输出。一边生成，一边播放。不用等很久。可用的音色有好几个。日文的音色有纳西妲、伊莉雅、空。中文和英文的刻晴。……还有，我的声音。如果你愿意的话，请用我的声音，来读你想说的话。使用的方法很简单。在终端里面，输入程序的名字。告诉它你要说的文字。选择一种语言。再指定一位发音人。最后按下回车。一行命令。一个声音。就这样，从这里开始。项目是开源的。代码放在 GitHub 上面。谁都可以使用。谁也都可以参与改进。……也许你也想做些什么。那样的话，我会很高兴。总之。这就是 Bert-VITS2 C++ 运行时。一个让文字说话的。安静的。可靠的工具。如果你有想传达的话，它会帮你。……以上。","language":"AUTO","speaker_name":"tachibana_ja","length_scale":1.1}'
#!/usr/bin/env python3
"""Generate Chinese frontend data files from Bert-VITS2 Python references."""

from __future__ import annotations

import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def fetch_tone_sandhi_source() -> str:
    url = "https://raw.githubusercontent.com/fishaudio/Bert-VITS2/master/text/tone_sandhi.py"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8")


def extract_set_block(source: str, name: str) -> list[str]:
    pattern = rf"{name}\s*=\s*\{{([^}}]+)\}}"
    match = re.search(pattern, source, re.S)
    if not match:
        raise RuntimeError(f"failed to extract {name} from tone_sandhi.py")
    block = match.group(1)
    return [item.strip().strip('"').strip("'") for item in re.findall(r'"([^"]+)"', block)]


def write_lines(path: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in sorted(set(lines)):
            if line:
                f.write(line + "\n")


def generate_neural_word_lists() -> None:
    source = fetch_tone_sandhi_source()
    must = extract_set_block(source, "must_neural_tone_words")
    must_not = extract_set_block(source, "must_not_neural_tone_words")
    write_lines(os.path.join(SRC, "zh_neural_tone_words.txt"), must)
    write_lines(os.path.join(SRC, "zh_not_neural_tone_words.txt"), must_not)
    print(f"wrote {len(must)} neutral words, {len(must_not)} non-neutral words")


def generate_word_pinyin_table() -> None:
    try:
        import jieba
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:
        print(f"skip zh_word_pinyin.tsv: {exc}", file=sys.stderr)
        return

    jieba.initialize()
    words: set[str] = set()
    for word in jieba.dt.FREQ:
        if len(word) < 2:
            continue
        if word and all("\u4e00" <= ch <= "\u9fff" for ch in word):
            words.add(word)

    out_path = os.path.join(SRC, "zh_word_pinyin.tsv")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        written = 0
        for word in sorted(words, key=lambda w: (-len(w), w)):
            finals = lazy_pinyin(
                word,
                style=Style.TONE3,
                neutral_tone_with_five=True,
            )
            finals = [item.replace("ü", "v").replace("u:", "v") for item in finals]
            if len(finals) != len(word):
                continue
            f.write(f"{word}\t{'|'.join(finals)}\n")
            written += 1
    print(f"wrote {written} entries to {out_path}")


def main() -> None:
    generate_neural_word_lists()
    generate_word_pinyin_table()


if __name__ == "__main__":
    main()

from pypinyin import Style, pinyin


def main():
    with open("src/zh_pinyin.tsv", "w", encoding="utf-8", newline="\n") as f:
        for code in range(0x4E00, 0x9FFF + 1):
            ch = chr(code)
            py = pinyin(
                ch,
                style=Style.TONE3,
                neutral_tone_with_five=True,
                heteronym=False,
                errors="ignore",
            )
            if py and py[0]:
                value = py[0][0].replace("ü", "v").replace("u:", "v")
                if value:
                    f.write(f"{ch}\t{value}\n")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys


def resolve_config_path(config_arg, root):
    default = os.path.join(root, "Data", "config.json")
    if config_arg and os.path.isabs(config_arg) and os.path.isfile(config_arg):
        return config_arg
    if os.path.isfile(default):
        return default
    if config_arg and os.path.isfile(config_arg):
        return os.path.abspath(config_arg)
    raise FileNotFoundError(f"config not found: {config_arg or default}")


def find_project_root():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.dirname(here),
        os.path.dirname(os.path.dirname(here)),
    ]
    for root in candidates:
        if os.path.isfile(os.path.join(root, "Data", "config.json")):
            return root
        if os.path.isfile(os.path.join(root, "infer.py")):
            return root
    raise FileNotFoundError("could not locate Bert-VITS2 project root")


def export_frontend(text, language, config_path):
    root = find_project_root()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    import commons
    from text import cleaned_text_to_sequence
    from text.cleaner import clean_text
    from utils import get_hparams_from_file

    hps = get_hparams_from_file(config_path)
    norm_text, phone, tone, word2ph = clean_text(text, language)
    phone, tone, languages = cleaned_text_to_sequence(phone, tone, language)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        languages = commons.intersperse(languages, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_tokens = []
    if language == "ZH":
        bert_tokens = list(norm_text)
    elif language == "JP":
        from text.japanese import text2sep_kata

        sep, _, _ = text2sep_kata(norm_text)
        bert_tokens = list("".join(sep))
    elif language == "EN":
        bert_tokens = []

    return {
        "phones": [int(x) for x in phone],
        "tones": [int(x) for x in tone],
        "languages": [int(x) for x in languages],
        "word2ph": [int(x) for x in word2ph],
        "bert_tokens": bert_tokens,
        "norm_text": norm_text,
        "language": language,
        "length": len(phone),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export Bert-VITS2 frontend metadata for the C++ ONNX runner."
    )
    parser.add_argument("--text", default="")
    parser.add_argument("--text-file", default="")
    parser.add_argument("--language", required=True, choices=["ZH", "JP", "EN"])
    parser.add_argument("--config", default="")
    args = parser.parse_args()

    root = find_project_root()
    config_path = resolve_config_path(args.config, root)

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        raise SystemExit("--text or --text-file is required")

    result = export_frontend(text, args.language, config_path)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

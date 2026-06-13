import argparse
import json
import os
import sys

import torch


def write_f32(path, tensor):
    arr = tensor.detach().cpu().contiguous().numpy().astype("float32")
    arr.tofile(path)


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


def build_frontend(text, language, config_path, device, seed, with_bert):
    root = find_project_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    import commons
    from infer import get_text
    from text import cleaned_text_to_sequence
    from text.cleaner import clean_text
    from utils import get_hparams_from_file

    hps = get_hparams_from_file(config_path)
    if with_bert:
        bert, ja_bert, en_bert, phones, tones, languages = get_text(
            text,
            language,
            hps,
            device,
        )
        bert = bert.transpose(0, 1).cpu()
        ja_bert = ja_bert.transpose(0, 1).cpu()
        en_bert = en_bert.transpose(0, 1).cpu()
    else:
        norm_text, phone, tone, word2ph = clean_text(text, language)
        phone, tone, languages = cleaned_text_to_sequence(phone, tone, language)
        if hps.data.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            languages = commons.intersperse(languages, 0)
        phones = torch.LongTensor(phone)
        tones = torch.LongTensor(tone)
        languages = torch.LongTensor(languages)
        length = len(phone)
        bert = torch.randn(length, 1024, generator=torch.manual_seed(seed + 1))
        ja_bert = torch.randn(length, 1024, generator=torch.manual_seed(seed + 2))
        en_bert = torch.randn(length, 1024, generator=torch.manual_seed(seed + 3))

    return {
        "phones": [int(x) for x in phones.tolist()],
        "tones": [int(x) for x in tones.tolist()],
        "languages": [int(x) for x in languages.tolist()],
        "active_bert": language,
        "bert_zh": bert,
        "bert_jp": ja_bert,
        "bert_en": en_bert,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Bert-VITS2 multilingual frontend features for the C++ runner."
    )
    parser.add_argument("--text", default="")
    parser.add_argument("--text-file", default="")
    parser.add_argument("--language", required=True, choices=["ZH", "JP", "EN"])
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--no-bert", action="store_true")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    root = find_project_root()
    config_path = args.config or os.path.join(root, "Data", "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        raise SystemExit("--text or --text-file is required")

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    result = build_frontend(
        text,
        args.language,
        config_path,
        device,
        args.seed,
        with_bert=not args.no_bert,
    )

    prefix = os.path.join(args.out_dir, "frontend")
    write_f32(prefix + ".bert_zh.bin", result["bert_zh"])
    write_f32(prefix + ".bert_jp.bin", result["bert_jp"])
    write_f32(prefix + ".bert_en.bin", result["bert_en"])

    meta = {
        "phones": result["phones"],
        "tones": result["tones"],
        "languages": result["languages"],
        "length": len(result["phones"]),
        "active_bert": result["active_bert"],
        "bert_zh": prefix + ".bert_zh.bin",
        "bert_jp": prefix + ".bert_jp.bin",
        "bert_en": prefix + ".bert_en.bin",
    }
    json_path = prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(json.dumps({"json": json_path, "length": meta["length"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

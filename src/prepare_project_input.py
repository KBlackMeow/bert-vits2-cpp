import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def write_f32(path, tensor):
    arr = tensor.detach().cpu().contiguous().numpy().astype("float32")
    arr.tofile(path)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare original Bert-VITS2 frontend features for the C++ runner."
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="ZH", choices=["ZH", "JP", "EN"])
    parser.add_argument("--config", default="Data/config.json")
    parser.add_argument("--out-dir", default="build-cl/frontend")
    parser.add_argument("--name", default="input")
    args = parser.parse_args()

    from infer import get_text
    from utils import get_hparams_from_file

    os.makedirs(args.out_dir, exist_ok=True)
    hps = get_hparams_from_file(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bert, ja_bert, en_bert, phones, tones, languages = get_text(
        args.text,
        args.language,
        hps,
        device,
    )

    # Python frontend returns BERT as [1024, T]; exported ONNX expects [T, 1024].
    bert = bert.transpose(0, 1)
    ja_bert = ja_bert.transpose(0, 1)
    en_bert = en_bert.transpose(0, 1)

    prefix = os.path.join(args.out_dir, args.name)
    write_f32(prefix + ".bert_zh.bin", bert)
    write_f32(prefix + ".bert_jp.bin", ja_bert)
    write_f32(prefix + ".bert_en.bin", en_bert)

    meta = {
        "phones": [int(x) for x in phones],
        "tones": [int(x) for x in tones],
        "languages": [int(x) for x in languages],
        "phone_csv": ",".join(str(int(x)) for x in phones),
        "tone_csv": ",".join(str(int(x)) for x in tones),
        "language_csv": ",".join(str(int(x)) for x in languages),
        "length": len(phones),
        "bert_zh": prefix + ".bert_zh.bin",
        "bert_jp": prefix + ".bert_jp.bin",
        "bert_en": prefix + ".bert_en.bin",
    }

    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

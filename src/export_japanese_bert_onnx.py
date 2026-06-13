import argparse
import os

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class JapaneseBertFeature(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        return out.hidden_states[-3]


def main():
    parser = argparse.ArgumentParser(description="Export Japanese BERT feature model to ONNX.")
    parser.add_argument("--model-dir", default="bert/deberta-v2-large-japanese-char-wwm")
    parser.add_argument("--output", default="onnx/deberta-v2-large-japanese-char-wwm.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForMaskedLM.from_pretrained(args.model_dir).eval().cpu()
    wrapped = JapaneseBertFeature(model).eval().cpu()

    sample = tokenizer("こんにちは、世界。", return_tensors="pt")
    input_ids = sample["input_ids"].cpu()
    attention_mask = sample["attention_mask"].cpu()
    token_type_ids = sample.get("token_type_ids", torch.zeros_like(input_ids)).cpu()

    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask, token_type_ids),
        args.output,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["hidden"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "tokens"},
            "attention_mask": {0: "batch", 1: "tokens"},
            "token_type_ids": {0: "batch", 1: "tokens"},
            "hidden": {0: "batch", 1: "tokens"},
        },
        opset_version=args.opset,
    )
    print(f"exported {args.output}")


if __name__ == "__main__":
    main()

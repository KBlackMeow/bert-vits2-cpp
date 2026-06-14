"""Convert a HuggingFace BERT / DeBERTa-v2 / DeBERTa-v3 checkpoint to a
flat safetensors file consumable by `bv2_mlx_bert.cpp`.

The C++ runtime expects:

* All weights named under their HF state-dict path, with the leading
  module prefix stripped (so `bert.embeddings.*` → `embeddings.*`,
  `deberta.embeddings.*` → `embeddings.*`, etc.).
* A `safetensors` metadata block describing the architecture
  hyper-parameters (kind, hidden_size, num_hidden_layers, ...).
* Linear layers stored as PyTorch shapes [out, in] with companion `.bias`
  of shape [out] - which is the default for `nn.Linear`. No conversion
  needed.

Three checkpoints are supported by Bert-VITS2 v2.3:

    bert/chinese-roberta-wwm-ext-large            -> kind=bert
    bert/deberta-v2-large-japanese-char-wwm       -> kind=deberta_v2
    bert/deberta-v3-large                         -> kind=deberta_v2 (v3 architecture is the same as v2)

Usage:
    python src/convert_bert_to_mlx.py bert/chinese-roberta-wwm-ext-large \
        -o onnx/bert/chinese-roberta-wwm-ext-large.safetensors
    python src/convert_bert_to_mlx.py bert/deberta-v2-large-japanese-char-wwm \
        -o onnx/bert/deberta-v2-large-japanese-char-wwm.safetensors
    python src/convert_bert_to_mlx.py bert/deberta-v3-large \
        -o onnx/bert/deberta-v3-large.safetensors

Optional flags:
    --print-keys      print every output tensor name + shape (sanity check)
    --feature-layer N override `hidden_states[N]` slice (default: -3, matching
                      Bert-VITS2 v2.3 ONNX exports)
    --keep-lm-head    keep the masked-LM head (we don't need it - default drops it)
    --strict-prefix   require all incoming keys to start with `bert.` /
                      `roberta.` / `deberta.`; otherwise, accept un-prefixed.

The script reads HF model directories that contain either
`pytorch_model.bin` or `model.safetensors` (potentially sharded via
`*.index.json`). Tokenizer files are not touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover
    print(f"ERROR: torch is required ({exc})", file=sys.stderr)
    sys.exit(2)

try:
    from safetensors.torch import save_file as st_save_file
except Exception as exc:  # pragma: no cover
    print("ERROR: pip install safetensors", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Loading state dicts
# ---------------------------------------------------------------------------


def _load_single(path: str) -> Dict[str, "torch.Tensor"]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_state_dict(model_dir: str) -> Dict[str, "torch.Tensor"]:
    """Read either the single-file or sharded HF format."""
    candidates = [
        ("model.safetensors", False),
        ("pytorch_model.bin",  False),
        ("model.safetensors.index.json", True),
        ("pytorch_model.bin.index.json", True),
    ]
    for name, is_index in candidates:
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            continue
        if is_index:
            with open(path, "r", encoding="utf-8") as fp:
                index = json.load(fp)
            sd: Dict[str, "torch.Tensor"] = {}
            for shard in sorted(set(index["weight_map"].values())):
                shard_path = os.path.join(model_dir, shard)
                sd.update(_load_single(shard_path))
            return sd
        return _load_single(path)
    raise FileNotFoundError(
        f"no model.safetensors / pytorch_model.bin (sharded or single) under {model_dir}")


def load_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {path}")
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------


# Drop-prefixes we strip so the C++ side sees a uniform key namespace
# regardless of which top-level model class produced the checkpoint.
PREFIXES_TO_STRIP = (
    "bert.",
    "roberta.",
    "deberta.",
    "model.",       # some HF checkpoints
)

# The MaskedLM head & pooler are not needed for hidden_states[-3] inference.
LM_HEAD_PREFIXES = (
    "cls.",         # BertForMaskedLM head
    "lm_predictions.",
    "lm_head.",
    "mask_predictions.",
    "pooler.",      # not used by Bert-VITS2 (we slice hidden states directly)
)


def normalise_key(key: str) -> Optional[str]:
    for p in PREFIXES_TO_STRIP:
        if key.startswith(p):
            return key[len(p):]
    return key


def is_lm_head(key: str) -> bool:
    return any(key.startswith(p) for p in LM_HEAD_PREFIXES)


# ---------------------------------------------------------------------------
# Config -> metadata
# ---------------------------------------------------------------------------


def _kind_of(config: dict) -> str:
    model_type = (config.get("model_type") or "").lower()
    if model_type in ("deberta-v2", "deberta_v2", "deberta-v3", "deberta_v3"):
        return "deberta_v2"  # v3 reuses the v2 architecture
    if model_type in ("deberta",):
        # v1 is different but Bert-VITS2 doesn't use it; refuse.
        raise ValueError("DeBERTa v1 is not supported (Bert-VITS2 uses v2/v3).")
    return "bert"


def build_metadata(config: dict, feature_layer: int) -> Dict[str, str]:
    kind = _kind_of(config)
    md: Dict[str, str] = {"kind": kind}

    def put(key: str, value, default=None):
        if value is None:
            value = default
        md[key] = str(value)

    put("hidden_size",             config.get("hidden_size"),                 1024)
    put("num_hidden_layers",       config.get("num_hidden_layers"),           24)
    put("num_attention_heads",     config.get("num_attention_heads"),         16)
    put("intermediate_size",       config.get("intermediate_size"),           4096)
    put("max_position_embeddings", config.get("max_position_embeddings"),     512)
    put("type_vocab_size",         config.get("type_vocab_size"),             2)
    put("vocab_size",              config.get("vocab_size"),                  30522)
    md["layer_norm_eps"] = str(config.get("layer_norm_eps", 1e-12))
    md["feature_layer"]  = str(feature_layer)

    if kind == "deberta_v2":
        md["relative_attention"]      = "true" if config.get("relative_attention", True) else "false"
        md["position_buckets"]        = str(config.get("position_buckets", 256))
        md["max_relative_positions"]  = str(config.get("max_relative_positions", 0) or 0)
        md["share_att_key"]           = "true" if config.get("share_att_key", False) else "false"
        md["position_biased_input"]   = "true" if config.get("position_biased_input", False) else "false"
        norm_rel = (config.get("norm_rel_ebd") or "")
        md["norm_rel_ebd_layer_norm"] = "true" if "layer_norm" in norm_rel else "false"
        pos_att = config.get("pos_att_type") or ["p2c", "c2p"]
        if isinstance(pos_att, str):
            pos_att = [s.strip() for s in pos_att.split("|") if s.strip()]
        md["pos_att_p2c"] = "true" if "p2c" in pos_att else "false"
        md["pos_att_c2p"] = "true" if "c2p" in pos_att else "false"

    return md


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", help="HuggingFace model directory")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--feature-layer", type=int, default=-3,
                        help="hidden_states index (default -3, matches Bert-VITS2 ONNX)")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                        help="float dtype for stored weights (default fp32). "
                             "fp16 halves the safetensors size; the C++ runtime "
                             "auto-casts the final hidden state back to fp32.")
    parser.add_argument("--keep-lm-head", action="store_true")
    parser.add_argument("--print-keys", action="store_true")
    args = parser.parse_args()

    config = load_config(args.model_dir)
    print(f"loading state dict from {args.model_dir}", file=sys.stderr)
    sd = load_state_dict(args.model_dir)

    metadata = build_metadata(config, args.feature_layer)
    metadata["weight_dtype"] = args.dtype   # informational only
    target_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]

    out: Dict[str, "torch.Tensor"] = {}
    skipped = 0
    for raw_key, tensor in sd.items():
        if not args.keep_lm_head and is_lm_head(raw_key):
            skipped += 1
            continue
        new_key = normalise_key(raw_key)
        if new_key is None:
            skipped += 1
            continue
        # Some HF checkpoints store running stats / position_ids buffers we
        # don't need.
        if new_key.endswith(".position_ids"):
            skipped += 1
            continue
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.int32)
        else:
            # LayerNorm gamma/beta and embedding norms are tiny but matter for
            # numeric stability. Conventional practice for BERT inference is
            # to keep them in the requested dtype too -- HF and llama.cpp do
            # this. Re-cast unconditionally so the on-disk dtype is uniform.
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
        out[new_key] = tensor.contiguous()

    print(
        f"keeping {len(out)} tensors, skipped {skipped} "
        f"(LM head / pooler / position_ids), dtype={args.dtype}",
        file=sys.stderr)

    if args.print_keys:
        for k in sorted(out):
            print(f"{k:80s} {tuple(out[k].shape)} {out[k].dtype}")

    print(f"writing -> {args.output}  (kind={metadata.get('kind')})", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    st_save_file(out, args.output, metadata=metadata)


if __name__ == "__main__":
    main()

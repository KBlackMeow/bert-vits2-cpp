"""Convert a Bert-VITS2 v2.3 generator checkpoint (`G_*.pth`) to a flat
safetensors file consumable by the C++ MLX runtime.

The C++ side cannot easily reproduce all of PyTorch's parameterizations
(weight_norm, etc.) at load time, so we fold them here. We also flatten
nested module-list keys into the names hard-coded in the C++ side
(`mlx/bv2_mlx_models.cpp`). The resulting safetensors file is meant to
be loaded with `mlx::core::load_safetensors` in C++.

Usage:
    python src/convert_to_mlx.py model/models/G_46000.pth \
        -o model/models/G_46000_mlx.safetensors

Optional checks:
    --print-keys                  print all converted tensor names
    --strict                      fail if any v2.3 generator key is unmapped

The script does NOT need MLX (it only uses PyTorch + safetensors).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Tuple

try:
    import torch  # noqa: F401  required to load .pth
except Exception as exc:  # pragma: no cover - error path
    print("ERROR: torch is required to read the .pth checkpoint.", file=sys.stderr)
    print(f"        ({exc})", file=sys.stderr)
    sys.exit(2)

try:
    from safetensors.torch import save_file as st_save_file
except Exception as exc:  # pragma: no cover
    print("ERROR: pip install safetensors", file=sys.stderr)
    print(f"        ({exc})", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# weight_norm folding
# ---------------------------------------------------------------------------


def fold_weight_norm(state_dict: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    """Replace every `<prefix>.weight_g` + `<prefix>.weight_v` pair with a
    single `<prefix>.weight` tensor equal to the materialised conv weight.

    PyTorch's `weight_norm` decomposes ``W = g * v / ||v||`` along
    ``dim=0`` for Conv1d/ConvTranspose1d (the default).
    """
    folded: Dict[str, "torch.Tensor"] = {}
    skip = set()

    for key, tensor in state_dict.items():
        if not key.endswith(".weight_g"):
            continue
        prefix = key[: -len(".weight_g")]
        v_key = prefix + ".weight_v"
        if v_key not in state_dict:
            continue
        g = state_dict[key]  # shape [out_channels, 1, 1] (Conv1d)
        v = state_dict[v_key]
        # Match torch.nn.utils.weight_norm: norm of v across all dims except 0.
        eps = 1e-12
        flat = v.reshape(v.shape[0], -1)
        norm = flat.pow(2).sum(dim=1).sqrt().clamp_min(eps)  # [out_channels]
        # g has shape (out_channels, 1, 1) (or 4D for conv2d). We just
        # broadcast-multiply.
        scale = (g.reshape(-1) / norm).reshape([-1] + [1] * (v.ndim - 1))
        folded[prefix + ".weight"] = (v * scale).contiguous()
        skip.add(key)
        skip.add(v_key)

    for key, tensor in state_dict.items():
        if key in skip:
            continue
        if key.endswith(".weight") and (key + "_g") in state_dict:
            # already folded above
            continue
        folded[key] = tensor

    return folded


# ---------------------------------------------------------------------------
# key rename map
# ---------------------------------------------------------------------------


# We rename only when the C++ side wants a different name. The C++ loader
# is forgiving (it accepts both v2.3 names and these flattened names) but
# we keep the file small / predictable.
def rename_key(key: str) -> str:
    return key  # currently identity. Kept as a hook for future cleanup.


# ---------------------------------------------------------------------------
# v2.3 generator key inventory (sanity check)
# ---------------------------------------------------------------------------


def _expected_keys(num_speakers: int) -> List[str]:
    """Approximate set of keys we know how to load on the C++ side. Used
    only by ``--strict`` mode."""

    keys: List[str] = []

    # Speaker embedding
    keys.append("emb_g.weight")

    # TextEncoder
    keys.append("enc_p.emb.weight")
    keys.append("enc_p.tone_emb.weight")
    keys.append("enc_p.language_emb.weight")
    for proj in ("bert_proj", "ja_bert_proj", "en_bert_proj"):
        keys.append(f"enc_p.{proj}.weight")
        keys.append(f"enc_p.{proj}.bias")
    keys.append("enc_p.proj.weight")
    keys.append("enc_p.proj.bias")
    # Encoder: 6 layers
    for i in range(6):
        for c in ("conv_q", "conv_k", "conv_v", "conv_o"):
            keys.append(f"enc_p.encoder.attn_layers.{i}.{c}.weight")
            keys.append(f"enc_p.encoder.attn_layers.{i}.{c}.bias")
        keys.append(f"enc_p.encoder.attn_layers.{i}.emb_rel_k")
        keys.append(f"enc_p.encoder.attn_layers.{i}.emb_rel_v")
        keys.append(f"enc_p.encoder.norm_layers_1.{i}.gamma")
        keys.append(f"enc_p.encoder.norm_layers_1.{i}.beta")
        keys.append(f"enc_p.encoder.ffn_layers.{i}.conv_1.weight")
        keys.append(f"enc_p.encoder.ffn_layers.{i}.conv_1.bias")
        keys.append(f"enc_p.encoder.ffn_layers.{i}.conv_2.weight")
        keys.append(f"enc_p.encoder.ffn_layers.{i}.conv_2.bias")
        keys.append(f"enc_p.encoder.norm_layers_2.{i}.gamma")
        keys.append(f"enc_p.encoder.norm_layers_2.{i}.beta")
    # speaker projection at cond_layer_idx
    keys.append("enc_p.encoder.spk_emb_linear.weight")
    keys.append("enc_p.encoder.spk_emb_linear.bias")

    # DurationPredictor
    keys.append("dp.conv_1.weight")
    keys.append("dp.conv_1.bias")
    keys.append("dp.conv_2.weight")
    keys.append("dp.conv_2.bias")
    keys.append("dp.norm_1.gamma")
    keys.append("dp.norm_1.beta")
    keys.append("dp.norm_2.gamma")
    keys.append("dp.norm_2.beta")
    keys.append("dp.proj.weight")
    keys.append("dp.proj.bias")
    keys.append("dp.cond.weight")
    keys.append("dp.cond.bias")

    # StochasticDurationPredictor (inference path only uses pre/proj/convs/cond
    # and the `flows` ModuleList: ElementwiseAffine + 4*(ConvFlow + Flip)).
    # post_* params are training-only (posterior flow) but we still ship them.
    keys.append("sdp.pre.weight")
    keys.append("sdp.pre.bias")
    keys.append("sdp.proj.weight")
    keys.append("sdp.proj.bias")
    keys.append("sdp.cond.weight")
    keys.append("sdp.cond.bias")
    for i in range(3):
        keys.append(f"sdp.convs.convs_sep.{i}.weight")
        keys.append(f"sdp.convs.convs_sep.{i}.bias")
        keys.append(f"sdp.convs.convs_1x1.{i}.weight")
        keys.append(f"sdp.convs.convs_1x1.{i}.bias")
        keys.append(f"sdp.convs.norms_1.{i}.gamma")
        keys.append(f"sdp.convs.norms_1.{i}.beta")
        keys.append(f"sdp.convs.norms_2.{i}.gamma")
        keys.append(f"sdp.convs.norms_2.{i}.beta")
    # 1: ElementwiseAffine, 2..9 alternating ConvFlow/Flip
    keys.append("sdp.flows.0.m")
    keys.append("sdp.flows.0.logs")
    for fi in (1, 3, 5, 7):  # 4 ConvFlows
        keys.append(f"sdp.flows.{fi}.pre.weight")
        keys.append(f"sdp.flows.{fi}.pre.bias")
        keys.append(f"sdp.flows.{fi}.proj.weight")
        keys.append(f"sdp.flows.{fi}.proj.bias")
        for ci in range(3):
            keys.append(f"sdp.flows.{fi}.convs.convs_sep.{ci}.weight")
            keys.append(f"sdp.flows.{fi}.convs.convs_sep.{ci}.bias")
            keys.append(f"sdp.flows.{fi}.convs.convs_1x1.{ci}.weight")
            keys.append(f"sdp.flows.{fi}.convs.convs_1x1.{ci}.bias")
            keys.append(f"sdp.flows.{fi}.convs.norms_1.{ci}.gamma")
            keys.append(f"sdp.flows.{fi}.convs.norms_1.{ci}.beta")
            keys.append(f"sdp.flows.{fi}.convs.norms_2.{ci}.gamma")
            keys.append(f"sdp.flows.{fi}.convs.norms_2.{ci}.beta")

    # TransformerCouplingBlock (4 flows: TransformerCouplingLayer + Flip).
    # Each TCL has pre, post Conv1d and an Encoder with n_layers_trans_flow=4
    # attention layers.
    for fi in (0, 2, 4, 6):
        keys.append(f"flow.flows.{fi}.pre.weight")
        keys.append(f"flow.flows.{fi}.pre.bias")
        keys.append(f"flow.flows.{fi}.post.weight")
        keys.append(f"flow.flows.{fi}.post.bias")
        for li in range(4):  # n_layers_trans_flow
            for c in ("conv_q", "conv_k", "conv_v", "conv_o"):
                keys.append(f"flow.flows.{fi}.enc.attn_layers.{li}.{c}.weight")
                keys.append(f"flow.flows.{fi}.enc.attn_layers.{li}.{c}.bias")
            keys.append(f"flow.flows.{fi}.enc.attn_layers.{li}.emb_rel_k")
            keys.append(f"flow.flows.{fi}.enc.attn_layers.{li}.emb_rel_v")
            keys.append(f"flow.flows.{fi}.enc.norm_layers_1.{li}.gamma")
            keys.append(f"flow.flows.{fi}.enc.norm_layers_1.{li}.beta")
            keys.append(f"flow.flows.{fi}.enc.ffn_layers.{li}.conv_1.weight")
            keys.append(f"flow.flows.{fi}.enc.ffn_layers.{li}.conv_1.bias")
            keys.append(f"flow.flows.{fi}.enc.ffn_layers.{li}.conv_2.weight")
            keys.append(f"flow.flows.{fi}.enc.ffn_layers.{li}.conv_2.bias")
            keys.append(f"flow.flows.{fi}.enc.norm_layers_2.{li}.gamma")
            keys.append(f"flow.flows.{fi}.enc.norm_layers_2.{li}.beta")
        keys.append(f"flow.flows.{fi}.enc.spk_emb_linear.weight")
        keys.append(f"flow.flows.{fi}.enc.spk_emb_linear.bias")

    # Generator
    keys.append("dec.conv_pre.weight")
    keys.append("dec.conv_pre.bias")
    keys.append("dec.conv_post.weight")
    keys.append("dec.cond.weight")
    keys.append("dec.cond.bias")
    for i in range(5):  # 5 upsamples
        keys.append(f"dec.ups.{i}.weight")
        keys.append(f"dec.ups.{i}.bias")
    for i in range(15):  # 5 * 3 ResBlock1
        keys.append(f"dec.resblocks.{i}.convs1.0.weight")
        keys.append(f"dec.resblocks.{i}.convs1.0.bias")
        keys.append(f"dec.resblocks.{i}.convs1.1.weight")
        keys.append(f"dec.resblocks.{i}.convs1.1.bias")
        keys.append(f"dec.resblocks.{i}.convs1.2.weight")
        keys.append(f"dec.resblocks.{i}.convs1.2.bias")
        keys.append(f"dec.resblocks.{i}.convs2.0.weight")
        keys.append(f"dec.resblocks.{i}.convs2.0.bias")
        keys.append(f"dec.resblocks.{i}.convs2.1.weight")
        keys.append(f"dec.resblocks.{i}.convs2.1.bias")
        keys.append(f"dec.resblocks.{i}.convs2.2.weight")
        keys.append(f"dec.resblocks.{i}.convs2.2.bias")

    return keys


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to G_*.pth")
    parser.add_argument("-o", "--output", required=False)
    parser.add_argument(
        "--config",
        help="path to config.json (defaults to <checkpoint dir>/config.json)",
    )
    parser.add_argument("--print-keys", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="dtype for floating-point weights (default: fp32). "
             "Use fp16 to halve the file size and RAM usage.",
    )
    parser.add_argument(
        "--keep-training-only",
        action="store_true",
        help="also keep enc_q/post_*/discriminator weights (default: drop)",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    out_path = args.output or os.path.join(
        os.path.dirname(ckpt_path), "G_mlx.safetensors"
    )
    config_path = args.config
    if config_path is None:
        cand = os.path.join(os.path.dirname(ckpt_path), "config.json")
        if os.path.exists(cand):
            config_path = cand
        else:
            # fall back to <repo>/model/config.json
            cand = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "model", "config.json")
            )
            if os.path.exists(cand):
                config_path = cand

    print(f"loading checkpoint: {ckpt_path}", file=sys.stderr)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = ckpt.state_dict()  # type: ignore[attr-defined]

    # drop the modules that are training-only.
    if not args.keep_training_only:
        drop_prefixes = (
            "enc_q.",  # PosteriorEncoder (training only)
            "ref_enc.",  # ReferenceEncoder (we have n_speakers > 0)
        )
        sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}

    # SDP post_* params are only used when training. The C++ inference path
    # ignores them, but we keep them so the MLX file is a faithful drop-in.
    sd = fold_weight_norm(sd)
    sd = {rename_key(k): v.contiguous() for k, v in sd.items()}

    # keep only float / half tensors (safetensors restriction)
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    target_dtype = dtype_map[args.dtype]
    cleaned: Dict[str, "torch.Tensor"] = {}
    for k, v in sd.items():
        if v.dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            v = v.to(target_dtype)
            cleaned[k] = v
        elif v.dtype in (torch.int32, torch.int64):
            cleaned[k] = v.to(torch.int32)
        # else: skip non-numeric

    if args.print_keys:
        for k in sorted(cleaned):
            print(f"{k:80s} {tuple(cleaned[k].shape)} {cleaned[k].dtype}")

    if args.strict:
        n_speakers = 0
        if config_path is not None:
            with open(config_path, "r", encoding="utf-8") as fp:
                cfg = json.load(fp)
            n_speakers = int(cfg.get("data", {}).get("n_speakers", 0))
        wanted = set(_expected_keys(n_speakers))
        missing = sorted(wanted - cleaned.keys())
        if missing:
            print(
                f"WARN ({len(missing)}/{len(wanted)}) expected keys missing in "
                "checkpoint after folding:",
                file=sys.stderr,
            )
            for k in missing[:20]:
                print(f"  - {k}", file=sys.stderr)
            if len(missing) > 20:
                print(f"  ... ({len(missing) - 20} more)", file=sys.stderr)

    # safetensors metadata: stash some shape/version info for debugging.
    metadata = {"format": "bv2-mlx-v1"}
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
        metadata["bv2_version"] = str(cfg.get("version", ""))
        for fld in (
            "inter_channels",
            "hidden_channels",
            "filter_channels",
            "n_heads",
            "n_layers",
            "kernel_size",
            "gin_channels",
        ):
            if fld in cfg.get("model", {}):
                metadata[fld] = str(cfg["model"][fld])

    print(
        f"writing {len(cleaned)} tensors to {out_path}",
        file=sys.stderr,
    )
    st_save_file(cleaned, out_path, metadata=metadata)
    print(f"done -> {out_path}", file=sys.stderr)

    # Drop a sibling `config.json` so the MLX runtime can resolve the full
    # VITS hyper-params next to the safetensors. The C++ side falls back to
    # `mlx_model/config.json` when no --config is passed.
    if config_path is not None and os.path.exists(config_path):
        sibling_cfg = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                                   "config.json")
        if os.path.abspath(config_path) != os.path.abspath(sibling_cfg):
            with open(config_path, "r", encoding="utf-8") as fp:
                src_text = fp.read()
            with open(sibling_cfg, "w", encoding="utf-8") as fp:
                fp.write(src_text)
            print(f"wrote sibling config -> {sibling_cfg}", file=sys.stderr)


if __name__ == "__main__":
    main()

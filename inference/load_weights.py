"""
inference/load_weights.py — load pretrained GPT-2 weights from HuggingFace.
Same assign_check + weight mapping as Notebook 5.

Usage:
    sair ui --hf gpt2-medium
    sair generate "..." --hf gpt2

    from inference.load_weights import load_from_hf
    model, config = load_from_hf("gpt2-medium")
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from config import HF_MODELS, CKPT_DIR


def _assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return nn.Parameter(right.clone().detach())


def load_from_hf(variant: str = "gpt2", cache_dir: str = str(CKPT_DIR)):
    """
    Download a pretrained GPT-2 variant and load it into our GPTModel.

    variant : "gpt2" | "gpt2-medium" | "gpt2-large" | "gpt2-xl"
    """
    from transformers import GPT2Model as HF_GPT2
    from model.gpt import GPTModel

    if variant not in HF_MODELS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(HF_MODELS)}")

    entry  = HF_MODELS[variant]
    config = entry["config"]

    print(f"Downloading {variant} from HuggingFace...")
    hf    = HF_GPT2.from_pretrained(entry["repo"], cache_dir=cache_dir)
    hf.eval()
    d = hf.state_dict()

    model = GPTModel(config)

    # embeddings
    model.pos_emb.weight = _assign(model.pos_emb.weight, d["wpe.weight"])
    model.tok_emb.weight = _assign(model.tok_emb.weight, d["wte.weight"])

    # transformer blocks
    for b in range(config["n_layers"]):
        # fused QKV → split into separate Q, K, V
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"],   3, axis=-1)

        model.trf_blocks[b].att.W_query.weight = _assign(model.trf_blocks[b].att.W_query.weight, q_w.T)
        model.trf_blocks[b].att.W_key.weight   = _assign(model.trf_blocks[b].att.W_key.weight,   k_w.T)
        model.trf_blocks[b].att.W_value.weight = _assign(model.trf_blocks[b].att.W_value.weight, v_w.T)
        model.trf_blocks[b].att.W_query.bias   = _assign(model.trf_blocks[b].att.W_query.bias,   q_b)
        model.trf_blocks[b].att.W_key.bias     = _assign(model.trf_blocks[b].att.W_key.bias,     k_b)
        model.trf_blocks[b].att.W_value.bias   = _assign(model.trf_blocks[b].att.W_value.bias,   v_b)

        model.trf_blocks[b].att.out_proj.weight = _assign(model.trf_blocks[b].att.out_proj.weight, d[f"h.{b}.attn.c_proj.weight"].T)
        model.trf_blocks[b].att.out_proj.bias   = _assign(model.trf_blocks[b].att.out_proj.bias,   d[f"h.{b}.attn.c_proj.bias"])

        model.trf_blocks[b].ff.layers[0].weight = _assign(model.trf_blocks[b].ff.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T)
        model.trf_blocks[b].ff.layers[0].bias   = _assign(model.trf_blocks[b].ff.layers[0].bias,   d[f"h.{b}.mlp.c_fc.bias"])
        model.trf_blocks[b].ff.layers[2].weight = _assign(model.trf_blocks[b].ff.layers[2].weight, d[f"h.{b}.mlp.c_proj.weight"].T)
        model.trf_blocks[b].ff.layers[2].bias   = _assign(model.trf_blocks[b].ff.layers[2].bias,   d[f"h.{b}.mlp.c_proj.bias"])

        model.trf_blocks[b].norm1.scale = _assign(model.trf_blocks[b].norm1.scale, d[f"h.{b}.ln_1.weight"])
        model.trf_blocks[b].norm1.shift = _assign(model.trf_blocks[b].norm1.shift, d[f"h.{b}.ln_1.bias"])
        model.trf_blocks[b].norm2.scale = _assign(model.trf_blocks[b].norm2.scale, d[f"h.{b}.ln_2.weight"])
        model.trf_blocks[b].norm2.shift = _assign(model.trf_blocks[b].norm2.shift, d[f"h.{b}.ln_2.bias"])

    # final norm + output head (weight-tied with tok_emb)
    model.final_norm.scale = _assign(model.final_norm.scale, d["ln_f.weight"])
    model.final_norm.shift = _assign(model.final_norm.shift, d["ln_f.bias"])
    model.out_head.weight  = _assign(model.out_head.weight,  d["wte.weight"])

    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {variant} — {params:,} parameters")
    return model, config

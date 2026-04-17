"""
inference/generate.py — generateV0→V3 from Notebook 5.
Each function is standalone and importable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import tiktoken


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode(prompt, tokenizer, device):
    return torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long
    ).unsqueeze(0).to(device)


def _decode(idx, tokenizer):
    return tokenizer.decode(idx.squeeze(0).tolist())


def apply_top_k(logits, k):
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits, p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # keep the token that first crosses p; remove everything after
    to_remove = (cumulative - sorted_probs) > p
    remove = torch.zeros_like(logits, dtype=torch.bool)
    remove.scatter_(dim=-1, index=sorted_idx, src=to_remove)
    return logits.masked_fill(remove, float("-inf"))


# ── V0: pure greedy ───────────────────────────────────────────────────────────

def generateV0(model, prompt, max_new_tokens, context_size, tokenizer, device):
    idx = _encode(prompt, tokenizer, device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(idx[:, -context_size:])[:, -1, :]
        idx = torch.cat([idx, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)
    return _decode(idx, tokenizer)


# ── V1: + temperature ─────────────────────────────────────────────────────────

def generateV1(model, prompt, max_new_tokens, context_size, tokenizer, device,
               temperature=1.0):
    idx = _encode(prompt, tokenizer, device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(idx[:, -context_size:])[:, -1, :]
        if temperature == 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs    = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
    return _decode(idx, tokenizer)


# ── V2: + top-k + top-p + EOS ─────────────────────────────────────────────────

def generateV2(model, prompt, max_new_tokens, context_size, tokenizer, device,
               temperature=1.0, top_k=None, top_p=None, eos=None):
    idx = _encode(prompt, tokenizer, device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(idx[:, -context_size:])[:, -1, :]
        if top_k is not None:
            logits = apply_top_k(logits, top_k)
        if top_p is not None:
            logits = apply_top_p(logits, top_p)
        if temperature == 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs    = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        if eos is not None and idx_next.item() == eos:
            break
        idx = torch.cat([idx, idx_next], dim=1)
    return _decode(idx, tokenizer)


# ── V3: + beam search ─────────────────────────────────────────────────────────

def _beam_sample(logits, beams, method, temperature, top_k, top_p):
    if method == "greedy":
        return torch.topk(logits, beams).indices
    if method == "top_k":
        logits = apply_top_k(logits.clone(), top_k)
    elif method == "nucleus":
        logits = apply_top_p(logits.clone(), top_p)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, beams)


def generateV3(model, prompt, max_new_tokens, context_size, tokenizer, device,
               beams=3, method="greedy", temperature=0.7, top_k=50, top_p=0.9):
    input_ids = _encode(prompt, tokenizer, device)
    sequences = [(input_ids, 0.0)]

    for _ in range(max_new_tokens):
        candidates = []
        for seq, score in sequences:
            with torch.no_grad():
                logits = model(seq[:, -context_size:])[:, -1, :].squeeze(0)
            token_ids = _beam_sample(logits, beams, method, temperature, top_k, top_p)
            for tid in token_ids:
                log_p     = torch.log(F.softmax(logits, dim=-1)[tid.item()]).item()
                new_seq   = torch.cat([seq, tid.view(1, 1)], dim=-1)
                candidates.append((new_seq, score + log_p))
        sequences = sorted(candidates, key=lambda x: x[1], reverse=True)[:beams]

    return _decode(sequences[0][0], tokenizer)


# ── Default alias used by trainer and UI ──────────────────────────────────────

def generate(model, prompt, max_new_tokens, context_size, tokenizer, device,
             temperature=0.8, top_k=50, top_p=0.9, beams=1, method="nucleus"):
    if beams > 1:
        return generateV3(model, prompt, max_new_tokens, context_size,
                          tokenizer, device, beams=beams, method=method,
                          temperature=temperature, top_k=top_k, top_p=top_p)
    return generateV2(model, prompt, max_new_tokens, context_size,
                      tokenizer, device, temperature=temperature,
                      top_k=top_k, top_p=top_p)


# ── Load latest checkpoint ────────────────────────────────────────────────────

def load_model(config, ckpt_dir, device):
    from model.gpt import GPTModel
    from config import CKPT_DIR
    import re

    ckpt_dir = Path(ckpt_dir)
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"),
                         key=lambda p: int(re.search(r"\d+", p.stem).group()))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir} — train first.")

    latest = checkpoints[-1]
    print(f"Loading {latest}")
    model = GPTModel(config).to(device)
    ckpt  = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

"""
config.py — the only file you need to touch.

Change MODEL_PRESET, drop files in data/raw/, tweak training params.
Everything else reads from here.

QUICK GUIDE
───────────
• Train your own model  → pick a preset or define a CUSTOM config below
• Load pretrained GPT-2 → use --hf flag: sair ui --hf gpt2-medium
• All architectures (custom and official GPT-2) live in MODELS
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
DATA_RAW  = ROOT / "data" / "raw"        # drop your .txt / .pdf files here
DATA_DIR  = ROOT / "data" / "processed"  # tokenized .bin files land here
CKPT_DIR  = ROOT / "checkpoints"         # model checkpoints

# ── All model architectures ────────────────────────────────────────────────────
#
#  SCRATCH PRESETS  — train from scratch on your own data
#  ┌─────────────┬───────────┬────────────┬──────────────────────────────────┐
#  │ preset      │  params   │ context    │ recommended hardware              │
#  ├─────────────┼───────────┼────────────┼──────────────────────────────────┤
#  │ tiny        │  ~10 M    │  256 tok   │ CPU — fast iteration / debugging  │
#  │ small       │  ~50 M    │  512 tok   │ laptop GPU                        │
#  │ medium      │  ~163 M   │  1024 tok  │ single GPU (8 GB+)                │
#  └─────────────┴───────────┴────────────┴──────────────────────────────────┘
#
#  OFFICIAL GPT-2 ARCHITECTURES  — load pretrained weights with --hf
#  ┌─────────────┬───────────┬────────────┬──────────────────────────────────┐
#  │ preset      │  params   │ layers/dim │ notes                            │
#  ├─────────────┼───────────┼────────────┼──────────────────────────────────┤
#  │ gpt2-124m   │  124 M    │ 12 / 768   │ GPT-2 small (original)           │
#  │ gpt2-355m   │  355 M    │ 24 / 1024  │ GPT-2 medium                     │
#  │ gpt2-774m   │  774 M    │ 36 / 1280  │ GPT-2 large                      │
#  │ gpt2-1558m  │  1558 M   │ 48 / 1600  │ GPT-2 XL                         │
#  └─────────────┴───────────┴────────────┴──────────────────────────────────┘

MODELS = {

    # ── Scratch presets ────────────────────────────────────────────────────────
    "tiny": {
        "vocab_size": 50257, "context_length": 256,
        "emb_dim": 256, "n_heads": 4, "n_layers": 4,
        "drop_rate": 0.1, "qkv_bias": False,
    },
    "small": {
        "vocab_size": 50257, "context_length": 512,
        "emb_dim": 512, "n_heads": 8, "n_layers": 6,
        "drop_rate": 0.1, "qkv_bias": False,
    },
    "medium": {
        "vocab_size": 50257, "context_length": 1024,
        "emb_dim": 768, "n_heads": 12, "n_layers": 12,
        "drop_rate": 0.1, "qkv_bias": False,
    },

    # ── Official GPT-2 architectures (exact — use with --hf flag) ──────────────
    "gpt2-124m": {
        "vocab_size": 50257, "context_length": 1024,
        "emb_dim": 768, "n_heads": 12, "n_layers": 12,
        "drop_rate": 0.0, "qkv_bias": True,
    },
    "gpt2-355m": {
        "vocab_size": 50257, "context_length": 1024,
        "emb_dim": 1024, "n_heads": 16, "n_layers": 24,
        "drop_rate": 0.0, "qkv_bias": True,
    },
    "gpt2-774m": {
        "vocab_size": 50257, "context_length": 1024,
        "emb_dim": 1280, "n_heads": 20, "n_layers": 36,
        "drop_rate": 0.0, "qkv_bias": True,
    },
    "gpt2-1558m": {
        "vocab_size": 50257, "context_length": 1024,
        "emb_dim": 1600, "n_heads": 25, "n_layers": 48,
        "drop_rate": 0.0, "qkv_bias": True,
    },

    # ── Custom — define your own architecture ──────────────────────────────────
    # Rules:
    #   • emb_dim must be divisible by n_heads
    #   • larger context_length = more memory
    #   • qkv_bias=True matches GPT-2; False is fine for training from scratch
    "custom": {
        "vocab_size": 50257, "context_length": 512,
        "emb_dim": 384, "n_heads": 6, "n_layers": 6,
        "drop_rate": 0.1, "qkv_bias": False,
    },
}

# ── Active model ───────────────────────────────────────────────────────────────
# Change this one line to switch architecture.
# Scratch training:  "tiny" | "small" | "medium" | "custom"
# Load pretrained:   use --hf flag instead (see below)
MODEL_PRESET = "small"
MODEL_CONFIG = MODELS[MODEL_PRESET]

# ── HuggingFace pretrained weights ────────────────────────────────────────────
# Maps --hf variant names to their HuggingFace repo + architecture.
# Architecture configs are shared with MODELS above — no duplication.
# Usage: sair generate "..." --hf gpt2-124m
#        sair ui --hf gpt2-355m
HF_MODELS = {
    "gpt2":        {"repo": "openai-community/gpt2",       "config": MODELS["gpt2-124m"]},
    "gpt2-medium": {"repo": "openai-community/gpt2-medium", "config": MODELS["gpt2-355m"]},
    "gpt2-large":  {"repo": "openai-community/gpt2-large",  "config": MODELS["gpt2-774m"]},
    "gpt2-xl":     {"repo": "openai-community/gpt2-xl",     "config": MODELS["gpt2-1558m"]},
    # also accept exact param-count names
    "gpt2-124m":   {"repo": "openai-community/gpt2",       "config": MODELS["gpt2-124m"]},
    "gpt2-355m":   {"repo": "openai-community/gpt2-medium", "config": MODELS["gpt2-355m"]},
    "gpt2-774m":   {"repo": "openai-community/gpt2-large",  "config": MODELS["gpt2-774m"]},
    "gpt2-1558m":  {"repo": "openai-community/gpt2-xl",     "config": MODELS["gpt2-1558m"]},
}

# ── Data splits ────────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.90
VAL_SPLIT   = 0.07
# test gets the rest

# ── DataLoader ─────────────────────────────────────────────────────────────────
MAX_LEN    = MODEL_CONFIG["context_length"]
STRIDE     = MAX_LEN // 2
BATCH_SIZE = 8

# ── Training ───────────────────────────────────────────────────────────────────
LEARNING_RATE  = 4e-4
WEIGHT_DECAY   = 0.1
BETAS          = (0.9, 0.95)
NUM_EPOCHS     = 5
EVAL_FREQ      = 100
EVAL_ITER      = 10
GRAD_CLIP      = 1.0
TOTAL_BATCH    = 32768   # effective batch tokens (for gradient accumulation)
MICRO_BATCH    = BATCH_SIZE
START_CONTEXT  = "Harry Potter"

# ── Inference defaults ─────────────────────────────────────────────────────────
GEN_MAX_TOKENS  = 100
GEN_TEMPERATURE = 0.8
GEN_TOP_K       = 50
GEN_TOP_P       = 0.9
GEN_BEAMS       = 3

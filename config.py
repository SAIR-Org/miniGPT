"""
config.py — the only file you need to touch.

Change MODEL_PRESET, drop files in data/raw/, tweak training params.
Everything else reads from here.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
DATA_RAW  = ROOT / "data" / "raw"        # drop your .txt / .pdf files here
DATA_DIR  = ROOT / "data" / "processed"  # tokenized .bin files land here
CKPT_DIR  = ROOT / "checkpoints"         # model checkpoints

# ── Model presets ──────────────────────────────────────────────────────────────
MODELS = {
    "tiny": {                            # ~10M params — runs on CPU in minutes
        "vocab_size": 50257, "context_length": 256,
        "emb_dim": 256, "n_heads": 4, "n_layers": 4,
        "drop_rate": 0.1, "qkv_bias": False,
    },
    "small": {                           # ~50M params — good for a single GPU
        "vocab_size": 50257, "context_length": 512,
        "emb_dim": 512, "n_heads": 8, "n_layers": 6,
        "drop_rate": 0.1, "qkv_bias": False,
    },
    "gpt2-124m": {                       # exact GPT-2 small — needs GPU
        "vocab_size": 50257, "context_length": 1024,
        "emb_dim": 768, "n_heads": 12, "n_layers": 12,
        "drop_rate": 0.1, "qkv_bias": False,
    },
}

# ── Active model — swap preset to scale up/down ───────────────────────────────
MODEL_PRESET = "small"          # "tiny" | "small" | "gpt2-124m"
MODEL_CONFIG = MODELS[MODEL_PRESET]

# ── HuggingFace pretrained GPT-2 variants ─────────────────────────────────────
# Use with: sair generate "..." --hf gpt2
#           sair ui --hf gpt2-medium
HF_MODELS = {
    "gpt2": {
        "repo": "openai-community/gpt2",
        "config": {**MODELS["gpt2-124m"], "qkv_bias": True, "drop_rate": 0.0},
    },
    "gpt2-medium": {
        "repo": "openai-community/gpt2-medium",
        "config": {
            "vocab_size": 50257, "context_length": 1024,
            "emb_dim": 1024, "n_heads": 16, "n_layers": 24,
            "drop_rate": 0.0, "qkv_bias": True,
        },
    },
    "gpt2-large": {
        "repo": "openai-community/gpt2-large",
        "config": {
            "vocab_size": 50257, "context_length": 1024,
            "emb_dim": 1280, "n_heads": 20, "n_layers": 36,
            "drop_rate": 0.0, "qkv_bias": True,
        },
    },
    "gpt2-xl": {
        "repo": "openai-community/gpt2-xl",
        "config": {
            "vocab_size": 50257, "context_length": 1024,
            "emb_dim": 1600, "n_heads": 25, "n_layers": 48,
            "drop_rate": 0.0, "qkv_bias": True,
        },
    },
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
START_CONTEXT  = "Once upon a time"

# ── Inference defaults ─────────────────────────────────────────────────────────
GEN_MAX_TOKENS  = 100
GEN_TEMPERATURE = 0.8
GEN_TOP_K       = 50
GEN_TOP_P       = 0.9
GEN_BEAMS       = 3

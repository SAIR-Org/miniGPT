"""
Shared fixtures for all tests.
Uses a tiny model config so every test runs fast on CPU.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from model.gpt import GPTModel


# ── Tiny config — fast on CPU, same structure as real configs ─────────────────
TINY_CONFIG = {
    "vocab_size"    : 256,
    "context_length": 16,
    "emb_dim"       : 32,
    "n_heads"       : 2,
    "n_layers"      : 2,
    "drop_rate"     : 0.0,
    "qkv_bias"      : False,
}

DEVICE = "cpu"


@pytest.fixture
def tiny_config():
    return TINY_CONFIG


@pytest.fixture
def tiny_model():
    model = GPTModel(TINY_CONFIG)
    model.eval()
    return model


@pytest.fixture
def tokenizer():
    import tiktoken
    return tiktoken.encoding_for_model("gpt2")


class TinyTokenizer:
    """Mock tokenizer that maps bytes → IDs within tiny vocab_size=256."""
    def encode(self, text):
        return [b % TINY_CONFIG["vocab_size"] for b in text.encode("utf-8")]
    def decode(self, ids):
        return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


@pytest.fixture
def tiny_tokenizer():
    return TinyTokenizer()


@pytest.fixture
def bin_file(tmp_path):
    """A temporary .bin file with random int32 token IDs."""
    ids = np.random.randint(0, TINY_CONFIG["vocab_size"],
                            size=1000, dtype=np.int32)
    path = tmp_path / "train_ids.bin"
    ids.tofile(path)
    return path


@pytest.fixture
def fake_checkpoint(tmp_path, tiny_model):
    """A saved checkpoint using the tiny model."""
    ckpt = tmp_path / "epoch_01.pt"
    torch.save({"epoch": 1, "model": tiny_model.state_dict(),
                "optimizer": {}}, ckpt)
    return tmp_path

"""test_model.py — GPTModel forward pass, shapes, parameter count."""
import torch
from model.gpt import GPTModel, TransformerBlock, MultiHeadAttention, FeedForward


def test_forward_output_shape(tiny_config):
    model = GPTModel(tiny_config)
    B, T  = 2, tiny_config["context_length"]
    x     = torch.randint(0, tiny_config["vocab_size"], (B, T))
    out   = model(x)
    assert out.shape == (B, T, tiny_config["vocab_size"])


def test_forward_shorter_sequence(tiny_config):
    model = GPTModel(tiny_config)
    x     = torch.randint(0, tiny_config["vocab_size"], (1, 4))
    out   = model(x)
    assert out.shape == (1, 4, tiny_config["vocab_size"])


def test_parameter_count(tiny_config):
    model  = GPTModel(tiny_config)
    params = sum(p.numel() for p in model.parameters())
    assert params > 0


def test_no_nan_in_output(tiny_config):
    model = GPTModel(tiny_config)
    x     = torch.randint(0, tiny_config["vocab_size"], (2, 8))
    out   = model(x)
    assert not torch.isnan(out).any()


def test_transformer_block_shape(tiny_config):
    block = TransformerBlock(tiny_config)
    x     = torch.randn(2, tiny_config["context_length"], tiny_config["emb_dim"])
    out   = block(x)
    assert out.shape == x.shape


def test_all_presets_build():
    from config import MODELS
    # Only build scratch-tier presets — GPT-2 variants (355m/774m/1558m)
    # allocate 1–6 GB of RAM just to instantiate architecture on CPU.
    SCRATCH_PRESETS = {"tiny", "small", "medium", "custom"}
    for name, cfg in MODELS.items():
        if name not in SCRATCH_PRESETS:
            continue
        model = GPTModel(cfg)
        assert model is not None, f"Failed to build preset '{name}'"

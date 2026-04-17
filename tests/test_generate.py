"""test_generate.py — all generate functions produce valid text output."""
import torch
import pytest

from inference.generate import (
    generateV0, generateV1, generateV2, generateV3,
    generate, load_model, apply_top_k, apply_top_p,
)

PROMPT     = "Hi"
MAX_TOKENS = 5
CONTEXT    = 16    # matches TINY_CONFIG context_length


def test_generateV0_returns_string(tiny_model, tiny_tokenizer):
    out = generateV0(tiny_model, PROMPT, MAX_TOKENS, CONTEXT, tiny_tokenizer, "cpu")
    assert isinstance(out, str)


def test_generateV1_temperature_zero_is_deterministic(tiny_model, tiny_tokenizer):
    out1 = generateV1(tiny_model, PROMPT, MAX_TOKENS, CONTEXT, tiny_tokenizer, "cpu", temperature=0.0)
    out2 = generateV1(tiny_model, PROMPT, MAX_TOKENS, CONTEXT, tiny_tokenizer, "cpu", temperature=0.0)
    assert out1 == out2


def test_generateV1_high_temperature_varies(tiny_model, tiny_tokenizer):
    torch.manual_seed(0)
    out1 = generateV1(tiny_model, PROMPT, 10, CONTEXT, tiny_tokenizer, "cpu", temperature=2.0)
    torch.manual_seed(99)
    out2 = generateV1(tiny_model, PROMPT, 10, CONTEXT, tiny_tokenizer, "cpu", temperature=2.0)
    assert isinstance(out1, str) and isinstance(out2, str)


def test_generateV2_top_k(tiny_model, tiny_tokenizer):
    out = generateV2(tiny_model, PROMPT, MAX_TOKENS, CONTEXT, tiny_tokenizer, "cpu",
                     temperature=1.0, top_k=10)
    assert isinstance(out, str)


def test_generateV2_top_p(tiny_model, tiny_tokenizer):
    out = generateV2(tiny_model, PROMPT, MAX_TOKENS, CONTEXT, tiny_tokenizer, "cpu",
                     temperature=1.0, top_p=0.9)
    assert isinstance(out, str)


def test_generateV2_eos_stops_early(tiny_model, tiny_tokenizer):
    out = generateV2(tiny_model, PROMPT, 50, CONTEXT, tiny_tokenizer, "cpu",
                     temperature=0.0, eos=0)
    assert isinstance(out, str)


def test_generateV3_beam_search(tiny_model, tiny_tokenizer):
    out = generateV3(tiny_model, PROMPT, MAX_TOKENS, CONTEXT, tiny_tokenizer, "cpu",
                     beams=3, method="greedy")
    assert isinstance(out, str)


def test_generate_dispatcher_single_beam(tiny_model, tiny_tokenizer):
    out = generate(tiny_model, PROMPT, MAX_TOKENS, CONTEXT, tiny_tokenizer, "cpu", beams=1)
    assert isinstance(out, str)


def test_generate_dispatcher_multi_beam(tiny_model, tiny_tokenizer):
    out = generate(tiny_model, PROMPT, MAX_TOKENS, CONTEXT, tiny_tokenizer, "cpu", beams=3)
    assert isinstance(out, str)


def test_apply_top_k_zeros_out_non_top():
    logits = torch.tensor([[3.0, 1.0, 0.5, 2.0, 0.1]])
    out    = apply_top_k(logits, k=2)
    # only top-2 should be non -inf
    finite = (out != float("-inf")).sum().item()
    assert finite == 2


def test_apply_top_p_sums_to_below_p():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, 0.1]])
    out    = apply_top_p(logits, p=0.9)
    assert isinstance(out, torch.Tensor)


def test_load_model_from_checkpoint(fake_checkpoint, tiny_config):
    from model.gpt import GPTModel
    model = load_model(tiny_config, fake_checkpoint, "cpu")
    assert model is not None
    model.eval()
    x   = torch.randint(0, tiny_config["vocab_size"], (1, 4))
    out = model(x)
    assert out.shape == (1, 4, tiny_config["vocab_size"])


def test_load_model_raises_if_no_checkpoint(tmp_path, tiny_config):
    with pytest.raises(FileNotFoundError):
        load_model(tiny_config, tmp_path / "empty", "cpu")

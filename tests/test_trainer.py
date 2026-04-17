"""test_trainer.py — training loop runs without error on synthetic data."""
import torch
import numpy as np
import pytest
from torch.utils.data import DataLoader

from model.gpt   import GPTModel
from data.dataset import GPT2Dataset
from train.trainer import calc_loss_batch, calc_loss_loader, evaluate, train


def _make_loader(tmp_path, config, n=500, batch_size=2):
    ids  = np.random.randint(0, config["vocab_size"], size=n, dtype=np.int32)
    path = tmp_path / "train_ids.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(path)
    ds   = GPT2Dataset(path, config["context_length"], stride=config["context_length"] // 2)
    return DataLoader(ds, batch_size=batch_size, drop_last=True)


def test_calc_loss_batch(tiny_config):
    model = GPTModel(tiny_config)
    B, T  = 2, tiny_config["context_length"]
    x     = torch.randint(0, tiny_config["vocab_size"], (B, T))
    y     = torch.randint(0, tiny_config["vocab_size"], (B, T))
    loss  = calc_loss_batch(x, y, model, "cpu")
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_calc_loss_loader(tmp_path, tiny_config):
    model  = GPTModel(tiny_config)
    loader = _make_loader(tmp_path, tiny_config)
    loss   = calc_loss_loader(loader, model, "cpu", num_batches=2)
    assert loss > 0


def test_evaluate_returns_two_values(tmp_path, tiny_config):
    model  = GPTModel(tiny_config)
    loader = _make_loader(tmp_path, tiny_config)
    tl, vl = evaluate(model, loader, loader, "cpu", eval_iter=2)
    assert tl > 0
    assert vl > 0


def test_train_one_epoch_reduces_loss(tmp_path, tiny_config, tiny_tokenizer):
    torch.manual_seed(42)

    model        = GPTModel(tiny_config)
    train_loader = _make_loader(tmp_path, tiny_config, n=2000)
    val_loader   = _make_loader(tmp_path / "val", tiny_config, n=500)

    loss_before  = calc_loss_loader(train_loader, model, "cpu", num_batches=3)

    train(
        model,
        train_loader,
        val_loader,
        device        = "cpu",
        num_epochs    = 1,
        eval_freq     = 999,
        eval_iter     = 1,
        save_dir      = tmp_path / "ckpts",
        start_context = "Hi",
        tokenizer     = tiny_tokenizer,
        context_size  = tiny_config["context_length"],
    )

    loss_after = calc_loss_loader(train_loader, model, "cpu", num_batches=3)
    assert loss_after < loss_before, "Loss should decrease after one epoch"


def test_checkpoint_is_saved(tmp_path, tiny_config, tiny_tokenizer):
    model        = GPTModel(tiny_config)
    train_loader = _make_loader(tmp_path, tiny_config, n=1000)
    val_loader   = _make_loader(tmp_path / "val", tiny_config, n=200)
    ckpt_dir     = tmp_path / "ckpts"

    train(model, train_loader, val_loader, "cpu",
          num_epochs=1, eval_freq=999, eval_iter=1,
          save_dir=ckpt_dir, start_context="Hi", tokenizer=tiny_tokenizer,
          context_size=tiny_config["context_length"])

    checkpoints = list(ckpt_dir.glob("epoch_*.pt"))
    assert len(checkpoints) == 1

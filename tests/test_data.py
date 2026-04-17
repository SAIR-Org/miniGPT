"""test_data.py — data preparation, dataset, and dataloader."""
import numpy as np
import pytest
import torch
from pathlib import Path

from data.dataset import GPT2Dataset


def test_dataset_length(bin_file):
    max_len, stride = 8, 4
    ds = GPT2Dataset(bin_file, max_len, stride)
    expected = (1000 - max_len) // stride
    assert len(ds) == expected


def test_dataset_item_shapes(bin_file):
    max_len = 8
    ds      = GPT2Dataset(bin_file, max_len, stride=4)
    x, y    = ds[0]
    assert x.shape == (max_len,)
    assert y.shape == (max_len,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_dataset_target_is_shifted(bin_file):
    ds   = GPT2Dataset(bin_file, max_length=8, stride=1)
    x, y = ds[0]
    # y should be x shifted by one position
    assert torch.equal(x[1:], y[:-1])


def test_dataloader_batching(bin_file):
    from torch.utils.data import DataLoader
    ds     = GPT2Dataset(bin_file, max_length=8, stride=4)
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    xb, yb = next(iter(loader))
    assert xb.shape == (4, 8)
    assert yb.shape == (4, 8)


def test_prepare_from_txt(tmp_path):
    from data.prepare import prepare

    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()

    # write a small .txt corpus
    (raw_dir / "book.txt").write_text(
        "Harry Potter and the Philosopher's Stone. " * 500,
        encoding="utf-8"
    )

    prepare(raw_dir=raw_dir, out_dir=out_dir)

    assert (out_dir / "train_ids.bin").exists()
    assert (out_dir / "val_ids.bin").exists()
    assert (out_dir / "test_ids.bin").exists()

    # check files are non-empty
    for split in ("train", "val", "test"):
        data = np.fromfile(out_dir / f"{split}_ids.bin", dtype=np.int32)
        assert len(data) > 0


def test_prepare_raises_on_empty_dir(tmp_path):
    from data.prepare import prepare
    with pytest.raises(FileNotFoundError):
        prepare(raw_dir=tmp_path / "empty", out_dir=tmp_path / "out")

"""
data/dataset.py — GPT2Dataset and DataLoader factory.
Identical to Notebook 1 — just made importable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_DIR, MAX_LEN, STRIDE, BATCH_SIZE


class GPT2Dataset(Dataset):
    def __init__(self, file_path: Path, max_length: int, stride: int):
        self.data       = np.fromfile(file_path, dtype=np.int32)
        self.max_length = max_length
        self.stride     = stride

    def __len__(self):
        return (len(self.data) - self.max_length) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        x = torch.tensor(self.data[start : start + self.max_length],         dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.max_length + 1], dtype=torch.long)
        return x, y


def get_loaders(
    data_dir:   Path = DATA_DIR,
    max_length: int  = MAX_LEN,
    stride:     int  = STRIDE,
    batch_size: int  = BATCH_SIZE,
):
    def _make(split, shuffle):
        path = Path(data_dir) / f"{split}_ids.bin"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run: python cli.py prepare"
            )
        ds = GPT2Dataset(path, max_length, stride)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return _make("train", True), _make("val", False), _make("test", False)

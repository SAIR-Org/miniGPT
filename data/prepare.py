"""
data/prepare.py — load .txt and .pdf files, tokenize, save .bin splits.

Usage:
    python cli.py prepare
    python data/prepare.py        # direct
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tiktoken

from config import DATA_RAW, DATA_DIR, TRAIN_SPLIT, VAL_SPLIT

try:
    import fitz   # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


def extract_pdf(path: Path) -> str:
    if not HAS_PYMUPDF:
        raise ImportError("pymupdf not installed — run: uv add pymupdf")
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def load_corpus(raw_dir: Path = DATA_RAW) -> str:
    paths = sorted(raw_dir.iterdir())
    supported = [p for p in paths if p.suffix in (".txt", ".pdf")]

    if not supported:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in {raw_dir}\n"
            "Drop your files there and re-run: python cli.py prepare"
        )

    texts = []
    for path in supported:
        if path.suffix == ".txt":
            print(f"  [txt] {path.name}")
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        elif path.suffix == ".pdf":
            print(f"  [pdf] {path.name}")
            texts.append(extract_pdf(path))

    return "\n\n".join(texts)


def prepare(raw_dir: Path = DATA_RAW, out_dir: Path = DATA_DIR) -> None:
    print(f"\nLoading corpus from {raw_dir} ...")
    corpus = load_corpus(raw_dir)
    print(f"Total characters : {len(corpus):,}")

    n           = len(corpus)
    train_end   = int(n * TRAIN_SPLIT)
    val_end     = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    splits      = {
        "train": corpus[:train_end],
        "val"  : corpus[train_end:val_end],
        "test" : corpus[val_end:],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = tiktoken.encoding_for_model("gpt2")

    print("\nTokenizing and saving ...")
    for name, text in splits.items():
        ids  = np.array(tokenizer.encode(text), dtype=np.int32)
        path = out_dir / f"{name}_ids.bin"
        ids.tofile(path)
        print(f"  {name:5s}: {len(ids):>10,} tokens  →  {path}")

    print("\nDone. Ready to train.")


if __name__ == "__main__":
    prepare()

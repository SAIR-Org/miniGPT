"""
train/modal_train.py — run trainerV3 on Modal cloud GPU.

Usage:
    modal run train/modal_train.py

Requires:
    modal token new    # one-time auth
    uv add modal
"""
import modal
from pathlib import Path

# ── Modal app + persistent volume for checkpoints ─────────────────────────────
app    = modal.App("sair-minigpt")
volume = modal.Volume.from_name("sair-minigpt-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "tiktoken", "numpy", "transformers", "pymupdf")
)

REMOTE_DATA = "/data"
REMOTE_CKPT = "/checkpoints"


# ── Upload your processed data to Modal ──────────────────────────────────────

@app.local_entrypoint()
def main():
    """Uploads data, kicks off training, downloads latest checkpoint."""
    import subprocess, shutil, sys
    from config import DATA_DIR, CKPT_DIR

    print("Uploading data to Modal volume...")
    train_fn.remote()


# ── Training function — runs on A100 in the cloud ────────────────────────────

@app.function(
    image   = image,
    gpu     = "A100",                        # swap to "T4" for cheaper runs
    volumes = {
        REMOTE_DATA : volume,
        REMOTE_CKPT : volume,
    },
    timeout = 3 * 3600,                      # 3 hours max
    mounts  = [
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent,    # entire sair-minigpt/ folder
            remote_path="/app",
        )
    ],
)
def train_fn():
    import sys
    sys.path.insert(0, "/app")

    import torch
    from config import MODEL_CONFIG
    from data.dataset   import get_loaders
    from model.gpt      import GPTModel
    from train.trainer  import train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    model = GPTModel(MODEL_CONFIG).to(device)
    train_loader, val_loader, _ = get_loaders(data_dir=REMOTE_DATA)

    train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        save_dir     = REMOTE_CKPT,
    )

    volume.commit()
    print("Training complete. Checkpoints saved to Modal volume.")

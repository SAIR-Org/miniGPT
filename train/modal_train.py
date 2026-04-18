"""
train/modal_train.py — run trainer on Modal cloud GPU.

Usage:
    uv run python -m modal run train/modal_train.py

Requires:
    uv run python -m modal token new    # one-time auth
"""
import modal
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "processed"

REMOTE_DATA = "/data"
REMOTE_CKPT = "/checkpoints"

# ── Modal app + persistent volume for checkpoints ─────────────────────────────
app    = modal.App("sair-minigpt")
volume = modal.Volume.from_name("sair-minigpt-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "tiktoken", "numpy", "transformers", "pymupdf", "wandb", "matplotlib")
    .add_local_dir(str(ROOT), remote_path="/app", ignore=[
        ".venv", "__pycache__", "*.pyc", "*.pyo",
        "checkpoints", "data/raw", "runs", "mlruns",
        ".git", "*.gif", "*.jpg", "*.png",
    ])
    .add_local_dir(str(DATA_DIR), remote_path=REMOTE_DATA)   # tokenized .bin files
)


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    print("Launching training on Modal A100...")
    train_fn.remote()


# ── Training function — runs on A100 in the cloud ────────────────────────────

@app.function(
    image   = image,
    gpu     = "A100",
    volumes = {REMOTE_CKPT: volume},
    secrets = [modal.Secret.from_name("wandb-secret")],
    timeout = 3 * 3600,
)
def train_fn():
    import sys
    sys.path.insert(0, "/app")

    import torch
    from config import MODEL_CONFIG
    from data.dataset  import get_loaders
    from model.gpt     import GPTModel
    from train.trainer import train

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

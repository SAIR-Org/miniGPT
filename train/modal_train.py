"""
train/modal_train.py — run trainer on Modal cloud GPU.

Usage:
    uv run python -m modal run train/modal_train.py          # train
    uv run python -m modal run train/modal_train.py::download # download final checkpoint

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
    timeout = 8 * 3600,
)
def train_fn():
    import sys
    sys.path.insert(0, "/app")

    import torch
    from config import MODELS
    from data.dataset  import get_loaders
    from model.gpt     import GPTModel
    from train.trainer import train

    config = MODELS["medium"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    model = GPTModel(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ctx = config["context_length"]  # 256 for tiny, 1024 for medium
    train_loader, val_loader, _ = get_loaders(
        data_dir=REMOTE_DATA,
        max_length=ctx,
        stride=ctx // 2,
    )

    train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        save_dir     = REMOTE_CKPT,
        num_epochs   = 30,
        context_size = ctx,
        model_config = config,
    )

    volume.commit()
    print("Training complete. Checkpoints saved to Modal volume.")


# ── Download final checkpoint to local machine ────────────────────────────────

@app.local_entrypoint()
def download():
    local_ckpt_dir = ROOT / "checkpoints"
    local_ckpt_dir.mkdir(exist_ok=True)

    files = list(volume.listdir("/"))
    ckpt_files = sorted(
        [f.path for f in files if f.path.startswith("epoch_") and f.path.endswith(".pt")]
    )
    extra_files = [f.path for f in files if not f.path.startswith("epoch_")]

    if not ckpt_files:
        print("No checkpoints found in Modal volume.")
        return

    # download the latest checkpoint + loss curve
    to_download = [ckpt_files[-1]] + [f for f in extra_files if f.endswith(".png")]

    for filename in to_download:
        dest = local_ckpt_dir / filename
        print(f"Downloading {filename} → {dest}")
        data = b"".join(volume.read_file(filename))
        dest.write_bytes(data)

    print(f"\nDone. Latest checkpoint: checkpoints/{ckpt_files[-1]}")
    print("Run inference with:  uv run python cli.py generate 'Harry Potter'")
    print("Or launch the UI:    uv run python cli.py ui")

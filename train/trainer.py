"""
train/trainer.py — trainerV3 from Notebook 4.
Gradient clipping + cosine LR + gradient accumulation.
W&B logging + matplotlib loss plot after training.

Usage:
    sair train
    python train/trainer.py    # direct
"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import matplotlib
matplotlib.use("Agg")   # headless — safe for Modal / servers
import matplotlib.pyplot as plt

from config import (
    MODEL_CONFIG, CKPT_DIR,
    LEARNING_RATE, WEIGHT_DECAY, BETAS,
    NUM_EPOCHS, EVAL_FREQ, EVAL_ITER,
    GRAD_CLIP, TOTAL_BATCH, MICRO_BATCH,
    MAX_LEN, START_CONTEXT, MODEL_PRESET,
)
from data.dataset        import get_loaders
from model.gpt           import GPTModel
from inference.generate  import generate


# ── Helpers ───────────────────────────────────────────────────────────────────

def calc_loss_batch(x, y, model, device):
    x, y   = x.to(device), y.to(device)
    logits = model(x)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())


def calc_loss_loader(loader, model, device, num_batches=None):
    if len(loader) == 0:
        return float("nan")
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        total += calc_loss_batch(x, y, model, device).item()
        count += 1
    return total / count


def evaluate(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        tl = calc_loss_loader(train_loader, model, device, eval_iter)
        vl = calc_loss_loader(val_loader,   model, device, eval_iter)
    model.train()
    return tl, vl


def _save_loss_plot(train_losses, val_losses, tokens_seen, save_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tokens_seen, train_losses, label="Train loss", linewidth=2)
    ax.plot(tokens_seen, val_losses,   label="Val loss",   linewidth=2, linestyle="--")
    ax.set_xlabel("Tokens seen")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"miniGPT — {MODEL_PRESET} — Harry Potter")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = save_dir / "loss_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Loss plot saved → {path}")
    return path


# ── trainerV3 ─────────────────────────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs    = NUM_EPOCHS,
    eval_freq     = EVAL_FREQ,
    eval_iter     = EVAL_ITER,
    save_dir      = CKPT_DIR,
    start_context = START_CONTEXT,
    tokenizer     = None,
    context_size  = None,
    use_wandb     = True,
    resume_from   = None,
    model_config  = None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if tokenizer is None:
        tokenizer = tiktoken.encoding_for_model("gpt2")
    if model_config is None:
        model_config = MODEL_CONFIG
    if context_size is None:
        context_size = model_config["context_length"]

    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, betas=BETAS, eps=1e-8,
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion  = nn.CrossEntropyLoss()
    grad_accum = max(1, TOTAL_BATCH // (MICRO_BATCH * context_size))

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        # advance scheduler to match saved epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from {resume_from} — continuing from epoch {start_epoch + 1}\n")

    # ── W&B init ──────────────────────────────────────────────────────────────
    wb = None
    if use_wandb:
        try:
            import wandb
            wb = wandb.init(
                project = "sair-minigpt",
                name    = f"{model_config.get('emb_dim', '?')}d-{model_config.get('n_layers', '?')}L-harry-potter",
                config  = {
                    **model_config,
                    "num_epochs":    num_epochs,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay":  WEIGHT_DECAY,
                    "batch_size":    MICRO_BATCH,
                    "grad_accum":    grad_accum,
                    "dataset":       "harry-potter-6books",
                },
            )
            print(f"W&B run: {wb.url}\n")
        except Exception as e:
            print(f"W&B disabled ({e})")
            wb = None

    print(f"Gradient accumulation steps : {grad_accum}")
    print(f"Effective batch              : {grad_accum * MICRO_BATCH * MAX_LEN:,} tokens\n")

    train_losses, val_losses, tokens_seen = [], [], []
    total_tokens = 0
    global_step  = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y    = x.to(device), y.to(device)
            logits  = model(x)
            loss    = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss    = loss / grad_accum
            loss.backward()

            total_tokens += x.numel()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % eval_freq == 0:
                    tl, vl = evaluate(model, train_loader, val_loader, device, eval_iter)
                    lr     = scheduler.get_last_lr()[0]
                    train_losses.append(tl)
                    val_losses.append(vl)
                    tokens_seen.append(total_tokens)

                    print(f"Ep {epoch+1} | step {global_step:05d} | "
                          f"train {tl:.3f} | val {vl:.3f} | lr {lr:.2e}")

                    if wb:
                        wb.log({
                            "train/loss": tl,
                            "val/loss":   vl,
                            "train/lr":   lr,
                            "tokens_seen": total_tokens,
                        }, step=global_step)

        # handle leftover batches
        if len(train_loader) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # ── end-of-epoch sample ───────────────────────────────────────────────
        sample = generate(
            model, start_context, max_new_tokens=60,
            context_size=context_size,
            tokenizer=tokenizer, device=device,
            temperature=0.8, top_k=50,
        )
        print(f"\nSample (epoch {epoch+1}):\n  {sample}\n")

        if wb:
            import wandb
            wb.log({
                "epoch": epoch + 1,
                "sample": wandb.Html(f"<pre>{sample}</pre>"),
            }, step=global_step)

        ckpt = save_dir / f"epoch_{epoch+1:02d}.pt"
        torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, ckpt)
        print(f"Checkpoint → {ckpt}  ({time.time()-t0:.1f}s)\n")

    # ── post-training plot ────────────────────────────────────────────────────
    if train_losses:
        plot_path = _save_loss_plot(train_losses, val_losses, tokens_seen, save_dir)
        if wb:
            import wandb
            wb.log({"loss_curve": wandb.Image(str(plot_path))})

    if wb:
        wb.finish()

    return train_losses, val_losses, tokens_seen


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = GPTModel(MODEL_CONFIG).to(device)
    train_loader, val_loader, _ = get_loaders()
    train(model, train_loader, val_loader, device)

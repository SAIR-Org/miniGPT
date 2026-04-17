"""
train/trainer.py — trainerV3 from Notebook 4.
Gradient clipping + cosine LR + gradient accumulation.

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

from config import (
    MODEL_CONFIG, CKPT_DIR,
    LEARNING_RATE, WEIGHT_DECAY, BETAS,
    NUM_EPOCHS, EVAL_FREQ, EVAL_ITER,
    GRAD_CLIP, TOTAL_BATCH, MICRO_BATCH,
    MAX_LEN, START_CONTEXT,
)
from data.dataset        import get_loaders
from model.gpt           import GPTModel
from inference.generate  import generate


# ── Helpers — same as Notebook 4 ─────────────────────────────────────────────

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
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if tokenizer is None:
        tokenizer = tiktoken.encoding_for_model("gpt2")
    if context_size is None:
        context_size = MODEL_CONFIG["context_length"]
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, betas=BETAS, eps=1e-8,
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion  = nn.CrossEntropyLoss()
    grad_accum = max(1, TOTAL_BATCH // (MICRO_BATCH * MAX_LEN))

    print(f"Gradient accumulation steps : {grad_accum}")
    print(f"Effective batch              : {grad_accum * MICRO_BATCH * MAX_LEN:,} tokens\n")

    train_losses, val_losses, tokens_seen = [], [], []
    total_tokens = 0
    global_step  = 0

    for epoch in range(num_epochs):
        model.train()
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):   # one full pass per epoch
            x, y    = x.to(device), y.to(device)
            logits  = model(x)
            loss    = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss    = loss / grad_accum                      # normalise for accumulation
            loss.backward()

            total_tokens += x.numel()

            # step every grad_accum batches
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % eval_freq == 0:
                    tl, vl = evaluate(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(tl)
                    val_losses.append(vl)
                    tokens_seen.append(total_tokens)
                    lr = scheduler.get_last_lr()[0]
                    print(f"Ep {epoch+1} | step {global_step:05d} | "
                          f"train {tl:.3f} | val {vl:.3f} | lr {lr:.2e}")

        # handle leftover batches that didn't fill a full accum window
        if (len(train_loader)) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # live sample after each epoch
        sample = generate(
            model, start_context, max_new_tokens=40,
            context_size=context_size,
            tokenizer=tokenizer, device=device,
            temperature=0.8, top_k=50,
        )
        print(f"\nSample: {sample}\n")

        ckpt = save_dir / f"epoch_{epoch+1:02d}.pt"
        torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, ckpt)
        print(f"Checkpoint → {ckpt}  ({time.time()-t0:.1f}s)\n")

    return train_losses, val_losses, tokens_seen


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = GPTModel(MODEL_CONFIG).to(device)
    train_loader, val_loader, _ = get_loaders()
    train(model, train_loader, val_loader, device)

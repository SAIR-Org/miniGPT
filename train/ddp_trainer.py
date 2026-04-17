"""
train/ddp_trainer.py — trainerV4 from Notebook 4: multi-GPU with DDP.

Launch with torchrun (NOT python directly):
    torchrun --standalone --nproc_per_node=NUM_GPUS train/ddp_trainer.py

Or via the CLI:
    sair train --ddp                         # uses all available GPUs
    sair train --ddp --nproc 2              # use 2 GPUs

How DDP works (quick recap from Notebook 4):
    1. torchrun spawns one process per GPU, each gets a unique rank (0, 1, 2 ...)
    2. Each process holds a full copy of the model
    3. DistributedSampler splits data so each GPU sees different batches
    4. After loss.backward(), gradients are AllReduced across all GPUs automatically
    5. All copies stay in sync — only rank 0 saves checkpoints and prints logs
"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import tiktoken

from config import (
    MODEL_CONFIG, CKPT_DIR,
    LEARNING_RATE, WEIGHT_DECAY, BETAS,
    NUM_EPOCHS, EVAL_FREQ, EVAL_ITER,
    GRAD_CLIP, TOTAL_BATCH, MICRO_BATCH,
    MAX_LEN, START_CONTEXT, DATA_DIR, BATCH_SIZE, STRIDE,
)
from data.dataset        import GPT2Dataset
from model.gpt           import GPTModel
from inference.generate  import generate
from train.trainer       import calc_loss_batch, calc_loss_loader, evaluate


# ── DDP setup / teardown ──────────────────────────────────────────────────────

def ddp_setup():
    dist.init_process_group(backend="nccl")   # torchrun sets env vars automatically
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def ddp_cleanup():
    dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


# ── DDP-aware DataLoaders ─────────────────────────────────────────────────────

def get_ddp_loaders(data_dir=DATA_DIR, max_length=MAX_LEN,
                    stride=STRIDE, batch_size=BATCH_SIZE):
    def _make(split):
        path = Path(data_dir) / f"{split}_ids.bin"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run: sair prepare")
        ds      = GPT2Dataset(path, max_length, stride)
        sampler = DistributedSampler(ds, shuffle=(split == "train"))
        return DataLoader(ds, batch_size=batch_size,
                          sampler=sampler, drop_last=True), sampler

    train_loader, train_sampler = _make("train")
    val_loader,   _             = _make("val")
    return train_loader, val_loader, train_sampler


# ── trainerV4 ─────────────────────────────────────────────────────────────────

def train_ddp(
    model,
    train_loader,
    val_loader,
    train_sampler,
    device,
    num_epochs    = NUM_EPOCHS,
    eval_freq     = EVAL_FREQ,
    eval_iter     = EVAL_ITER,
    save_dir      = CKPT_DIR,
    start_context = START_CONTEXT,
):
    save_dir = Path(save_dir)
    if is_main():
        save_dir.mkdir(parents=True, exist_ok=True)

    # wrap model — DDP syncs gradients across all GPUs after every backward()
    model = DDP(model, device_ids=[device])

    tokenizer  = tiktoken.encoding_for_model("gpt2")
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, betas=BETAS, eps=1e-8,
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion  = nn.CrossEntropyLoss()
    grad_accum = max(1, TOTAL_BATCH // (MICRO_BATCH * MAX_LEN))

    if is_main():
        world = dist.get_world_size() if dist.is_initialized() else 1
        print(f"World size           : {world} GPUs")
        print(f"Grad accum steps     : {grad_accum}")
        print(f"Effective batch      : {grad_accum * MICRO_BATCH * MAX_LEN * world:,} tokens\n")

    train_losses, val_losses, tokens_seen = [], [], []
    total_tokens = 0
    global_step  = 0

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)   # different shuffle per epoch across ranks
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss   = loss / grad_accum
            loss.backward()

            total_tokens += x.numel()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # only rank 0 logs — avoids N duplicate lines
                if is_main() and global_step % eval_freq == 0:
                    tl, vl = evaluate(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(tl)
                    val_losses.append(vl)
                    tokens_seen.append(total_tokens)
                    lr = scheduler.get_last_lr()[0]
                    print(f"[rank 0] Ep {epoch+1} | step {global_step:05d} | "
                          f"train {tl:.3f} | val {vl:.3f} | lr {lr:.2e}")

        if len(train_loader) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # only rank 0 generates samples and saves checkpoints
        if is_main():
            raw_model = model.module   # unwrap DDP to get the actual GPTModel
            sample = generate(
                raw_model, start_context, max_new_tokens=40,
                context_size=MODEL_CONFIG["context_length"],
                tokenizer=tokenizer, device=device,
                temperature=0.8, top_k=50,
            )
            print(f"\nSample: {sample}\n")

            ckpt = save_dir / f"epoch_{epoch+1:02d}.pt"
            torch.save({"epoch": epoch + 1, "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict()}, ckpt)
            print(f"Checkpoint → {ckpt}  ({time.time()-t0:.1f}s)\n")

    return train_losses, val_losses, tokens_seen


# ── Entry point (called by torchrun) ──────────────────────────────────────────

def main():
    ddp_setup()
    device = int(os.environ["LOCAL_RANK"])

    model = GPTModel(MODEL_CONFIG).to(device)

    if is_main():
        params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {params:,}")

    train_loader, val_loader, train_sampler = get_ddp_loaders()
    train_ddp(model, train_loader, val_loader, train_sampler, device)
    ddp_cleanup()


if __name__ == "__main__":
    main()

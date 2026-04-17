# SAIR miniGPT

> **Capstone project for [SAIR Jr. — Module 5: GPT from Scratch](https://github.com/SAIR-Org/SAIR_Jr/tree/main/5_GPT%20from%20scratch).**  
> This is the full production implementation of everything built across the five core notebooks in that module. If you haven't worked through the notebooks first, start there.

A full-stack, hackable GPT system — from raw text to a running UI.  
Every function here (`GPTModel`, `generateV0`→`V3`, `trainerV3`, beam search) maps directly to a cell in the Module 5 notebooks, so the code is readable, modifiable, and yours to break.

```
your data (.txt / .pdf)
    │
    ▼  python cli.py prepare
tokenized .bin files
    │
    ▼  python cli.py train  (local GPU)
    │  python cli.py train --modal  (Modal A100)
checkpoints/
    │
    ▼  python cli.py ui
web UI (http://localhost:7860)  ←→  generate text
```

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/SAIR-Org/miniGPT
cd sair-minigpt
uv sync                  # installs deps + registers the `sair` command

# 2. Drop your data
cp your_book.txt  data/raw/
cp your_paper.pdf data/raw/   # .txt and .pdf both work, mix freely

# 3. Prepare
sair prepare

# 4. Train
sair train                    # local — CPU or single GPU (start here)
sair train --modal            # Modal A100 — the real power-up (recommended)
sair train --ddp              # local multi-GPU if you have it
sair train --ddp --nproc 2    # specify number of GPUs

# 5. Generate
sair generate "Once upon a time"
sair generate "Once upon a time" --method nucleus --temperature 0.8 --beams 3

# 6. UI
sair ui
```

---

## Customise

Everything lives in **`config.py`** — it's the only file most users touch.

```python
# Swap model size
MODEL_PRESET = "tiny"     # "tiny" | "small" | "gpt2-124m"

# Tune training
NUM_EPOCHS    = 10
LEARNING_RATE = 3e-4
BATCH_SIZE    = 16

# Tune generation defaults
GEN_TEMPERATURE = 0.9
GEN_BEAMS       = 5
```

To train on your own data: drop any `.txt` or `.pdf` files in `data/raw/` and re-run `prepare`.

---

## Structure

```
sair-minigpt/
├── config.py              ← start here
├── cli.py                 ← single entry point
├── data/
│   ├── prepare.py         ← loads .txt + .pdf, tokenizes, saves .bin
│   └── dataset.py         ← GPT2Dataset + DataLoader
├── model/
│   └── gpt.py             ← GPTModel (LayerNorm, MHA, FFN, TransformerBlock)
├── train/
│   ├── trainer.py         ← trainerV3 (clip + cosine LR + grad accumulation)
│   └── modal_train.py     ← same trainer wrapped for Modal A100
├── inference/
│   └── generate.py        ← generateV0 (greedy) → V3 (beam search)
└── ui/
    ├── server.py          ← FastAPI backend
    └── index.html         ← SAIR-branded dark UI
```

---

## Training options

### Local — single GPU (start here)
```bash
sair train
```
Works on CPU too. Use the `"tiny"` preset in `config.py` for fast iteration.

### Modal A100 — recommended for real runs
[Modal](https://modal.com) gives you an A100 in the cloud with one command.

```bash
modal token new          # one-time browser login
sair train --modal       # launches on A100, streams logs back
```

Checkpoints are saved to a Modal persistent volume between runs.  
Change `gpu="A100"` to `gpu="T4"` in `train/modal_train.py` for a cheaper option.

### Local multi-GPU DDP — if you have the hardware
```bash
sair train --ddp              # uses all available GPUs
sair train --ddp --nproc 2    # use exactly 2 GPUs
```

Launches `torchrun` under the hood. Requires CUDA and multiple GPUs on the same machine.  
Same trainerV4 code from Notebook 4 — `train/ddp_trainer.py` is readable and hackable.

---

## Generation strategies

| Flag | What it does |
|------|-------------|
| `--method greedy` | Always pick the most probable token |
| `--method nucleus` | Sample from the top-p probability mass |
| `--method top_k` | Sample from the top-k tokens |
| `--beams N` | Beam search — explore N sequences in parallel |
| `--temperature T` | `< 1` = focused, `> 1` = creative |

---

## Based on

- [SAIR Jr. — Module 5: GPT from Scratch](https://github.com/SAIR-Org/SAIR_Jr/tree/main/5_GPT%20from%20scratch) — the notebooks this project implements
- Raschka, *Build a Large Language Model From Scratch*, Manning 2024
- Vaswani et al., *Attention Is All You Need*, 2017

<p align="center">
  <img src="ui/SAiR_logo.jpg" alt="SAIR Logo" width="160"/>
</p>

<h1 align="center">SAIR miniGPT</h1>

<p align="center">
  <b>Build a GPT. Train it. Talk to it.</b><br/>
  A full-stack, hackable GPT playground — from raw text to a live web UI.
</p>

<p align="center">
  <a href="https://github.com/SAIR-Org/SAIR_Jr/tree/main/5_GPT%20from%20scratch">
    <img src="https://img.shields.io/badge/SAIR%20Jr.-Module%205%20Capstone-blue?style=flat-square" alt="Module 5 Capstone"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.12%2B-brightgreen?style=flat-square" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-orange?style=flat-square" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/tests-39%20passing-success?style=flat-square" alt="Tests"/>
  <img src="https://img.shields.io/badge/package%20manager-uv-purple?style=flat-square" alt="uv"/>
</p>

---

> This is the **capstone project** for [SAIR Jr. — Module 5: GPT from Scratch](https://github.com/SAIR-Org/SAIR_Jr/tree/main/5_GPT%20from%20scratch).
> Every function here (`GPTModel`, `generateV0`→`V3`, `trainerV3`, beam search) maps 1-to-1 to a notebook cell you already wrote.
> **If you haven't finished the notebooks yet, do that first — then come back here and bring it to life.**

---

## What is this?

You spent Module 5 building a GPT from scratch — attention heads, transformer blocks, training loops, and all.
**miniGPT packages all of that into a real, runnable system** with:

- A `sair` CLI so you can go from data to trained model in a few commands
- Support for `.txt` and `.pdf` files as training data
- Three training modes: local CPU/GPU, Modal A100 cloud, multi-GPU DDP
- A web UI to chat with your model in the browser
- The option to skip training entirely and load a pretrained GPT-2 from HuggingFace

---

## Quickstart

> **Prerequisites:** Python 3.12+, [uv](https://github.com/astral-sh/uv) (`pip install uv`)

```bash
# 1. Clone
git clone https://github.com/SAIR-Org/miniGPT
cd miniGPT
uv sync           # creates .venv, installs everything, registers the `sair` command

# 2. Drop your data in data/raw/
#    Any .txt or .pdf file works — books, papers, articles, whatever you like
cp my_book.txt  data/raw/
cp my_paper.pdf data/raw/

# 3. Tokenize
sair prepare

# 4. Train
sair train                    # local CPU or GPU — start here
sair train --modal            # Modal A100 cloud (see below)
sair train --ddp              # multi-GPU on your own machine

# 5. Generate text from the command line
sair generate "Once upon a time"
sair generate "In the beginning" --method nucleus --temperature 0.9 --beams 3

# 6. Launch the web UI
sair ui
# → open http://localhost:7860
```

> **No GPU? No problem.** Set `MODEL_PRESET = "tiny"` in `config.py` and train on CPU in minutes.

---

## One file to rule them all — `config.py`

Everything tuneable lives here. Most users only ever touch this file.

```python
# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PRESET = "small"        # "tiny" | "small" | "gpt2-124m"

# ── Training ─────────────────────────────────────────────────────────────────
NUM_EPOCHS    = 10
LEARNING_RATE = 3e-4
BATCH_SIZE    = 16

# ── Generation defaults ───────────────────────────────────────────────────────
GEN_TEMPERATURE = 0.9
GEN_TOP_K       = 50
GEN_TOP_P       = 0.9
GEN_BEAMS       = 1
```

| Preset | Parameters | Good for |
|--------|-----------|----------|
| `tiny` | ~500 K | Fast iteration, CPU, debugging |
| `small` | ~30 M | Laptop GPU, real experiments |
| `gpt2-124m` | 124 M | Full GPT-2 size, serious training |

---

## Training options

### Option 1 — Local (CPU or GPU)
```bash
sair train
```
The default. Works on any machine. Switch to `"tiny"` preset for fast experiments.

### Option 2 — Modal A100 (recommended for real training)

[Modal](https://modal.com) gives you a free-tier A100 in the cloud with one command. No setup beyond a browser login.

```bash
modal token new       # one-time login
sair train --modal    # runs on A100, streams logs back to your terminal
```

Checkpoints are saved to a persistent Modal volume — safe between runs.
Change `gpu="A100"` → `gpu="T4"` in `train/modal_train.py` for a cheaper option.

### Option 3 — Multi-GPU DDP
```bash
sair train --ddp              # use all available GPUs
sair train --ddp --nproc 2    # use exactly 2
```
Launches `torchrun` under the hood. Code is in `train/ddp_trainer.py` — same pattern as Notebook 4, fully readable.

---

## Skip training — load GPT-2 from HuggingFace

Already have a model? Load OpenAI's pretrained GPT-2 weights and start generating immediately:

```bash
sair generate "The meaning of life is" --hf gpt2
sair generate "The meaning of life is" --hf gpt2-medium
sair ui --hf gpt2-large
```

Supported variants: `gpt2` (124M) · `gpt2-medium` (355M) · `gpt2-large` (774M) · `gpt2-xl` (1.5B)

---

## Generation strategies

| Flag | What it does |
|------|-------------|
| `--method greedy` | Always pick the highest-probability token (deterministic) |
| `--method top_k` | Sample from the top-K tokens |
| `--method nucleus` | Sample from tokens covering the top-P probability mass |
| `--beams N` | Beam search — explore N candidate sequences in parallel |
| `--temperature T` | `< 1.0` = focused · `1.0` = balanced · `> 1.0` = creative |

---

## Project structure

```
miniGPT/
├── config.py               ← start here — all hyperparams and paths
├── cli.py                  ← sair prepare | train | generate | ui
│
├── data/
│   ├── prepare.py          ← reads .txt + .pdf, tokenizes, saves .bin files
│   └── dataset.py          ← GPT2Dataset + DataLoader helpers
│
├── model/
│   └── gpt.py              ← GPTModel (LayerNorm → MHA → FFN → TransformerBlock)
│
├── train/
│   ├── trainer.py          ← trainerV3: grad accumulation + cosine LR + grad clip
│   ├── ddp_trainer.py      ← trainerV4: DistributedDataParallel via torchrun
│   └── modal_train.py      ← trainerV3 wrapped for Modal A100
│
├── inference/
│   ├── generate.py         ← generateV0 (greedy) → V1 (temp) → V2 (top-k/p) → V3 (beam)
│   └── load_weights.py     ← HuggingFace GPT-2 weight loading
│
├── ui/
│   ├── server.py           ← FastAPI backend
│   └── index.html          ← SAIR-branded dark web UI
│
└── tests/                  ← 39 tests covering the full pipeline
```

---

## Acknowledgements

- [SAIR Jr. — Module 5: GPT from Scratch](https://github.com/SAIR-Org/SAIR_Jr/tree/main/5_GPT%20from%20scratch) — the course this project is built on
- Raschka, *Build a Large Language Model From Scratch*, Manning 2024
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017

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

> **Capstone project for [SAIR Jr. — Module 5: GPT from Scratch](https://github.com/SAIR-Org/SAIR_Jr/tree/main/5_GPT%20from%20scratch).**
> Every function here (`GPTModel`, `generateV0`→`V3`, `trainerV3`, beam search) maps 1-to-1 to a notebook cell you already wrote.
> **Haven't finished the notebooks yet? Start there first — then come back here.**

---

## What is this?

You built a GPT from scratch in Module 5. miniGPT packages all of that into a real, runnable system:

- A `sair` CLI — go from raw text to a trained model in a few commands
- `.txt` and `.pdf` files as training data
- Three training modes: local CPU/GPU · Modal A100 cloud · multi-GPU DDP
- A web UI to chat with your model in the browser
- Load any pretrained GPT-2 (124M → 1.5B) without training at all

---

## Which path are you on?

**Choose one to get started:**

| | Path A — Train your own GPT | Path B — Use a pretrained GPT-2 |
|---|---|---|
| **What you need** | Text or PDF files to train on | Nothing — weights download automatically |
| **Time to first output** | Minutes (tiny) to hours (medium) | ~2 minutes |
| **Jump to** | [Step 1 below](#step-1-install) | [Skip to: Load pretrained GPT-2](#skip-training--load-pretrained-gpt-2) |

---

## Path A — Train your own GPT

### Step 1 — Install

You need **Python 3.12+** and **uv** (a fast Python package manager).

```bash
# Install uv if you don't have it
pip install uv

# Clone the repo
git clone https://github.com/SAIR-Org/miniGPT
cd miniGPT

# Set up the environment (creates .venv and installs all dependencies)
uv sync
```

> You should see: `All packages installed. Resolved N packages.`
> The `sair` command is now available inside this environment. Run all commands with `uv run sair ...` or activate the venv first with `source .venv/bin/activate`.

---

### Step 2 — Pick your model size

Open **`config.py`** and set `MODEL_PRESET` to match your hardware:

```python
MODEL_PRESET = "tiny"    # ← change this line
```

| Preset | Params | Context window | Best for |
|--------|--------|---------------|---------|
| `tiny` | ~10 M | 256 tokens | No GPU — fast, good for testing |
| `small` | ~50 M | 512 tokens | Laptop GPU (4–6 GB) |
| `medium` | ~100 M | 1024 tokens | Dedicated GPU (8 GB+) |
| `custom` | you decide | you decide | [Define your own](#build-your-own-architecture) |

> **Not sure?** Start with `"tiny"`. You can always retrain with a bigger preset.

---

### Step 3 — Add your training data

Create the data folder and drop in your files:

```bash
mkdir -p data/raw
cp my_book.txt   data/raw/      # .txt files work
cp my_paper.pdf  data/raw/      # .pdf files work too — mix freely
```

Any plain text or PDF works — a novel, Wikipedia articles, research papers, anything.
The more text, the better the model.

> **Don't have any data handy?** Download a free book from [Project Gutenberg](https://www.gutenberg.org) and save it as a `.txt` file.

---

### Step 4 — Tokenize

```bash
sair prepare
```

This reads everything in `data/raw/`, tokenizes it with the GPT-2 tokenizer, and saves the result to `data/processed/`.

> You should see something like: `Tokenized 1,234,567 tokens → data/processed/train_ids.bin`

---

### Step 5 — Train

Pick the option that matches your setup:

**Option A — Local (CPU or GPU)**
```bash
sair train
```
Works on any machine. On CPU with the `tiny` preset, expect ~5–10 min per epoch.
Watch the loss go down — that's your model learning.

**Option B — Modal A100 cloud** *(recommended for real training)*

[Modal](https://modal.com) gives you an A100 GPU in the cloud for free (within limits). One-time setup:

```bash
pip install modal          # install the Modal client
modal token new            # opens browser for login — do this once
sair train --modal         # launches on A100, streams logs to your terminal
```

Checkpoints save to a persistent Modal volume between runs.
Want a cheaper GPU? Change `gpu="A100"` to `gpu="T4"` in `train/modal_train.py`.

**Option C — Multi-GPU DDP** *(if you have multiple GPUs)*
```bash
sair train --ddp              # uses all available GPUs automatically
sair train --ddp --nproc 2    # or specify exactly how many
```

---

### Step 6 — Generate text

```bash
sair generate "Once upon a time"
```

Try different strategies:

```bash
# Greedy — always picks the most likely next token (deterministic)
sair generate "Once upon a time" --method greedy

# Nucleus sampling — more creative
sair generate "Once upon a time" --method nucleus --temperature 0.9

# Beam search — explores multiple paths and picks the best
sair generate "Once upon a time" --beams 3
```

| Flag | Effect |
|------|--------|
| `--method greedy` | Deterministic, repetitive |
| `--method nucleus` | Natural, varied |
| `--method top_k` | Sample from top K tokens |
| `--beams N` | Beam search — N candidate sequences |
| `--temperature T` | `< 1` = focused · `> 1` = creative |
| `--max-tokens N` | How many tokens to generate |

---

### Step 7 — Open the web UI

```bash
sair ui
```

Then open **http://localhost:7860** in your browser.
You'll see a chat interface where you can type prompts, adjust settings, and generate text interactively.

---

## Path B — Skip training, load pretrained GPT-2

No data, no training required. Download OpenAI's pretrained weights and start immediately:

```bash
# Install and set up (same as Step 1 above)
git clone https://github.com/SAIR-Org/miniGPT
cd miniGPT
uv sync

# Generate text straight away
sair generate "The future of AI is" --hf gpt2

# Or open the full web UI
sair ui --hf gpt2-medium
```

The weights download automatically on first use and are cached locally.

**Available variants:**

| Flag | Params | Notes |
|------|--------|-------|
| `--hf gpt2` or `--hf gpt2-124m` | 124 M | Fastest, lightest |
| `--hf gpt2-medium` or `--hf gpt2-355m` | 355 M | Good balance |
| `--hf gpt2-large` or `--hf gpt2-774m` | 774 M | Needs 4 GB+ RAM |
| `--hf gpt2-xl` or `--hf gpt2-1558m` | 1.5 B | Needs 8 GB+ RAM |

---

## Build your own architecture

Want a model that's bigger, smaller, or just different? Edit the `"custom"` entry in `config.py`:

```python
MODEL_PRESET = "custom"

MODELS["custom"] = {
    "vocab_size"    : 50257,   # keep this — matches the GPT-2 tokenizer
    "context_length": 512,     # tokens the model sees at once (more = more memory)
    "emb_dim"       : 384,     # embedding size
    "n_heads"       : 6,       # attention heads — emb_dim must be divisible by n_heads
    "n_layers"      : 6,       # number of transformer blocks
    "drop_rate"     : 0.1,     # dropout regularisation (0.0 = off)
    "qkv_bias"      : False,   # True matches official GPT-2; False is fine for scratch
}
```

> Rule of thumb: doubling both `emb_dim` and `n_layers` roughly 4× the parameter count.

---

## Project structure

Once you're comfortable using the project, here's how the code is organised — every file is short and readable:

```
miniGPT/
├── config.py               ← all hyperparams and paths — start here
├── cli.py                  ← sair prepare | train | generate | ui
│
├── data/
│   ├── prepare.py          ← reads .txt + .pdf, tokenizes, saves .bin files
│   └── dataset.py          ← GPT2Dataset + DataLoader
│
├── model/
│   └── gpt.py              ← GPTModel: LayerNorm → MHA → FFN → TransformerBlock
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

## Design philosophy — intentionally hackable

miniGPT is deliberately **not** DRY (Don't Repeat Yourself).

`trainer.py`, `ddp_trainer.py`, and `modal_train.py` each contain their own full training loop. `generate.py` has four versions of the same function — `generateV0` through `V3` — each one adding exactly one idea. This is a conscious choice:

- **Each file is self-contained.** You can read, edit, or break any one of them without touching the others.
- **Each version is a learning step.** Want to understand beam search? Read `generateV3` in isolation. Want to understand gradient accumulation? Read `trainer.py` top to bottom.
- **No abstraction hides the detail.** A real production system would share a single training loop and call it from three entry points. That's good software engineering — but it's bad for learning. Here you can see the full picture in every file.

If you want to apply proper software engineering principles (DRY, abstraction layers, shared utilities), that is a great exercise — and this codebase is small enough to refactor cleanly.

For a production-grade LLM system built with those principles from scratch, see the founder's project:

> **[MyLLM](https://github.com/silvaxxx1/MyLLM)** — a serious, optimised LLM system built from scratch with clean architecture, designed for students ready to go beyond the playground.

---

## Acknowledgements

- [SAIR Jr. — Module 5: GPT from Scratch](https://github.com/SAIR-Org/SAIR_Jr/tree/main/5_GPT%20from%20scratch) — the course this project implements
- Raschka, *Build a Large Language Model From Scratch*, Manning 2024
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017

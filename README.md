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

## 📹 Demo

<p align="center">
  <img src="sair_gpt_demo.gif" alt="SAIR miniGPT Demo" width="800"/>
</p>

<p align="center">
  <i>Train → Generate → Chat — all in one system</i>
</p>

---

## 🎯 What is this?

> **Capstone project for [SAIR Jr. — Module 5: GPT from Scratch](https://github.com/SAIR-Org/SAIR_Jr/tree/main/5_GPT%20from%20scratch).**
> Every function here (`GPTModel`, `generateV0`→`V3`, `trainerV3`, beam search) maps 1-to-1 to a notebook cell you already wrote.
> **Haven't finished the notebooks yet? Start there first — then come back here.**

You built a GPT from scratch in Module 5. miniGPT packages all of that into a real, runnable system:

| Feature | What it does |
|---------|--------------|
| 🖥️ **`sair` CLI** | Go from raw text to trained model in a few commands |
| 📄 **Multi-format data** | `.txt` and `.pdf` files as training data |
| ☁️ **Flexible training** | Local CPU/GPU · Modal A100 cloud · multi-GPU DDP |
| 📊 **W&B + plots** | Live loss curves in your browser + PNG saved after training |
| 🌐 **Web UI** | Chat with your model in the browser |
| 🚀 **Pretrained models** | Load GPT-2 (124M → 1.5B) without training |

---

## 🗺️ Choose your path

| | Path A — Train your own GPT | Path B — Use pretrained GPT-2 |
|---|---|---|
| **What you need** | Text or PDF files | Nothing — weights auto-download |
| **Time to first output** | Minutes (tiny) to hours (medium) | ~2 minutes |
| **Jump to** | [Step 1 below](#-step-1-install) | [Skip to Path B](#path-b--skip-training-load-pretrained-gpt-2) |

---

## Path A — Train your own GPT

### ⚡ Step 1 — Install

You need **Python 3.12+** and **uv** (fast Python package manager).

```bash
# Install uv if you don't have it
pip install uv

# Clone the repo
git clone https://github.com/SAIR-Org/miniGPT
cd miniGPT

# Set up the environment
uv sync
```

> ✅ You should see: `All packages installed. Resolved N packages.`
> All commands use `uv run sair ...` — works immediately without activating the venv.

---

### 📂 Step 2 — Add your training data

Drop any `.txt` or `.pdf` files into `data/raw/`:

```bash
cp my_book.txt  data/raw/
cp my_paper.pdf data/raw/
```

Any text works — novels, Wikipedia, research papers. More text = better model.

> **No data handy?** Download a free book from [Project Gutenberg](https://www.gutenberg.org).
>
> **Using Harry Potter books?** The SAIR repo has 6 of the 7 books at:
> `4_Applied Deep Learning with PyTorch/3_Sequence and NLP/harry_potter_txt/`
> ```bash
> cp "../4_Applied Deep Learning with PyTorch/3_Sequence and NLP/harry_potter_txt/"*.txt data/raw/
> ```

---

### 🔧 Step 3 — Tokenize

```bash
uv run sair prepare
```

Reads everything in `data/raw/`, strips formatting artifacts, tokenizes with the GPT-2 tokenizer, saves to `data/processed/`.

Expected output for 6 Harry Potter books:
```
Loading corpus from data/raw ...
  [txt] Book 1 - The Philosopher's Stone.txt
  [txt] Book 3 - The Prisoner of Azkaban.txt
  ...
Total characters : 6,233,476

  train:  1,740,472 tokens  →  data/processed/train_ids.bin
  val  :    137,152 tokens  →  data/processed/val_ids.bin
  test :     56,651 tokens  →  data/processed/test_ids.bin

Done. Ready to train.
```

---

### 🎛️ Step 4 — Pick your model size

| Preset | Params | Context | Best for |
|--------|--------|---------|----------|
| `tiny` | ~10 M | 256 tokens | No GPU — fast testing, first run |
| `small` | ~50 M | 512 tokens | Laptop GPU |
| `medium` | ~163 M | 1024 tokens | Cloud GPU — best quality |
| `custom` | you decide | you decide | [Custom architecture](#-build-your-own-architecture) |

> The param count includes the full vocabulary embedding table (50,257 × 768 ≈ 38M), which is why `medium` is 163M rather than the ~124M you might expect.

**Start with `tiny` if you have no GPU** — it trains in minutes on CPU and lets you verify the full pipeline works before committing to a longer run.

Open `config.py` and set:
```python
MODEL_PRESET = "tiny"    # ← start here, change when ready
```

---

### 🚂 Step 5 — Train locally

```bash
uv run sair train
```

This runs on your machine using whatever hardware you have (CPU or GPU).

**Expected time per epoch:**

| Hardware | `tiny` | `small` | `medium` |
|----------|--------|---------|---------|
| CPU only | ~5–10 min | ~30–60 min | not recommended |
| Laptop GPU (4 GB) | ~1–2 min | ~5–10 min | out of memory |
| Desktop GPU (8 GB+) | ~30 sec | ~2–3 min | ~8–10 min |

> **No GPU and too slow?** Jump to [Modal cloud training](#️-scale-up--modal-cloud-training) to run on an A100.

**Multi-GPU (if you have more than one GPU):**
```bash
uv run sair train --ddp              # uses all GPUs
uv run sair train --ddp --nproc 2    # specify count
```

---

### 💬 Step 6 — Generate text

```bash
uv run sair generate "Harry Potter walked into"
```

The CLI automatically loads the latest checkpoint from `checkpoints/`.

**Generation strategies:**

| Method | Flag | Effect |
|--------|------|--------|
| Nucleus (default) | `--method nucleus` | Natural, varied output |
| Top-K | `--method top_k` | Sample from top K tokens |
| Greedy | `--method greedy` | Deterministic, can be repetitive |
| Beam search | `--beams 3` | Explores multiple paths |

**Useful flags:**
- `--temperature 0.7` — lower = more focused, higher = more creative
- `--max-tokens 200` — how many tokens to generate (default: 100)

---

### 🌐 Step 7 — Open the web UI

```bash
uv run sair ui
```

Then open **http://localhost:7860** in your browser.

**What weights does it use?**
- By default → loads the **latest `epoch_XX.pt`** from your local `checkpoints/` folder
- With `--hf` flag → loads **pretrained GPT-2** from HuggingFace (no training needed)

```bash
uv run sair ui              # your trained model
uv run sair ui --hf gpt2    # OpenAI pretrained GPT-2 124M
```

> **No checkpoint yet?** Use `--hf gpt2` to demo with pretrained weights immediately.

---

## ☁️ Scale Up — Modal Cloud Training

Once you've verified the pipeline works locally, use Modal to train on a cloud A100 GPU — much faster and enables bigger models.

### What is Modal?

[Modal](https://modal.com) is a cloud compute platform. You write normal Python, decorate it with `@app.function(gpu="A100")`, and it runs on a cloud GPU. No server setup, no SSH, no Docker knowledge required.

### Step 1 — Create a Modal account

Go to `modal.com` and sign up with GitHub.

**Free tier:** You get **$5 immediately** (no card needed). Add a credit card to unlock **$30/month**.

| GPU | $/hr | 30 epochs on `medium` |
|-----|------|----------------------|
| T4  | ~$0.59 | ~8–10 hrs → ~$6 |
| A100 | ~$3.70 | ~3 hrs → ~$12–15 |

> A100 finishes 3–4× faster — often cheaper overall for large runs.

### Step 2 — Authenticate

```bash
uv run python -m modal token new
```

Opens your browser. Click approve and come back.

> ⚠️ Always use `uv run python -m modal` — not just `modal`. The `modal` binary in the venv has a broken shebang.

### Step 3 — Set up W&B for live loss curves

Get your API key at `wandb.ai/authorize`, then:

```bash
uv run python -m modal secret create wandb-secret WANDB_API_KEY=your_key_here
```

Skip this step if you don't want W&B — training will fall back silently.

### Step 4 — Launch training

> ⚠️ Always specify `::main` or `::download` explicitly — the file has two entrypoints so Modal requires it.

```bash
uv run python -m modal run train/modal_train.py::main
```

Modal will:
- Build a Docker image with all dependencies (~2 min, cached after first run)
- Upload your code + tokenized data
- Spin up the GPU and start training
- Stream logs to your terminal in real time
- Print a W&B URL — open it to watch loss curves live

### Step 5 — Download your checkpoint

Training runs on Modal's cloud — the checkpoint lives there, not on your machine. Pull it down:

```bash
uv run python -m modal run train/modal_train.py::download
```

This downloads the latest `epoch_XX.pt` + `loss_curve.png` into your local `checkpoints/` folder.

### Step 6 — Run the UI

```bash
uv run sair ui    # → http://localhost:7860
```

---

## 🔁 Training from scratch vs. resuming

#### Train from scratch (default)

Starts with random weights. Use this when:
- You're training for the first time
- You changed the model size (`small` → `medium`) — **must start fresh if architecture changes**

#### Resume from a checkpoint

Picks up where a previous run left off — same weights, optimizer state, and LR schedule. Use this when training was interrupted or you want more epochs on the same model.

```python
train(
    model        = model,
    ...
    resume_from  = "/checkpoints/epoch_05.pt",  # ← continues from epoch 6
    num_epochs   = 10,                           # ← total target (not additional)
)
```

> ⚠️ **You cannot resume across model sizes.** `small` → `medium` means weight shapes are incompatible — start fresh.

The checkpoint saves everything needed:
```python
{"epoch": 5, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
```

---

## Path B — Skip training, load pretrained GPT-2

No data. No training. Start immediately:

```bash
git clone https://github.com/SAIR-Org/miniGPT
cd miniGPT
uv sync

# Generate instantly
uv run sair generate "The future of AI is" --hf gpt2

# Or open the full web UI
uv run sair ui --hf gpt2-medium
```

**Available variants:**

| Flag | Params | Notes |
|------|--------|-------|
| `--hf gpt2` or `--hf gpt2-124m` | 124 M | Fastest, lightest |
| `--hf gpt2-medium` or `--hf gpt2-355m` | 355 M | Good balance |
| `--hf gpt2-large` or `--hf gpt2-774m` | 774 M | Needs 4 GB+ RAM |
| `--hf gpt2-xl` or `--hf gpt2-1558m` | 1.5 B | Needs 8 GB+ RAM |

Weights download automatically on first use and cache locally.

---

## 📊 Real Training Example — Harry Potter

### Run 1 — tiny model, 5 epochs (verify pipeline)

**Setup:** `tiny` (~10M params, 256 context) · CPU · 6 Harry Potter books  
**Purpose:** confirm everything works before investing GPU time

### Run 2 — small model, 5 epochs (first real run)

**Setup:** `small` (~50M params, 512 context) · Modal A100 · 5 epochs · ~$2.50

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | ~5.2 | ~5.4 |
| 2 | ~4.3 | ~4.5 |
| 3 | ~3.9 | ~4.1 |
| 4 | ~3.6 | ~3.8 |
| 5 | **3.49** | **3.74** |

**Generated sample after 5 epochs:**
```
Prompt: "Harry Potter walked into"

Harry Potter walked into the Phoenix - J. Rowling
"So it't you't let us if they't him?" Harry said Mr. "Why
you know I mean ...'re going to kill me, he've got to yourself."
```

Character names, dialogue structure, and HP vocabulary appear after just 5 epochs. Not fully coherent yet — that improves with a bigger model and more epochs.

### Run 3 — medium model, 30 epochs (full quality run)

**Setup:** `medium` (~163M params, 1024 context) · Modal A100 · 30 epochs · ~$12–15

The 1024-token context lets the model learn longer-range structure — multi-sentence dialogue, paragraph flow, consistent character voice.

<p align="center">
  <img src="assets/loss_curve.png" alt="Train vs Val Loss — Harry Potter 30 epochs" width="700"/>
</p>

**Best inference settings for this model:**
```bash
uv run sair generate "Harry Potter walked into" --temperature 0.7 --max-tokens 200
```

---

## 📁 Project structure

Every file is short, readable, and self-contained:

```
miniGPT/
├── config.py               ← all hyperparams — start here
├── cli.py                  ← sair prepare | train | generate | ui
│
├── data/
│   ├── prepare.py          ← reads .txt + .pdf, cleans, tokenizes, saves .bin
│   └── dataset.py          ← GPT2Dataset + DataLoader
│
├── model/
│   └── gpt.py              ← GPTModel: LayerNorm → MHA → FFN → Block
│
├── train/
│   ├── trainer.py          ← trainerV3: grad accumulation + cosine LR + W&B + plots
│   ├── ddp_trainer.py      ← trainerV4: DistributedDataParallel
│   └── modal_train.py      ← trainerV3 wrapped for Modal cloud GPU
│
├── inference/
│   ├── generate.py         ← generateV0 (greedy) → V3 (beam search)
│   └── load_weights.py     ← HuggingFace GPT-2 weight loading
│
├── ui/
│   ├── server.py           ← FastAPI backend
│   └── index.html          ← SAIR-branded dark web UI
│
└── tests/                  ← 39 tests covering full pipeline
```

---

## 🔨 Build your own architecture

Edit the `"custom"` entry in `config.py`:

```python
MODEL_PRESET = "custom"

MODELS["custom"] = {
    "vocab_size"    : 50257,   # keep this — matches GPT-2 tokenizer
    "context_length": 512,     # tokens the model sees at once
    "emb_dim"       : 384,     # embedding size
    "n_heads"       : 6,       # attention heads (emb_dim must be divisible by this)
    "n_layers"      : 6,       # number of transformer blocks
    "drop_rate"     : 0.1,     # dropout regularization
    "qkv_bias"      : False,   # True matches official GPT-2
}
```

> **Rule of thumb:** Doubling both `emb_dim` and `n_layers` roughly 4× the parameter count.

---

## 📈 W&B + Matplotlib integration

Training automatically logs to [Weights & Biases](https://wandb.ai) and saves a loss plot.

**What gets logged:**
- Every eval step: `train/loss`, `val/loss`, `learning_rate`, `tokens_seen`
- Every epoch: generated text sample as W&B artifact
- After training: loss curve PNG saved to `checkpoints/loss_curve.png`

**To disable W&B:**
```bash
# Just don't create the wandb-secret — training falls back silently
# Or pass use_wandb=False directly to train()
```

---

## 🎓 Design philosophy — intentionally hackable

miniGPT is deliberately **not** DRY (Don't Repeat Yourself).

- `trainer.py`, `ddp_trainer.py`, and `modal_train.py` each contain their own full training loop
- `generate.py` has four versions — `generateV0` through `V3` — each adding one idea

**Why?**
- **Each file is self-contained** — read, edit, or break any one without touching others
- **Each version is a learning step** — want to understand beam search? Read `generateV3`
- **No abstraction hides the detail** — see the full picture in every file

For a production-grade LLM system with clean architecture, see:

> **[MyLLM](https://github.com/silvaxxx1/MyLLM)** — optimized LLM system from scratch, designed for students ready to go beyond the playground.

---

## 🧪 Testing

```bash
uv run python -m pytest tests/ -v
```

```
============================= test session starts ==============================
collected 39 items

tests/test_config.py .....                                              [ 12%]
tests/test_data.py ......                                               [ 28%]
tests/test_generate.py .............                                    [ 61%]
tests/test_model.py ......                                              [ 76%]
tests/test_server.py .....                                              [ 89%]
tests/test_trainer.py .....                                             [100%]

39 passed in 8.65s
```

---

## 🐛 Known gotchas

| Problem | Fix |
|---------|-----|
| `modal: command not found` | Use `uv run python -m modal` instead of `modal` |
| `Specify a Modal Function or local entrypoint` | Always use `::main` or `::download` — the file has two entrypoints so Modal requires it |
| `CUDA out of memory` locally | Your local GPU is too small for `medium` — use `tiny`/`small` locally, or run on Modal A100 |
| Resumed run but loss jumped up | You changed model size between runs — architectures are incompatible, start fresh |
| Model generates `Page \| 548 Harry Potter...` | Run `uv run sair prepare` again — `prepare.py` strips page headers automatically |
| UI returns `{"error": "Model not loaded"}` | No checkpoint in `checkpoints/` — train first or use `--hf gpt2` |

---

## 🙏 Acknowledgements

- [SAIR Jr. — Module 5: GPT from Scratch](https://github.com/SAIR-Org/SAIR_Jr/tree/main/5_GPT%20from%20scratch) — the course this project implements
- Raschka, *Build a Large Language Model From Scratch*, Manning 2024
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017

---

## 📄 License

MIT — free for learning and building.

---

<p align="center">
  <b>Built with ⚡ and 🧠 by SAIR</b>
</p>

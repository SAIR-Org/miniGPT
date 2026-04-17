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

> **Capstone project for [SAIR Jr. — Module 5: GPT from Scratch](https://github.com/SAIR-Org/SAIR_Jr/tree/main/5_GPT%20from%20scratch).**
> Every function here (`GPTModel`, `generateV0`→`V3`, `trainerV3`, beam search) maps 1-to-1 to a notebook cell you already wrote.
> **Haven't finished the notebooks yet? Start there first — then come back here.**

---

## 🎯 What is this?

You built a GPT from scratch in Module 5. miniGPT packages all of that into a real, runnable system:

| Feature | What it does |
|---------|--------------|
| 🖥️ **`sair` CLI** | Go from raw text to trained model in a few commands |
| 📄 **Multi-format data** | `.txt` and `.pdf` files as training data |
| ☁️ **Flexible training** | Local CPU/GPU · Modal A100 cloud · multi-GPU DDP |
| 🌐 **Web UI** | Chat with your model in the browser |
| 🚀 **Pretrained models** | Load GPT-2 (124M → 1.5B) without training |

---

## 🗺️ Which path are you on?

**Choose one to get started:**

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

### 🎛️ Step 2 — Pick your model size

Open **`config.py`** and set `MODEL_PRESET`:

```python
MODEL_PRESET = "tiny"    # ← change this line
```

| Preset | Params | Context | Best for |
|--------|--------|---------|----------|
| `tiny` | ~10 M | 256 tokens | No GPU — fast testing |
| `small` | ~50 M | 512 tokens | Laptop GPU (4–6 GB) |
| `medium` | ~100 M | 1024 tokens | Dedicated GPU (8 GB+) |
| `custom` | you decide | you decide | [Custom architecture](#build-your-own-architecture) |

> **Not sure?** Start with `"tiny"`. You can always retrain with a bigger preset.

---

### 📂 Step 3 — Add your training data

```bash
mkdir -p data/raw
cp my_book.txt   data/raw/      # .txt files work
cp my_paper.pdf  data/raw/      # .pdf files work too
```

Any text works — novels, Wikipedia, research papers. More text = better model.

> **No data handy?** Download a free book from [Project Gutenberg](https://www.gutenberg.org).

---

### 🔧 Step 4 — Tokenize

```bash
uv run sair prepare
```

Reads everything in `data/raw/`, tokenizes with GPT-2 tokenizer, saves to `data/processed/`.

> ✅ Expected: `Tokenized 1,234,567 tokens → data/processed/train_ids.bin`

---

### 🚂 Step 5 — Train

**Option A — Local (CPU or GPU)**
```bash
uv run sair train
```
On CPU with `tiny` preset: ~5–10 min per epoch.

**Option B — Modal A100 cloud** *(recommended)*

[Modal](https://modal.com) gives free A100 GPU access (within limits):

```bash
pip install modal
modal token new                # one-time login
uv run sair train --modal      # launches on A100
```

**Option C — Multi-GPU DDP**
```bash
uv run sair train --ddp              # uses all GPUs
uv run sair train --ddp --nproc 2    # specify count
```

---

### 💬 Step 6 — Generate text

> Requires a trained checkpoint. Run Step 5 first. Or use [Path B](#path-b--skip-training-load-pretrained-gpt-2).

```bash
uv run sair generate "Once upon a time"
```

**Generation strategies:**

| Method | Command | Effect |
|--------|---------|--------|
| Greedy | `--method greedy` | Deterministic, repetitive |
| Nucleus | `--method nucleus --temperature 0.9` | Natural, varied |
| Top-K | `--method top_k` | Sample from top K tokens |
| Beam search | `--beams 3` | Explores multiple paths |

**Additional flags:**
- `--temperature T` — `<1` focused · `>1` creative
- `--max-tokens N` — how many tokens to generate

---

### 🌐 Step 7 — Open the web UI

```bash
uv run sair ui
```

Then open **http://localhost:7860** in your browser.

> **No checkpoint?** Use `uv run sair ui --hf gpt2` (see Path B).

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

## 🔨 Build your own architecture

Edit the `"custom"` entry in `config.py`:

```python
MODEL_PRESET = "custom"

MODELS["custom"] = {
    "vocab_size"    : 50257,   # keep this — matches GPT-2 tokenizer
    "context_length": 512,     # tokens the model sees at once
    "emb_dim"       : 384,     # embedding size
    "n_heads"       : 6,       # attention heads (emb_dim divisible by this)
    "n_layers"      : 6,       # number of transformer blocks
    "drop_rate"     : 0.1,     # dropout regularization
    "qkv_bias"      : False,   # True matches official GPT-2
}
```

> **Rule of thumb:** Doubling both `emb_dim` and `n_layers` roughly 4× the parameter count.

---

## 📁 Project structure

Every file is short, readable, and self-contained:

```
miniGPT/
├── config.py               ← all hyperparams — start here
├── cli.py                  ← sair prepare | train | generate | ui
│
├── data/
│   ├── prepare.py          ← reads .txt + .pdf, tokenizes, saves .bin
│   └── dataset.py          ← GPT2Dataset + DataLoader
│
├── model/
│   └── gpt.py              ← GPTModel: LayerNorm → MHA → FFN → Block
│
├── train/
│   ├── trainer.py          ← trainerV3: grad accumulation + cosine LR
│   ├── ddp_trainer.py      ← trainerV4: DistributedDataParallel
│   └── modal_train.py      ← trainerV3 wrapped for Modal A100
│
├── inference/
│   ├── generate.py         ← generateV0 (greedy) → V3 (beam search)
│   └── load_weights.py     ← HuggingFace GPT-2 weight loading
│
├── ui/
│   ├── server.py           ← FastAPI backend
│   └── index.html          ← SAIR-branded dark web UI
│
├── tests/                  ← 39 tests covering full pipeline
└── sair_gpt_demo.gif       ← demo animation
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
# Run the test suite
uv run pytest tests/ -v
```

```
============================= test session starts ==============================
collected 39 items

tests/test_data.py .......                                              [ 17%]
tests/test_model.py ............                                        [ 48%]
tests/test_training.py .........                                        [ 71%]
tests/test_inference.py .......                                         [ 89%]
tests/test_cli.py ....                                                  [100%]

============================== 39 passed ==============================
```

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
```

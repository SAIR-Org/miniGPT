"""
ui/server.py — FastAPI backend. One endpoint, that's it.

Usage:
    sair ui
    python ui/server.py    # direct
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import tiktoken
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from config import MODEL_CONFIG, CKPT_DIR, GEN_MAX_TOKENS, GEN_TEMPERATURE, GEN_TOP_K, GEN_TOP_P, GEN_BEAMS
from inference.generate import generate, load_model

# ── Load model once at startup ────────────────────────────────────────────────

device    = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = tiktoken.encoding_for_model("gpt2")
model     = None
config    = MODEL_CONFIG
_tokenizer_override = None   # set in tests to use a non-tiktoken tokenizer


def load(hf_variant=None):
    global model, config
    if hf_variant:
        from inference.load_weights import load_from_hf
        model, config = load_from_hf(hf_variant)
        model = model.to(device)
    else:
        model = load_model(config, CKPT_DIR, device)
    print(f"Model ready on {device}")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="SAIR miniGPT")

# serve static files (logo, etc.) and the index.html
UI_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(UI_DIR / "index.html")


# ── Generate endpoint ─────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt:      str
    max_tokens:  int   = GEN_MAX_TOKENS
    temperature: float = GEN_TEMPERATURE
    top_k:       int   = GEN_TOP_K
    top_p:       float = GEN_TOP_P
    beams:       int   = GEN_BEAMS
    method:      str   = "nucleus"


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    if model is None:
        return {"error": "Model not loaded"}
    tok = _tokenizer_override if _tokenizer_override is not None else tokenizer
    text = generate(
        model          = model,
        prompt         = req.prompt,
        max_new_tokens = req.max_tokens,
        context_size   = config["context_length"],
        tokenizer      = tok,
        device         = device,
        temperature    = req.temperature,
        top_k          = req.top_k,
        top_p          = req.top_p,
        beams          = req.beams,
        method         = req.method,
    )
    return {"text": text}


@app.get("/info")
def info():
    return {
        "device" : device,
        "params" : sum(p.numel() for p in model.parameters()) if model else 0,
        "config" : config,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run(hf_variant=None, host="0.0.0.0", port=7860):
    load(hf_variant)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()

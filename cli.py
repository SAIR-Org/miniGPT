"""
cli.py — single entry point for the entire pipeline.

Commands:
    sair prepare                          tokenize data/raw/ → data/processed/
    sair train                            train locally
    sair train --modal                    train on Modal cloud (A100)
    sair generate "Once upon a time"      generate text
    sair generate "..." --method nucleus  choose sampling strategy
    sair ui                               launch web UI (http://localhost:7860)
"""
import argparse
import sys
from pathlib import Path

# make all submodules importable from repo root
sys.path.insert(0, str(Path(__file__).parent))


def cmd_prepare(args):
    from data.prepare import prepare
    prepare()


def cmd_train(args):
    import subprocess

    if args.modal:
        print("Launching training on Modal (A100)...")
        subprocess.run(["python", "-m", "modal", "run", "train/modal_train.py::main"], check=True)
        return

    if args.ddp:
        import torch
        nproc = args.nproc or torch.cuda.device_count()
        if nproc < 2:
            print("Warning: --ddp requested but only 1 GPU found. Falling back to single-GPU.")
        else:
            print(f"Launching DDP training on {nproc} GPUs via torchrun...")
            subprocess.run([
                "torchrun", "--standalone",
                f"--nproc_per_node={nproc}",
                "train/ddp_trainer.py",
            ], check=True)
            return

    import torch
    from config import MODEL_CONFIG
    from data.dataset  import get_loaders
    from model.gpt     import GPTModel
    from train.trainer import train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training locally on: {device}")
    model = GPTModel(MODEL_CONFIG).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_loader, val_loader, _ = get_loaders()
    train(model, train_loader, val_loader, device)


def cmd_generate(args):
    import torch
    import tiktoken
    from config import MODEL_CONFIG, CKPT_DIR, GEN_MAX_TOKENS, GEN_TEMPERATURE, GEN_TOP_K, GEN_TOP_P, GEN_BEAMS
    from inference.generate      import generate, load_model
    from inference.load_weights  import load_from_hf

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.encoding_for_model("gpt2")

    if args.hf:
        model, config = load_from_hf(args.hf)
        model = model.to(device)
    else:
        model  = load_model(MODEL_CONFIG, CKPT_DIR, device)
        config = MODEL_CONFIG

    out = generate(
        model          = model,
        prompt         = args.prompt,
        max_new_tokens = args.max_tokens,
        context_size   = config["context_length"],
        tokenizer      = tokenizer,
        device         = device,
        temperature    = args.temperature,
        top_k          = args.top_k,
        top_p          = args.top_p,
        beams          = args.beams,
        method         = args.method,
    )
    print(out)


def cmd_ui(args):
    from ui.server import run
    run(hf_variant=args.hf)


# ── Argument parser ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="sair",
        description="SAIR miniGPT — data → train → generate → UI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare
    sub.add_parser("prepare", help="Tokenize data/raw/ → data/processed/")

    # train
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--modal", action="store_true",
                         help="Run on Modal cloud GPU instead of locally")
    p_train.add_argument("--ddp", action="store_true",
                         help="Multi-GPU training with DistributedDataParallel via torchrun")
    p_train.add_argument("--nproc", type=int, default=None,
                         help="Number of GPUs for DDP (default: all available)")

    # generate
    p_gen = sub.add_parser("generate", help="Generate text from a prompt")
    p_gen.add_argument("prompt", type=str)
    p_gen.add_argument("--max-tokens",  type=int,   default=100)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top-k",       type=int,   default=50)
    p_gen.add_argument("--top-p",       type=float, default=0.9)
    p_gen.add_argument("--beams",       type=int,   default=1)
    p_gen.add_argument("--method",      type=str,   default="nucleus",
                       choices=["greedy", "top_k", "nucleus"])
    p_gen.add_argument("--hf",          type=str,   default=None,
                       metavar="VARIANT",
                       help="Load pretrained weights from HuggingFace instead of checkpoint. "
                            "Choices: gpt2 | gpt2-medium | gpt2-large | gpt2-xl")

    # ui
    p_ui = sub.add_parser("ui", help="Launch web UI (http://localhost:7860)")
    p_ui.add_argument("--hf", type=str, default=None,
                      metavar="VARIANT",
                      help="Load pretrained weights from HuggingFace instead of checkpoint. "
                           "Choices: gpt2 | gpt2-medium | gpt2-large | gpt2-xl")

    args = parser.parse_args()
    # normalise hyphen → underscore for dest
    if hasattr(args, "max_tokens") and args.max_tokens is None:
        args.max_tokens = 100

    dispatch = {
        "prepare" : cmd_prepare,
        "train"   : cmd_train,
        "generate": cmd_generate,
        "ui"      : cmd_ui,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

from pathlib import Path
import torch
from config import HF_MODELS, MODEL_CONFIG, VARIANT, data_dir_classification
from data.dataset import get_classification_dataloaders
from model.gpt import GPTModel
from finetune.instructure_follower_finetuning import loading_model
from train.trainer import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_classification_model(variant: str, num_classes: int = 2) -> tuple[torch.nn.Module, dict]:
    """Load a pretrained GPT and adapt it for sequence classification."""
    model, config = loading_model(variant)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the output head with a classification head
    torch.manual_seed(123)
    # The original out_head was Linear(emb_dim, vocab_size, bias=False)
    # We replace it with Linear(emb_dim, num_classes)
    emb_dim = model.out_head.in_features
    model.out_head = torch.nn.Linear(emb_dim, num_classes)

    # Unfreeze the last transformer block and the final layer norm
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    return model, config

if __name__ == "__main__":
    # Local test run
    print(f"Loading data from: {data_dir_classification}")
    train_loader, val_loader, test_loader = get_classification_dataloaders(
        csv_path=data_dir_classification,
        output_dir="data/classification-processed"
    )
    
    print("Setting up model...")
    model, config = setup_classification_model(variant=VARIANT)
    model.to(device)
    
    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Training locally on: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, train_loader, val_loader, device)
    print("model trained")

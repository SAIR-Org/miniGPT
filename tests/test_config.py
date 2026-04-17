"""test_config.py — config loads correctly and all presets are valid."""
from config import MODELS, MODEL_CONFIG, HF_MODELS, DATA_RAW, DATA_DIR, CKPT_DIR


def test_all_presets_have_required_keys():
    required = {"vocab_size", "context_length", "emb_dim",
                "n_heads", "n_layers", "drop_rate", "qkv_bias"}
    for name, cfg in MODELS.items():
        assert required == set(cfg.keys()), f"Preset '{name}' missing keys"


def test_active_config_is_valid():
    assert MODEL_CONFIG["n_heads"] > 0
    assert MODEL_CONFIG["emb_dim"] % MODEL_CONFIG["n_heads"] == 0, \
        "emb_dim must be divisible by n_heads"


def test_hf_models_have_required_keys():
    for name, entry in HF_MODELS.items():
        assert "repo"   in entry, f"{name} missing 'repo'"
        assert "config" in entry, f"{name} missing 'config'"
        assert "qkv_bias" in entry["config"], f"{name} config missing 'qkv_bias'"


def test_paths_are_path_objects():
    from pathlib import Path
    assert isinstance(DATA_RAW, Path)
    assert isinstance(DATA_DIR,  Path)
    assert isinstance(CKPT_DIR,  Path)

"""test_server.py — FastAPI endpoints return correct responses."""
import torch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ui.server import app, load
from tests.conftest import TINY_CONFIG, DEVICE, TinyTokenizer


@pytest.fixture
def client(tmp_path_factory, tiny_model, tiny_tokenizer):
    """Load the tiny model into the server then return a test client."""
    import ui.server as srv
    srv.model               = tiny_model
    srv.config              = TINY_CONFIG
    srv.device              = DEVICE
    srv._tokenizer_override = tiny_tokenizer
    return TestClient(app)


def test_root_returns_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_info_endpoint(client):
    res  = client.get("/info")
    data = res.json()
    assert res.status_code == 200
    assert "device" in data
    assert "params" in data
    assert data["params"] > 0


def test_generate_endpoint_returns_text(client):
    res  = client.post("/generate", json={
        "prompt"     : "Hello",
        "max_tokens" : 5,
        "temperature": 0.0,
        "method"     : "greedy",
        "beams"      : 1,
    })
    data = res.json()
    assert res.status_code == 200
    assert "text" in data
    assert isinstance(data["text"], str)
    assert data["text"].startswith("Hello")


def test_generate_endpoint_nucleus(client):
    res = client.post("/generate", json={
        "prompt"     : "Once",
        "max_tokens" : 5,
        "temperature": 0.8,
        "top_k"      : 10,
        "top_p"      : 0.9,
        "method"     : "nucleus",
        "beams"      : 1,
    })
    assert res.status_code == 200
    assert "text" in res.json()


def test_generate_endpoint_beam_search(client):
    res = client.post("/generate", json={
        "prompt"     : "The",
        "max_tokens" : 5,
        "temperature": 0.7,
        "method"     : "greedy",
        "beams"      : 3,
    })
    assert res.status_code == 200
    assert "text" in res.json()

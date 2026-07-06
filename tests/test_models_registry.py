from app.models_registry import load_models, resolve_ollama
import pytest


def test_load_models_has_base():
    ms = load_models()
    assert any(m["id"] == "base-qwen3-4b" for m in ms)
    base = next(m for m in ms if m["id"] == "base-qwen3-4b")
    assert base["ollama"] == "qwen3:4b" and base["kind"] == "base"


def test_resolve_ollama():
    assert resolve_ollama("base-qwen3-4b") == "qwen3:4b"


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve_ollama("nope")

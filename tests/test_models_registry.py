from app.models_registry import load_models, resolve_ollama
import pytest


def test_load_models_has_base():
    ms = load_models()
    assert any(m["id"] == "base-llama32-3b-instruct" for m in ms)
    base = next(m for m in ms if m["id"] == "base-llama32-3b-instruct")
    # Don't hard-code the Ollama tag (config-driven); just require it's set + base kind.
    assert base["ollama"] and base["kind"] == "base"


def test_resolve_ollama():
    ms = load_models()
    base = next(m for m in ms if m["id"] == "base-llama32-3b-instruct")
    assert resolve_ollama("base-llama32-3b-instruct") == base["ollama"]


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve_ollama("nope")

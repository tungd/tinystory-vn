from app.models_registry import load_models, resolve_ollama
import pytest


def test_load_models_has_base():
    ms = load_models()
    assert any(m["id"] == "base-qwen3-4b" for m in ms)
    base = next(m for m in ms if m["id"] == "base-qwen3-4b")
    # Don't hard-code the Ollama tag (config-driven); just require it's set + base kind.
    assert base["ollama"] and base["kind"] == "base"


def test_registry_points_to_current_fable_model():
    model = next(m for m in load_models() if m["id"] == "fable-200m")
    assert model["name"] == "Fable-64M (from scratch)"
    assert model["ollama"] == "fable-64m-mlx"
    assert model["kind"] == "finetuned"


def test_resolve_ollama():
    ms = load_models()
    base = next(m for m in ms if m["id"] == "base-qwen3-4b")
    assert resolve_ollama("base-qwen3-4b") == base["ollama"]


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve_ollama("nope")

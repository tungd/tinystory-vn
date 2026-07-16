import json
from pathlib import Path

import httpx

from app.config import BACKEND, MODELS_PATH, OLLAMA_BASE_URL, REQUEST_TIMEOUT_SECONDS

# Cache for the OpenAI-compatible server's model ID (avoids querying every request).
_openai_model_cache: dict[str, str] = {}


def load_models() -> list[dict]:
    return json.loads(Path(MODELS_PATH).read_text(encoding="utf-8"))


def _resolve_openai_model(hint: str) -> str:
    """Query the OpenAI-compatible server for its loaded model ID.

    The MLX server identifies models by their absolute path, which is fragile
    to hardcode. Instead, query /v1/models and return the first (or matching)
    model ID. Falls back to the hint if the server is unreachable.
    """
    if hint in _openai_model_cache:
        return _openai_model_cache[hint]

    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=5) as client:
            resp = client.get("/v1/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                model_id = models[0]["id"]
                _openai_model_cache[hint] = model_id
                return model_id
    except Exception:
        pass

    # Fallback to the configured hint
    return hint


def resolve_ollama(model_id: str) -> str:
    for m in load_models():
        if m["id"] == model_id:
            # Auto-detect model ID only for local MLX-served models (finetuned).
            # Judge/base models use external APIs and keep their configured name.
            if BACKEND == "openai" and m.get("kind") == "finetuned":
                return _resolve_openai_model(m["ollama"])
            return m["ollama"]
    raise KeyError(f"Unknown model_id: {model_id}")

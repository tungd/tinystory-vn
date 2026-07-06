import json
from pathlib import Path
from app.config import MODELS_PATH


def load_models() -> list[dict]:
    return json.loads(Path(MODELS_PATH).read_text(encoding="utf-8"))


def resolve_ollama(model_id: str) -> str:
    for m in load_models():
        if m["id"] == model_id:
            return m["ollama"]
    raise KeyError(f"Unknown model_id: {model_id}")

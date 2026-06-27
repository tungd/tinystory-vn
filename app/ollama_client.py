import json
from collections.abc import Iterator

import httpx

from app.config import ENABLE_THINKING, MODEL_NAME, OLLAMA_BASE_URL, REQUEST_TIMEOUT_SECONDS


class OllamaError(Exception):
    pass


def generate(prompt: str, system: str, model: str | None = None, num_predict: int | None = None) -> str:
    payload = {
        "model": model or MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": ENABLE_THINKING,
    }
    if num_predict is not None:
        payload["options"] = {"num_predict": num_predict}
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = client.post("/api/chat", json=payload)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Lỗi gọi Ollama: {exc}") from exc

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    if not content:
        raise OllamaError("Ollama trả về nội dung rỗng.")
    return content


def generate_stream(prompt: str, system: str, model: str | None = None,
                    num_predict: int | None = None) -> Iterator[str]:
    payload = {
        "model": model or MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "think": ENABLE_THINKING,
    }
    if num_predict is not None:
        payload["options"] = {"num_predict": num_predict}
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            with client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
    except httpx.HTTPError as exc:
        raise OllamaError(f"Lỗi gọi Ollama (stream): {exc}") from exc

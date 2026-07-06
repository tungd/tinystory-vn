import json
import time
from collections.abc import Iterator

import httpx

from app.config import ENABLE_THINKING, MODEL_NAME, OLLAMA_BASE_URL, REQUEST_TIMEOUT_SECONDS


class OllamaError(Exception):
    pass


def generate(
    prompt: str,
    system: str,
    model: str | None = None,
    num_predict: int | None = None,
    seed: int | None = None,
    **kwargs,
) -> str:
    options: dict = {}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if seed is not None:
        options["seed"] = seed
    payload: dict = {
        "model": model or MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": ENABLE_THINKING,
    }
    if options:
        payload["options"] = options
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


def generate_meta(
    prompt: str,
    system: str,
    model: str | None = None,
    num_predict: int | None = None,
    seed: int | None = None,
    **kwargs,
) -> dict:
    """Non-streaming generate that also returns token counts and latency."""
    options: dict = {}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if seed is not None:
        options["seed"] = seed
    payload: dict = {
        "model": model or MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": ENABLE_THINKING,
    }
    if options:
        payload["options"] = options

    t0 = time.perf_counter()
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = client.post("/api/chat", json=payload)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Lỗi gọi Ollama: {exc}") from exc
    latency_ms = int((time.perf_counter() - t0) * 1000)

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    if not content:
        raise OllamaError("Ollama trả về nội dung rỗng.")
    return {
        "text": content,
        "input_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "latency_ms": latency_ms,
    }


def generate_stream(
    prompt: str,
    system: str,
    model: str | None = None,
    num_predict: int | None = None,
    seed: int | None = None,
    **kwargs,
) -> Iterator[str]:
    options: dict = {}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if seed is not None:
        options["seed"] = seed
    payload: dict = {
        "model": model or MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "think": ENABLE_THINKING,
    }
    if options:
        payload["options"] = options
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

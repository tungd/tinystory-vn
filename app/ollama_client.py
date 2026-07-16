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
    for k in ("temperature", "top_p", "repeat_penalty"):
        if k in kwargs and kwargs[k] is not None:
            options[k] = kwargs[k]
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
    for k in ("temperature", "top_p", "repeat_penalty"):
        if k in kwargs and kwargs[k] is not None:
            options[k] = kwargs[k]
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
        # "stop" = model phát <|end|> (Ollama nuốt token này) = kết hoàn thiện;
        # "length" = đụng trần num_predict/context = truyện bị cắt.
        "done_reason": data.get("done_reason"),
    }


def generate_stream(
    prompt: str,
    system: str,
    model: str | None = None,
    num_predict: int | None = None,
    seed: int | None = None,
    on_done=None,
    **kwargs,
) -> Iterator[str]:
    """Stream story tokens. Nếu truyền `on_done`, gọi `on_done(done_reason)` khi
    chunk cuối về ("stop" = kết hoàn thiện, "length" = bị cắt) để caller xử lý."""
    options: dict = {}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if seed is not None:
        options["seed"] = seed
    for k in ("temperature", "top_p", "repeat_penalty"):
        if k in kwargs and kwargs[k] is not None:
            options[k] = kwargs[k]
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
                        if on_done is not None:
                            on_done(chunk.get("done_reason"))
                        break
    except httpx.HTTPError as exc:
        raise OllamaError(f"Lỗi gọi Ollama (stream): {exc}") from exc

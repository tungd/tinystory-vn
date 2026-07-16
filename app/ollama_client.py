"""LLM client — supports Ollama and OpenAI-compatible backends (MLX server, llama.cpp, etc.).

Backend is selected via FABLE_BACKEND env var:
  - "ollama" (default): Ollama's /api/chat endpoint (chat format)
  - "openai": OpenAI-compatible API. Uses /v1/completions for base LMs (raw text
    continuation, supports repetition_penalty) or /v1/chat/completions for
    instruction-tuned models (chat format). Controlled by FABLE_USE_COMPLETION
    (default "true" — use completions endpoint, which works for the from-scratch
    fable model that expects raw prefix continuation, not chat format).

Per-call overrides: pass backend=, base_url=, use_completion=, api_key= to
generate()/generate_meta() to use a different backend for that call only
(e.g. Gemini for judging while MLX serves generation).
"""

import json
import os
import time
from collections.abc import Iterator

import httpx

from app.config import (
    BACKEND,
    ENABLE_THINKING,
    JUDGE_API_KEY,
    JUDGE_BACKEND,
    JUDGE_BASE_URL,
    JUDGE_USE_COMPLETION,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
)

# When using the openai backend, use /v1/completions (raw text) by default.
# Set FABLE_USE_COMPLETION=false to use /v1/chat/completions instead.
_USE_COMPLETION = os.getenv("FABLE_USE_COMPLETION", "true").lower() == "true"


class OllamaError(Exception):
    pass


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _build_messages(prompt: str, system: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def _ollama_payload(model, messages, stream, num_predict, seed, **kwargs):
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
        "messages": messages,
        "stream": stream,
        "think": ENABLE_THINKING,
    }
    if options:
        payload["options"] = options
    return payload


def _openai_completion_payload(model, prompt_text, stream, num_predict, seed, **kwargs):
    payload: dict = {
        "model": model or MODEL_NAME,
        "prompt": prompt_text,
        "stream": stream,
    }
    if num_predict is not None:
        payload["max_tokens"] = num_predict
    if seed is not None:
        payload["seed"] = seed
    if kwargs.get("temperature") is not None:
        payload["temperature"] = kwargs["temperature"]
    if kwargs.get("top_p") is not None:
        payload["top_p"] = kwargs["top_p"]
    if kwargs.get("repeat_penalty") is not None:
        payload["repetition_penalty"] = kwargs["repeat_penalty"]
    return payload


def _openai_chat_payload(model, messages, stream, num_predict, seed, **kwargs):
    payload: dict = {
        "model": model or MODEL_NAME,
        "messages": messages,
        "stream": stream,
    }
    if num_predict is not None:
        payload["max_tokens"] = num_predict
    if seed is not None:
        payload["seed"] = seed
    if kwargs.get("temperature") is not None:
        payload["temperature"] = kwargs["temperature"]
    if kwargs.get("top_p") is not None:
        payload["top_p"] = kwargs["top_p"]
    if kwargs.get("repeat_penalty") is not None:
        payload["frequency_penalty"] = max(0, kwargs["repeat_penalty"] - 1.0)
    return payload


def _build_ollama_payload(model, messages, stream, num_predict, seed, **kwargs):
    return _ollama_payload(model, messages, stream, num_predict, seed, **kwargs)


def _build_openai_payload(model, messages, system, prompt, stream, num_predict, seed, **kwargs):
    if _USE_COMPLETION:
        # Combine system + prompt into a single text for raw completion
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        return _openai_completion_payload(model, full_prompt, stream, num_predict, seed, **kwargs)
    return _openai_chat_payload(model, messages, stream, num_predict, seed, **kwargs)


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_ollama_content(data: dict) -> str:
    return data.get("message", {}).get("content", "")


def _parse_openai_completion_content(data: dict) -> str:
    return data.get("choices", [{}])[0].get("text", "")


def _parse_openai_chat_content(data: dict) -> str:
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def _parse_content(data: dict) -> str:
    if BACKEND == "openai":
        if _USE_COMPLETION:
            return _parse_openai_completion_content(data)
        return _parse_openai_chat_content(data)
    return _parse_ollama_content(data)


def _parse_ollama_meta(data: dict) -> dict:
    return {
        "input_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
    }


def _parse_openai_meta(data: dict) -> dict:
    usage = data.get("usage", {})
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }


def _parse_meta(data: dict) -> dict:
    if BACKEND == "openai":
        return _parse_openai_meta(data)
    return _parse_ollama_meta(data)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

def _chat_endpoint() -> str:
    if BACKEND == "openai":
        if _USE_COMPLETION:
            return "/v1/completions"
        return "/v1/chat/completions"
    return "/api/chat"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    system: str,
    model: str | None = None,
    num_predict: int | None = None,
    seed: int | None = None,
    **kwargs,
) -> str:
    messages = _build_messages(prompt, system)
    if BACKEND == "openai":
        payload = _build_openai_payload(model, messages, system, prompt, False, num_predict, seed, **kwargs)
    else:
        payload = _build_ollama_payload(model, messages, False, num_predict, seed, **kwargs)
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = client.post(_chat_endpoint(), json=payload)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"LLM call failed: {exc}") from exc

    data = resp.json()
    content = _parse_content(data)
    if not content:
        raise OllamaError("LLM returned empty content.")
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
    messages = _build_messages(prompt, system)
    if BACKEND == "openai":
        payload = _build_openai_payload(model, messages, system, prompt, False, num_predict, seed, **kwargs)
    else:
        payload = _build_ollama_payload(model, messages, False, num_predict, seed, **kwargs)

    t0 = time.perf_counter()
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = client.post(_chat_endpoint(), json=payload)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"LLM call failed: {exc}") from exc
    latency_ms = int((time.perf_counter() - t0) * 1000)

    data = resp.json()
    content = _parse_content(data)
    if not content:
        raise OllamaError("LLM returned empty content.")
    meta = _parse_meta(data)
    meta["text"] = content
    meta["latency_ms"] = latency_ms
    return meta


def generate_stream(
    prompt: str,
    system: str,
    model: str | None = None,
    num_predict: int | None = None,
    seed: int | None = None,
    **kwargs,
) -> Iterator[str]:
    messages = _build_messages(prompt, system)
    if BACKEND == "openai":
        payload = _build_openai_payload(model, messages, system, prompt, True, num_predict, seed, **kwargs)
    else:
        payload = _build_ollama_payload(model, messages, True, num_predict, seed, **kwargs)
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            with client.stream("POST", _chat_endpoint(), json=payload) as resp:
                resp.raise_for_status()
                if BACKEND == "openai" and _USE_COMPLETION:
                    # OpenAI SSE for completions: data: {"choices":[{"text":"..."}]}
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw.strip() == "[DONE]":
                            break
                        chunk = json.loads(raw)
                        piece = (
                            chunk.get("choices", [{}])[0].get("text", "")
                        )
                        if piece:
                            yield piece
                elif BACKEND == "openai":
                    # OpenAI SSE for chat: data: {"choices":[{"delta":{"content":"..."}}]}
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw.strip() == "[DONE]":
                            break
                        chunk = json.loads(raw)
                        piece = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if piece:
                            yield piece
                else:
                    # Ollama NDJSON: one JSON object per line
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
        raise OllamaError(f"LLM stream failed: {exc}") from exc

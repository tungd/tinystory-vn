"""LLM client — supports Ollama and OpenAI-compatible backends (MLX server, llama.cpp, etc.).

Backend is selected via FABLE_BACKEND env var:
  - "ollama" (default): Ollama's /api/chat endpoint (chat format)
  - "openai": OpenAI-compatible API. Uses /v1/completions for base LMs (raw text
    continuation, supports repetition_penalty) or /v1/chat/completions for
    instruction-tuned models (chat format). Controlled by FABLE_USE_COMPLETION
    (default "true").

Per-call overrides: pass backend=, base_url=, use_completion=, api_key= to
generate()/generate_meta() to use a different backend for that call only
(e.g. Gemma via Google AI Studio for judging while MLX serves generation).
"""

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

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


@dataclass
class _CallConfig:
    """Resolved backend config for a single call."""
    backend: str
    base_url: str
    use_completion: bool
    api_key: str = ""
    # Path prefix for OpenAI-compatible endpoints. Default "/v1" (MLX, Ollama,
    # most servers). Set to "" for Google AI Studio (base_url already includes
    # /v1beta/openai, endpoints are /chat/completions, /completions).
    path_prefix: str = "/v1"


def _resolve_config(
    backend: str | None = None,
    base_url: str | None = None,
    use_completion: bool | None = None,
    api_key: str | None = None,
    is_judge: bool = False,
) -> _CallConfig:
    """Resolve call config, falling back to globals then judge config."""
    b = backend or BACKEND
    bu = base_url or OLLAMA_BASE_URL
    uc = use_completion if use_completion is not None else _USE_COMPLETION
    ak = api_key or ""

    # If a judge backend is configured and the caller explicitly requests it
    # (is_judge=True), use the judge config. This prevents generation calls
    # from accidentally hitting the judge endpoint.
    if JUDGE_BACKEND and is_judge:
        b = JUDGE_BACKEND
        bu = JUDGE_BASE_URL or OLLAMA_BASE_URL
        uc = JUDGE_USE_COMPLETION
        ak = JUDGE_API_KEY
        # Google AI Studio uses /v1beta/openai as base, endpoints without /v1
        pp = "" if "googleapis.com" in bu else "/v1"
        return _CallConfig(backend=b, base_url=bu, use_completion=uc, api_key=ak, path_prefix=pp)

    return _CallConfig(backend=b, base_url=bu, use_completion=uc, api_key=ak)


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


def _build_payload(cfg: _CallConfig, model, messages, system, prompt, stream, num_predict, seed, **kwargs):
    if cfg.backend == "openai":
        if cfg.use_completion:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            return _openai_completion_payload(model, full_prompt, stream, num_predict, seed, **kwargs)
        return _openai_chat_payload(model, messages, stream, num_predict, seed, **kwargs)
    return _ollama_payload(model, messages, stream, num_predict, seed, **kwargs)


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_content(cfg: _CallConfig, data: dict) -> str:
    if cfg.backend == "openai":
        if cfg.use_completion:
            return data.get("choices", [{}])[0].get("text", "")
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return data.get("message", {}).get("content", "")


def _parse_meta(cfg: _CallConfig, data: dict) -> dict:
    if cfg.backend == "openai":
        usage = data.get("usage", {})
        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
    return {
        "input_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
    }


def _endpoint(cfg: _CallConfig) -> str:
    if cfg.backend == "openai":
        if cfg.use_completion:
            return f"{cfg.path_prefix}/completions"
        return f"{cfg.path_prefix}/chat/completions"
    return "/api/chat"


def _headers(cfg: _CallConfig) -> dict:
    h = {}
    if cfg.api_key:
        h["Authorization"] = f"Bearer {cfg.api_key}"
    return h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    system: str,
    model: str | None = None,
    num_predict: int | None = None,
    seed: int | None = None,
    *,
    backend: str | None = None,
    base_url: str | None = None,
    use_completion: bool | None = None,
    api_key: str | None = None,
    is_judge: bool = False,
    **kwargs,
) -> str:
    cfg = _resolve_config(backend, base_url, use_completion, api_key, is_judge=is_judge)
    messages = _build_messages(prompt, system)
    payload = _build_payload(cfg, model, messages, system, prompt, False, num_predict, seed, **kwargs)
    try:
        with httpx.Client(base_url=cfg.base_url, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = client.post(_endpoint(cfg), json=payload, headers=_headers(cfg))
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"LLM call failed: {exc}") from exc

    data = resp.json()
    content = _parse_content(cfg, data)
    if not content:
        raise OllamaError("LLM returned empty content.")
    return content


def generate_meta(
    prompt: str,
    system: str,
    model: str | None = None,
    num_predict: int | None = None,
    seed: int | None = None,
    *,
    backend: str | None = None,
    base_url: str | None = None,
    use_completion: bool | None = None,
    api_key: str | None = None,
    is_judge: bool = False,
    **kwargs,
) -> dict:
    """Non-streaming generate that also returns token counts and latency."""
    cfg = _resolve_config(backend, base_url, use_completion, api_key, is_judge=is_judge)
    messages = _build_messages(prompt, system)
    payload = _build_payload(cfg, model, messages, system, prompt, False, num_predict, seed, **kwargs)

    t0 = time.perf_counter()
    try:
        with httpx.Client(base_url=cfg.base_url, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = client.post(_endpoint(cfg), json=payload, headers=_headers(cfg))
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"LLM call failed: {exc}") from exc
    latency_ms = int((time.perf_counter() - t0) * 1000)

    data = resp.json()
    content = _parse_content(cfg, data)
    if not content:
        raise OllamaError("LLM returned empty content.")
    meta = _parse_meta(cfg, data)
    meta["text"] = content
    meta["latency_ms"] = latency_ms
    return meta


def generate_stream(
    prompt: str,
    system: str,
    model: str | None = None,
    num_predict: int | None = None,
    seed: int | None = None,
    *,
    backend: str | None = None,
    base_url: str | None = None,
    use_completion: bool | None = None,
    api_key: str | None = None,
    **kwargs,
) -> Iterator[str]:
    cfg = _resolve_config(backend, base_url, use_completion, api_key)
    messages = _build_messages(prompt, system)
    payload = _build_payload(cfg, model, messages, system, prompt, True, num_predict, seed, **kwargs)
    try:
        with httpx.Client(base_url=cfg.base_url, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            with client.stream("POST", _endpoint(cfg), json=payload, headers=_headers(cfg)) as resp:
                resp.raise_for_status()
                if cfg.backend == "openai" and cfg.use_completion:
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw.strip() == "[DONE]":
                            break
                        chunk = json.loads(raw)
                        piece = chunk.get("choices", [{}])[0].get("text", "")
                        if piece:
                            yield piece
                elif cfg.backend == "openai":
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

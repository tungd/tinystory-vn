import httpx

from app.config import MODEL_NAME, OLLAMA_BASE_URL, REQUEST_TIMEOUT_SECONDS


class OllamaError(Exception):
    pass


def generate(prompt: str, system: str, model: str | None = None) -> str:
    payload = {
        "model": model or MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
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

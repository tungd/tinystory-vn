"""Native Google GenAI client for the Gemma 4 judge."""

from functools import lru_cache

from google import genai
from google.genai import types

from app.config import JUDGE_API_KEY, JUDGE_THINKING_LEVEL, REQUEST_TIMEOUT_SECONDS


class GoogleJudgeError(Exception):
    pass


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    if not JUDGE_API_KEY:
        raise GoogleJudgeError("FABLE_JUDGE_API_KEY is required for Google judging.")
    return genai.Client(
        api_key=JUDGE_API_KEY,
        http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT_SECONDS * 1000)),
    )


def _thinking_level() -> types.ThinkingLevel:
    levels = {
        "minimal": types.ThinkingLevel.MINIMAL,
        "high": types.ThinkingLevel.HIGH,
    }
    try:
        return levels[JUDGE_THINKING_LEVEL]
    except KeyError as exc:
        raise GoogleJudgeError(
            "Gemma 4 judge thinking must be 'minimal' or 'high'."
        ) from exc


def _answer_text(response) -> str:
    candidates = response.candidates or []
    if not candidates or not candidates[0].content:
        raise GoogleJudgeError("Google judge returned no candidate.")
    parts = candidates[0].content.parts or []
    answer = "".join(
        part.text or "" for part in parts if not getattr(part, "thought", False)
    ).strip()
    if not answer:
        raise GoogleJudgeError("Google judge returned no final answer.")
    return answer


def generate(
    prompt: str,
    system: str,
    model: str | None = None,
    num_predict: int | None = None,
    temperature: float | None = None,
    response_schema: dict | None = None,
    **_,
) -> str:
    """Generate one structured judgment through Google's native API."""
    response = _client().models.generate_content(
        model=model or "gemma-4-26b-a4b-it",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=num_predict,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(
                thinking_level=_thinking_level(),
                include_thoughts=False,
            ),
            response_mime_type="application/json",
            response_json_schema=response_schema,
        ),
    )
    return _answer_text(response)

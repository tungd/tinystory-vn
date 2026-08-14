"""Google GenAI judge for Demo Studio; deliberately independent of Ollama."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from app.judge import (
    build_google_judge_prompt,
    build_google_judge_schema,
    scores_from_google_response,
    validate_google_scores,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(ROOT / ".env")


class GoogleJudge:
    def __init__(self) -> None:
        self.backend = os.getenv("FABLE_JUDGE_BACKEND", "google")
        self.model_id = os.getenv("FABLE_JUDGE_MODEL_ID", "gemma-4-26b-a4b-it")
        self.base_url = os.getenv(
            "FABLE_JUDGE_BASE_URL", "https://generativelanguage.googleapis.com"
        )
        self.api_key_configured = bool(os.getenv("FABLE_JUDGE_API_KEY"))
        self._client: Any = None

    def status(self) -> dict[str, Any]:
        sdk_available = False
        try:
            from google import genai  # noqa: F401

            sdk_available = True
        except ImportError:
            pass
        configured = self.backend == "google" and self.api_key_configured and sdk_available
        return {
            "backend": self.backend,
            "provider": "Google GenAI",
            "model_id": self.model_id,
            "base_url": self.base_url,
            "api_key_configured": self.api_key_configured,
            "sdk_available": sdk_available,
            "configured": configured,
            "detail": (
                "Google GenAI judge ready"
                if configured
                else "Set FABLE_JUDGE_API_KEY and install google-genai to enable the judge"
            ),
        }

    def _get_client(self) -> Any:
        if self.backend != "google":
            raise RuntimeError(
                f"Unsupported judge backend '{self.backend}'. Demo Studio expects Google GenAI."
            )
        if not os.getenv("FABLE_JUDGE_API_KEY"):
            raise RuntimeError("FABLE_JUDGE_API_KEY is not configured")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is not installed in the active Python environment") from exc
        if self._client is None:
            self._client = genai.Client(api_key=os.environ["FABLE_JUDGE_API_KEY"])
        return self._client

    @staticmethod
    def _schema() -> dict[str, Any]:
        return build_google_judge_schema()

    @staticmethod
    def _prompt(story: str, prompt: str) -> str:
        return build_google_judge_prompt(story, prompt)

    @staticmethod
    def _scores_from_response(response: Any) -> tuple[dict[str, Any], str]:
        return scores_from_google_response(response)

    def evaluate(self, story: str, prompt: str) -> dict[str, Any]:
        from google.genai import types

        client = self._get_client()
        thinking_level = os.getenv("FABLE_JUDGE_THINKING_LEVEL", "minimal").upper()
        thinking = getattr(types.ThinkingLevel, thinking_level, types.ThinkingLevel.MINIMAL)
        config = types.GenerateContentConfig(
            system_instruction="You are a strict, fair evaluator. Output JSON only.",
            temperature=0,
            seed=20260726,
            max_output_tokens=512,
            response_mime_type="application/json",
            response_json_schema=self._schema(),
            thinking_config=types.ThinkingConfig(
                thinking_level=thinking,
                include_thoughts=False,
            ),
        )
        last_error: Exception | None = None
        last_raw = ""
        for attempt in range(1, 4):
            started = time.perf_counter()
            try:
                response = client.models.generate_content(
                    model=self.model_id,
                    contents=self._prompt(story, prompt),
                    config=config,
                )
                scores, raw = self._scores_from_response(response)
                last_raw = raw
                validate_google_scores(scores)
                return {
                    **scores,
                    "judge_model": self.model_id,
                    "judge_backend": "google-genai",
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "raw": raw,
                    "attempt": attempt,
                }
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(0.5 * attempt)
        raise RuntimeError(
            f"Google judge returned invalid scores after 3 attempts: {last_error}; "
            f"raw={last_raw[:600]!r}"
        ) from last_error


judge = GoogleJudge()

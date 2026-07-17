from types import SimpleNamespace

import pytest
from google.genai import types

from app import google_judge_client as client


def test_generate_uses_native_minimal_thinking_and_json_mode(monkeypatch):
    seen = {}
    final = '{"grammar":{"score":9,"reason":"clean"}}'

    class Models:
        def generate_content(self, **kwargs):
            seen.update(kwargs)
            part = SimpleNamespace(text=final, thought=False)
            content = SimpleNamespace(parts=[part])
            return SimpleNamespace(candidates=[SimpleNamespace(content=content)])

    monkeypatch.setattr(client, "_client", lambda: SimpleNamespace(models=Models()))
    monkeypatch.setattr(client, "JUDGE_THINKING_LEVEL", "minimal")

    output = client.generate(
        prompt="judge this",
        system="JSON only",
        model="gemma-4-26b-a4b-it",
        num_predict=500,
        temperature=0.1,
        response_schema={"type": "object"},
    )

    config = seen["config"]
    assert output == final
    assert seen["model"] == "gemma-4-26b-a4b-it"
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
    assert config.thinking_config.include_thoughts is False
    assert config.response_mime_type == "application/json"
    assert config.response_schema is None
    assert config.response_json_schema == {"type": "object"}


def test_answer_text_ignores_thought_parts():
    content = SimpleNamespace(
        parts=[
            SimpleNamespace(text="draft", thought=True),
            SimpleNamespace(text='{"grammar": 9}', thought=False),
        ]
    )
    response = SimpleNamespace(candidates=[SimpleNamespace(content=content)])
    assert client._answer_text(response) == '{"grammar": 9}'


def test_rejects_unsupported_gemma_thinking_level(monkeypatch):
    monkeypatch.setattr(client, "JUDGE_THINKING_LEVEL", "low")
    with pytest.raises(client.GoogleJudgeError, match="minimal.*high"):
        client._thinking_level()

from scripts.judge_v5_pairwise import (
    PAIRWISE_RESPONSE_SCHEMA,
    blind_order,
    build_prompt,
    parse_pairwise,
    summarize,
)


def test_blind_order_is_reproducible_and_uses_both_models():
    assert blind_order("source", 42) == blind_order("source", 42)
    assert set(blind_order("source", 42)) == {"v3-full", "v5"}


def test_pairwise_prompt_is_blind_and_requires_causal_moral():
    prompt = build_prompt("request", "old", "new").lower()
    assert "v3" not in prompt and "v5" not in prompt
    assert "causally" in prompt
    assert PAIRWISE_RESPONSE_SCHEMA["required"] == [
        "a", "b", "winner", "confidence", "reason"
    ]
    assert "mixed" in PAIRWISE_RESPONSE_SCHEMA["properties"]["reason"]["enum"]


def test_parse_and_summarize_pairwise_result():
    raw = """{
      "a":{"grammar":4,"creativity":3,"moral_clarity":2,"prompt_adherence":5},
      "b":{"grammar":7,"creativity":6,"moral_clarity":7,"prompt_adherence":8},
      "winner":"b","confidence":4,"reason":"B has a causal resolution."
    }"""
    parsed = parse_pairwise(raw)
    assert parsed["scores"]["b"]["overall"] == 7.0
    summary = summarize([{
        "winner_model": "v5",
        "scores": {"v3-full": parsed["scores"]["a"], "v5": parsed["scores"]["b"]},
    }])
    assert summary["wins"] == {"v5": 1}
    assert summary["mean_scores"]["v5"]["overall"] == 7.0

"""Tests for pure aggregation functions in scripts/eval_slm.py (ADR-0002 TDD scope)."""
from trieulh.scripts.eval_slm import aggregate_axis_scores, conclude_by_rank


def test_aggregate_axis_scores_means_over_judges():
    panel = {
        "j1": [{"grammar": 8, "creativity": 6, "moral_clarity": 10, "prompt_adherence": 9, "overall": 8.25}],
        "j2": [{"grammar": 10, "creativity": 8, "moral_clarity": 10, "prompt_adherence": 9, "overall": 9.25}],
    }
    agg = aggregate_axis_scores(panel)
    assert agg["grammar"] == 9.0 and agg["overall"] == 8.75


def test_conclude_by_rank_picks_highest():
    c = conclude_by_rank({"slm-30m": 8.9, "qwen3-4b": 9.0, "slm-10m": 8.1})
    assert c["winner"] == "qwen3-4b"
    assert "rank" in c["by_rank"].lower() or "qwen3-4b" in c["by_rank"]

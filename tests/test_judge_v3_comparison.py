from scripts.judge_v3_comparison import select_sources, summarize


def test_select_sources_is_paired_and_reproducible():
    rows = [
        {"source": source, "model": model}
        for model in ("v2", "v3-full")
        for source in ("a", "b", "c")
    ]
    assert select_sources(rows, 2, 42) == select_sources(rows, 2, 42)
    assert len(select_sources(rows, 2, 42)) == 2


def test_summarize_averages_scores():
    rows = []
    for model, overall in (("v2", 4.0), ("v3-full", 8.0)):
        rows.append(
            {
                "model": model,
                "judge": {
                    "grammar": overall,
                    "creativity": overall,
                    "moral_clarity": overall,
                    "prompt_adherence": overall,
                    "overall": overall,
                },
                "judge_latency_ms": 5000,
                "checks": {"ended": model == "v3-full"},
            }
        )
    summary = summarize(rows)
    assert summary["v2"]["judge_mean"]["overall"] == 4.0
    assert summary["v3-full"]["checks_passed"]["ended"] == 1

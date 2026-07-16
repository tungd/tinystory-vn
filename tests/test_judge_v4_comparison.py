from scripts.judge_v4_comparison import select_sources


def test_v4_judge_selects_paired_v3_sources_reproducibly():
    rows = [
        {"source": source, "model": model}
        for model in ("v3-full", "v4")
        for source in ("a", "b", "c")
    ]
    assert select_sources(rows, 2, 42) == select_sources(rows, 2, 42)
    assert len(select_sources(rows, 2, 42)) == 2

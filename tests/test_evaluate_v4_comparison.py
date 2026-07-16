from scripts.evaluate_v4_comparison import paired_changes


def test_v4_paired_changes_compare_v3_full_to_v4():
    rows = [
        {"source": "one", "model": "v3-full", "checks": {"ended": False}},
        {"source": "one", "model": "v4", "checks": {"ended": True}},
        {"source": "two", "model": "v3-full", "checks": {"ended": True}},
        {"source": "two", "model": "v4", "checks": {"ended": True}},
    ]
    assert paired_changes(rows)["ended"] == {"gained": 1, "lost": 0, "unchanged": 1}

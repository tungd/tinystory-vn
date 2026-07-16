from scripts.evaluate_v5_comparison import paired_changes


def test_v5_paired_changes_compare_against_v3():
    rows = [
        {"source": "s", "model": "v3-full", "checks": {"exact_character": False}},
        {"source": "s", "model": "v5", "checks": {"exact_character": True}},
    ]
    assert paired_changes(rows)["exact_character"] == {
        "gained": 1, "lost": 0, "unchanged": 0
    }

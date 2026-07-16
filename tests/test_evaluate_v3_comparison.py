from scripts.evaluate_v3_comparison import checks, contains_phrase, paired_changes


def test_contains_phrase_respects_word_boundaries():
    assert contains_phrase("a kind fox", "There lived a kind fox.")
    assert not contains_phrase("a kind fox", "There lived a kind foxhound.")


def test_checks_require_exact_character_and_moral():
    row = {
        "character": "a kind fox",
        "moral": "sharing is caring",
        "story": "A kind fox helped everyone. Sharing is caring.",
        "ended": True,
    }
    result = checks(row)
    assert result["exact_both"] is True
    assert result["exact_moral_near_end"] is True


def test_paired_changes_counts_gains_and_losses():
    rows = [
        {"source": "one", "model": "v2", "checks": {"ended": False}},
        {"source": "one", "model": "v3-full", "checks": {"ended": True}},
        {"source": "two", "model": "v2", "checks": {"ended": True}},
        {"source": "two", "model": "v3-full", "checks": {"ended": False}},
    ]
    assert paired_changes(rows)["ended"] == {"gained": 1, "lost": 1, "unchanged": 0}

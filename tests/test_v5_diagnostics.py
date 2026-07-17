from scripts.evaluate_v5_diagnostics import evaluate, similarity
from scripts.generate_v5_diagnostics import diagnostic_groups, select_human_controls


def row(source: str, moral: str = "kindness helps everyone") -> dict:
    return {
        "source": source,
        "source_type": "human-authored",
        "character": f"a patient fox {source}",
        "moral": moral,
        "target": f"A story about {source}.\n\nMoral: {moral}\n</story>",
    }


def test_select_human_controls_deduplicates_repeated_training_rows():
    rows = [row(str(i)) for i in range(4)] + [row("0")]
    selected = select_human_controls(rows, 4, 42)
    assert len(selected) == 4
    assert len({item["source"] for item in selected}) == 4


def test_diagnostic_groups_match_sources_and_swap_morals():
    train = [row(f"t{i}", f"train moral {i}") for i in range(4)]
    validation = [row(f"v{i}", f"valid moral {i}") for i in range(4)]
    groups = diagnostic_groups(train, validation, count=4, lengths=(120, 180))
    assert len(groups) == 8
    original = next(g for g in groups if g["split"] == "train" and g["condition"] == "original" and g["max_new_tokens"] == 180)
    swapped = next(g for g in groups if g["split"] == "train" and g["condition"] == "swapped_moral")
    assert [r["source"] for r in original["controls"]] == [r["source"] for r in swapped["controls"]]
    assert all(a["requested_moral"] != b["requested_moral"] for a, b in zip(original["controls"], swapped["controls"]))


def generated(source: str, split: str, condition: str, story: str, requested: str) -> dict:
    return {
        "source": source,
        "split": split,
        "condition": condition,
        "max_new_tokens": 180,
        "character": "a patient fox",
        "original_moral": "kindness helps everyone",
        "requested_moral": requested,
        "story": story,
        "ended": True,
    }


def test_evaluation_reports_moral_sensitivity():
    rows = []
    for split in ("train", "holdout"):
        rows.extend([
            generated("s", split, "original", "A patient fox helps.\nMoral: kindness helps everyone", "kindness helps everyone"),
            generated("s", split, "swapped_moral", "A patient fox waits.\nMoral: patience wins", "patience wins"),
            generated("s", split, "blank_moral", "A patient fox wanders.", ""),
        ])
    result = evaluate(rows)
    assert result["groups"]["train:original:180"]["exact_both"] == 1.0
    assert result["moral_sensitivity"]["holdout"]["swapped_requested_moral_rate"] == 1.0
    assert similarity("Same body.\nMoral: one", "Same body.\nMoral: two") == 1.0

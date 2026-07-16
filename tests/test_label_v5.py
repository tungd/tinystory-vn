from scripts.label_v5 import (
    build_annotation_prompt,
    parse_annotation,
    latest_by_source,
    stratified_rows,
)


STORY = "A Lion spared a mouse. Later the mouse freed the Lion from a net."


def test_annotation_prompt_forbids_rewriting():
    prompt = build_annotation_prompt({
        "collection": "Aesop",
        "title": "Lion and Mouse",
        "story": STORY,
    }).lower()
    assert "do not rewrite" in prompt
    assert "causally demonstrated" in prompt


def test_parse_annotation_accepts_strict_exact_label():
    raw = """{
      "protagonist_anchor": "A Lion",
      "trait": "merciful",
      "moral": "kindness is often repaid in unexpected ways",
      "coherence": 5,
      "moral_causality": 5,
      "complete": true,
      "child_suitable": true,
      "accept": true,
      "reason": "The spared mouse later frees the lion."
    }"""
    annotation = parse_annotation(raw, STORY)
    assert annotation["accepted"] is True
    assert annotation["trait"] == "merciful"


def test_parse_annotation_rejects_invented_anchor_and_weak_moral():
    raw = """{
      "protagonist_anchor": "The brave tiger",
      "trait": "brave",
      "moral": "be good",
      "coherence": 5,
      "moral_causality": 2,
      "complete": true,
      "child_suitable": true,
      "accept": true,
      "reason": "Weak."
    }"""
    annotation = parse_annotation(raw, STORY)
    assert annotation["accepted"] is False
    assert "anchor_not_exact" in annotation["rejection_reasons"]
    assert "weak_moral_causality" in annotation["rejection_reasons"]


def test_stratified_rows_puts_each_collection_before_repeats():
    rows = [
        {"source": "a1", "collection": "a"},
        {"source": "a2", "collection": "a"},
        {"source": "b1", "collection": "b"},
        {"source": "c1", "collection": "c"},
    ]
    selected = stratified_rows(rows, seed=7)
    assert {row["collection"] for row in selected[:3]} == {"a", "b", "c"}


def test_latest_annotation_replaces_prior_api_error():
    rows = [
        {"source": "s1", "annotation": {"rejection_reasons": ["api_error"]}},
        {"source": "s1", "annotation": {"accepted": True, "rejection_reasons": []}},
    ]
    assert latest_by_source(rows)["s1"]["annotation"]["accepted"] is True

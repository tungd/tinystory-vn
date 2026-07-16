from scripts.summarize_v5_annotations import summarize


def row(source: str, accepted: bool) -> dict:
    return {
        "source": source,
        "collection": "Aesop",
        "title": "Fox",
        "story": (
            "A patient Fox calmly helped every animal cross a dangerous river. "
            + "He listened, carried stones, and chose a safe path for every animal. " * 8
        ),
        "annotation": {
            "accepted": accepted,
            "protagonist_anchor": "A patient Fox",
            "trait": "patient",
            "moral": "patience helps friends overcome difficult problems",
            "rejection_reasons": [] if accepted else ["low_coherence"],
        },
    }


def test_summary_uses_latest_annotation_and_preparation_filter():
    external = row("s3", True)
    external["source_split"] = "external_holdout"
    result = summarize([row("s1", False), row("s1", True), row("s2", False), external])
    assert result["annotated"] == 3
    assert result["annotation_accepted"] == 2
    assert result["prepared_after_cleanup"] == 1
    assert result["external_controls_after_cleanup"] == 1
    assert result["rejection_reasons"] == {"low_coherence": 1}

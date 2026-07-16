import json

from scripts.prepare_v5 import build_real_example, prepare_v5


def annotation(source: str, *, accepted: bool = True) -> dict:
    story = f"A patient Fox named {source} crossed a river and calmly helped his friends."
    return {
        "source": source,
        "collection": "Human Fables",
        "title": f"Story {source}",
        "story": story,
        "annotation": {
            "accepted": accepted,
            "protagonist_anchor": "A patient Fox",
            "moral": "patience helps a community overcome trouble",
        },
    }


def replay(index: int) -> dict:
    return {
        "character": "a replay fox",
        "moral": "old lessons remain useful",
        "prompt": "prompt",
        "target": f"replay story {index}</story>",
    }


def test_real_example_preserves_story_and_exact_anchor():
    row = annotation("one")
    example = build_real_example(row)
    assert example is not None
    assert example["character"] == "A patient Fox"
    assert example["target"].startswith(row["story"])
    assert example["target"].endswith(
        "Moral: patience helps a community overcome trouble\n</story>"
    )


def test_real_example_rejects_nonexact_anchor():
    row = annotation("one")
    row["annotation"]["protagonist_anchor"] = "The patient Fox"
    assert build_real_example(row) is None


def test_prepare_v5_uses_latest_annotations_and_separates_real_validation(tmp_path):
    annotations = [annotation(str(i)) for i in range(20)]
    annotations.insert(0, annotation("0", accepted=False))
    tokenizer = tmp_path / "tokenizer.json"
    controls = tmp_path / "controls.json"
    tokenizer.write_text('{"version":"test"}')
    controls.write_text("[]")
    out = tmp_path / "prepared"
    meta = prepare_v5(
        annotations,
        [replay(i) for i in range(30)],
        out,
        tokenizer,
        controls,
        validation_fraction=0.2,
        real_repeats=2,
        replay_ratio=1.0,
        seed=7,
    )
    train = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    validation = [
        json.loads(line) for line in (out / "validation.jsonl").read_text().splitlines()
    ]
    train_real_sources = {
        row["source"] for row in train if row["source_type"] == "public-domain-human"
    }
    assert meta["real_accepted"] == 20
    assert train_real_sources.isdisjoint({row["source"] for row in validation})
    assert sum(row["source_type"] == "v3-replay" for row in train) == len(train_real_sources)
    assert len(train) == len(train_real_sources) * 3

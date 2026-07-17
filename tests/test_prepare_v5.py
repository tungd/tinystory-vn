import json

from scripts.prepare_v5 import (
    build_external_control,
    build_real_example,
    inject_character,
    prepare_v5,
    restrict_annotations,
)


def annotation(source: str, *, accepted: bool = True) -> dict:
    story = (
        f"A patient Fox named {source} crossed a river and calmly helped his friends. "
        + "He listened, carried stones, and chose a safe path for every animal. " * 8
    )
    return {
        "source": source,
        "collection": "Human Fables",
        "title": f"Story {source}",
        "story": story,
        "annotation": {
            "accepted": accepted,
            "protagonist_anchor": "A patient Fox",
            "trait": "patient",
            "moral": "patience helps a community overcome trouble",
        },
    }


def replay(index: int) -> dict:
    return {
        "character": "a replay fox",
        "moral": "old lessons remain useful",
        "prompt": "prompt",
        "target": f"replay story {index}\n\nMoral: old lessons remain useful\n</story>",
    }


def test_real_example_preserves_story_and_exact_anchor():
    row = annotation("one")
    example = build_real_example(row)
    assert example is not None
    assert example["character"] == "A patient Fox"
    assert example["target"].startswith(row["story"].strip())
    assert example["target"].endswith(
        "Moral: patience helps a community overcome trouble\n</story>"
    )


def test_real_example_rejects_nonexact_anchor():
    row = annotation("one")
    row["annotation"]["protagonist_anchor"] = "The patient Fox"
    assert build_real_example(row) is None


def test_real_example_rejects_nonadjective_trait():
    row = annotation("one")
    row["annotation"]["trait"] = "misunderstanding"
    assert build_real_example(row) is None


def test_real_example_rejects_outdated_child_unsuitable_language():
    row = annotation("one")
    row["story"] += " It used an outdated tale about Hottentots."
    assert build_real_example(row) is None


def test_real_example_rejects_story_over_generation_budget():
    row = annotation("one")
    row["story"] = "A patient Fox " + "carefully helped friends. " * 130
    row["annotation"]["protagonist_anchor"] = "A patient Fox"
    assert build_real_example(row) is None


def test_modern_paraphrase_accepts_short_complete_fable():
    row = annotation("modern")
    row["collection"] = "Understanding Fables"
    row["story"] = "A patient Fox " + "helped every friend cross safely. " * 10
    row["annotation"]["protagonist_anchor"] = "A patient Fox"
    assert 50 <= len(row["story"].split()) < 70
    assert build_real_example(row) is not None


def test_external_holdout_becomes_control_not_training_example():
    row = annotation("external")
    row["source_split"] = "external_holdout"
    assert build_real_example(row) is None
    control = build_external_control(row)
    assert control is not None
    assert control["source"] == "external"
    assert control["character"] in control["reference_story"]


def test_injects_trait_and_repairs_article_once():
    character, story = inject_character(
        "A Owl watched while another Owl slept.", "A Owl", "observant"
    )
    assert character == "An observant Owl"
    assert story == "An observant Owl watched while another Owl slept."


def test_injects_trait_after_count_for_plural_anchor():
    character, story = inject_character(
        "The three Fishes crossed the pond.", "The three Fishes", "resourceful"
    )
    assert character == "The three resourceful Fishes"
    assert story.startswith(character)


def test_injection_does_not_match_noun_prefix():
    character, story = inject_character(
        "The Goatherd warned the Goat.", "the Goat", "truthful"
    )
    assert character == "the truthful Goat"
    assert story == "The Goatherd warned the truthful Goat."


def test_prepare_v5_uses_latest_annotations_and_separates_real_validation(tmp_path):
    annotations = [annotation(str(i)) for i in range(20)]
    annotations.insert(0, annotation("0", accepted=False))
    external = annotation("external")
    external["source_split"] = "external_holdout"
    annotations.append(external)
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
        row["source"] for row in train if row["source_type"] == "human-authored"
    }
    assert meta["real_accepted"] == 20
    assert meta["external_controls"] == 1
    assert train_real_sources.isdisjoint({row["source"] for row in validation})
    assert sum(row["source_type"] == "v3-replay" for row in train) == len(train_real_sources)
    assert len(train) == len(train_real_sources) * 3
    assert len(json.loads((out / "external_controls.json").read_text())) == 1


def test_prepare_v5_excludes_replay_with_extra_moral_marker(tmp_path):
    rows = [annotation(str(i)) for i in range(10)]
    malformed = replay(99)
    malformed["target"] = (
        "Moral: accidental marker\n\nMoral: old lessons remain useful\n</story>"
    )
    tokenizer = tmp_path / "tokenizer.json"
    controls = tmp_path / "controls.json"
    tokenizer.write_text('{"version":"test"}')
    controls.write_text("[]")
    out = tmp_path / "prepared"
    prepare_v5(
        rows,
        [malformed] + [replay(i) for i in range(20)],
        out,
        tokenizer,
        controls,
        validation_fraction=0.2,
        real_repeats=1,
        replay_ratio=1.0,
    )
    train = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert all(row["target"].count("Moral:") == 1 for row in train)


def test_restrict_annotations_excludes_stale_sources_and_requires_complete_success():
    current = [annotation("current")]
    stale = annotation("stale")
    assert restrict_annotations(current + [stale], [{"source": "current"}]) == current

    failed = annotation("current")
    failed["annotation"]["rejection_reasons"] = ["api_error"]
    try:
        restrict_annotations([failed], [{"source": "current"}])
    except ValueError as error:
        assert "api_errors=['current']" in str(error)
    else:
        raise AssertionError("API error must block v5 preparation")

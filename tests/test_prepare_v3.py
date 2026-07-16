import json

from scripts.prepare_v3 import (
    build_example,
    character_headword,
    parse_formatted_v2,
    prepare_v3,
    story_mentions_character,
    story_mentions_exact_character,
)


V2_SAMPLE = (
    "<char> a wise old tortoise </char>\n"
    "<moral> patience defeats haste </moral>\n"
    "<story>\n"
    "A wise old tortoise lived beside a river. A hurried hare ignored her advice and fell into mud. "
    "The tortoise calmly found a branch and pulled him free. The hare learned to slow down.\n"
    "</story>"
)


def test_parse_formatted_v2_recovers_exact_fields():
    character, moral, story = parse_formatted_v2(V2_SAMPLE)
    assert character == "a wise old tortoise"
    assert moral == "patience defeats haste"
    assert story.startswith("A wise old tortoise")


def test_character_headword_and_story_match():
    assert character_headword("a brave little mouse") == "mouse"
    assert story_mentions_character("a clever fox", "Two foxes crossed the river.")
    assert not story_mentions_character("a proud lion", "A snake found a crystal.")


def test_exact_character_phrase_match():
    assert story_mentions_exact_character("a clever fox", "A clever   fox crossed the river.")
    assert not story_mentions_exact_character("a clever fox", "The fox crossed the river.")


def test_v3_target_appends_exact_moral():
    example = build_example(V2_SAMPLE)
    assert example is not None
    assert example["prompt"].endswith("<story>\n")
    assert example["target"].endswith("Moral: patience defeats haste\n</story>")


def test_v3_rejects_story_without_requested_character():
    bad = V2_SAMPLE.replace("a wise old tortoise </char>", "a proud lion </char>")
    assert build_example(bad) is None


def test_v3_rejects_headword_only_character_match():
    bad = V2_SAMPLE.replace("A wise old tortoise lived", "The tortoise lived")
    assert build_example(bad) is None


def test_prepare_v3_is_deterministic_and_reuses_tokenizer(tmp_path):
    tokenizer = tmp_path / "source-tokenizer.json"
    tokenizer.write_text('{"version":"test"}')
    records = [
        V2_SAMPLE,
        V2_SAMPLE.replace("tortoise", "fox").replace("patience defeats haste", "wit beats force"),
        V2_SAMPLE.replace("tortoise", "mouse").replace("patience defeats haste", "kindness returns"),
    ]
    out = tmp_path / "v3"
    meta = prepare_v3(records, out, tokenizer, validation_fraction=1 / 3, seed=7)

    assert meta["accepted"] == 3
    assert meta["train"] == 2
    assert meta["validation"] == 1
    assert json.loads((out / "tokenizer.json").read_text()) == {"version": "test"}
    assert len((out / "train.jsonl").read_text().splitlines()) == 2

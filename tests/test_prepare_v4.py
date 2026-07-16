import json

from scripts.prepare_v4 import build_example, clean_story, prepare_v4, tail_supports_moral


def record(index: int, *, supported: bool = True) -> dict:
    character = f"a patient fox{index}"
    moral = "patience defeats haste"
    opening = f"In a bright forest, {character} watched the morning river. "
    middle = "The animals carried stones and listened to one another. " * 30
    ending = (
        "At sunset, patience helped the fox finish safely while haste caused mistakes."
        if supported else
        "At sunset, everyone went home after sharing berries."
    )
    return {
        "language": "en",
        "prompt_hash": f"hash-{index}",
        "prompt": (
            f"- Main Character: {character}\n"
            "- Setting: a bright forest\n"
            "- Challenge: a difficult crossing\n"
            "- Outcome: careful work succeeds\n"
            f"- Teaching: {moral}"
        ),
        "fable": f"**A River Lesson**\n\n{opening}{middle}{ending}\n\nMoral: {moral}",
    }


def test_clean_story_removes_markdown_and_existing_moral_footer():
    cleaned = clean_story("**A Title**\n\nA story.\n\nMoral: patience wins", "patience wins")
    assert cleaned == "A Title\n\nA story."


def test_clean_story_removes_meta_note_and_normalizes_inline_moral():
    cleaned = clean_story(
        "A story echoed the moral: patience wins.\n\n"
        "Moral: patience wins\n\n(Note: Written for children.)",
        "patience wins",
    )
    assert cleaned == "A story echoed the lesson that patience wins."


def test_v4_target_has_one_canonical_moral_and_scaffold_metadata():
    example = build_example(record(1))
    assert example is not None
    assert example["target"].count("Moral:") == 1
    assert "**" not in example["target"]
    assert example["setting"] == "a bright forest"


def test_v4_rejects_footer_only_moral_alignment():
    assert build_example(record(2, supported=False)) is None
    assert not tail_supports_moral("Everyone shared berries.", "patience defeats haste")


def test_prepare_v4_writes_splits_and_fresh_controls(tmp_path):
    tokenizer = tmp_path / "tokenizer-source.json"
    tokenizer.write_text('{"version":"test"}')
    records = [record(i) for i in range(20)]
    meta = prepare_v4(
        records,
        tmp_path / "v4",
        tokenizer,
        skip_valid=0,
        limit=10,
        validation_fraction=0.5,
        eval_controls=3,
        seed=7,
    )
    assert meta["accepted"] == 10
    assert meta["train"] + meta["validation"] == 10
    assert meta["eval_controls"] == 3
    controls = json.loads((tmp_path / "v4" / "eval_controls.json").read_text())
    assert all("valid-index" in row["source"] for row in controls)
    assert json.loads((tmp_path / "v4" / "tokenizer.json").read_text()) == {
        "version": "test"
    }

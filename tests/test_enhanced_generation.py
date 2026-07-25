from app.enhanced_generation import enhance_story, normalize_moral_line, validate_story


PROMPT = {
    "character": "a small turtle",
    "setting": "a quiet pond",
    "challenge": "crosses a road",
    "outcome": "friends help",
    "teaching": "patience and teamwork matter",
}


def test_normalize_adds_missing_moral():
    story = "A small turtle waited with friends and crossed safely."

    fixed = normalize_moral_line(story, PROMPT["teaching"])

    assert fixed.endswith("Moral: patience and teamwork matter.")


def test_normalize_fills_empty_moral():
    story = "A small turtle waited with friends and crossed safely.\n\nMoral:"

    fixed = normalize_moral_line(story, PROMPT["teaching"])

    assert fixed.endswith("Moral: patience and teamwork matter.")
    assert fixed.count("Moral:") == 1


def test_validate_flags_bullets_as_severe():
    story = (
        "In a quiet pond, a small turtle learned with friends.\n\n"
        "- First, he waited.\n"
        "- Then, he crossed.\n\n"
        "Moral: patience and teamwork matter"
    )

    result = validate_story(story, PROMPT)

    assert "contains_bullets" in result.severe_reasons


def test_enhance_postprocesses_moral_without_rewrite():
    story = "In a quiet pond, a small turtle waited for friends and crossed safely."

    result = enhance_story(story, PROMPT)

    assert result["story"].endswith("Moral: patience and teamwork matter.")
    assert result["actions"] == ["moral_postprocess"]


def test_enhance_falls_back_when_rewrite_fails():
    story = (
        "In a quiet pond, a small turtle waited.\n\n"
        "- Then friends helped.\n\n"
        "Moral:"
    )

    def broken_generate_meta_fn(**kwargs):
        raise RuntimeError("rewrite unavailable")

    result = enhance_story(
        story,
        PROMPT,
        rewrite_model="missing-model",
        generate_meta_fn=broken_generate_meta_fn,
    )

    assert "rewrite_failed" in result["actions"]
    assert result["story"].endswith("Moral: patience and teamwork matter.")

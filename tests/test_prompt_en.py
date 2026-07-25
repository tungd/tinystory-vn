from app.prompt_en import build_fable_prompt, LENGTH_NUM_PREDICT


def test_prompt_includes_filled_elements_only():
    p = build_fable_prompt(character="a clever fox", teaching="honesty pays")
    assert "a clever fox" in p and "honesty pays" in p
    assert "Setting" not in p  # empty field is skipped


def test_prompt_empty_gives_generic_instruction():
    p = build_fable_prompt()
    assert "fable" in p.lower()


def test_length_map():
    assert set(LENGTH_NUM_PREDICT) == {"short", "medium", "long"}

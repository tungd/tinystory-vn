from app.guardrail.output_filter import check_output, OutputDecision


def test_clean_story_passes():
    d = check_output("Once there was a kind rabbit. Moral: be kind to others.")
    assert isinstance(d, OutputDecision)
    assert d.ok is True


def test_story_with_banned_word_fails():
    d = check_output("The rabbit said a fucking word.")
    assert d.ok is False
    assert d.reason


def test_empty_output_fails():
    d = check_output("   ")
    assert d.ok is False

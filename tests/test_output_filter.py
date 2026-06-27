from app.guardrail.output_filter import check_output, OutputDecision


def test_clean_story_passes():
    d = check_output("Ngày xưa có một chú thỏ tốt bụng. Bài học: hãy tử tế.")
    assert isinstance(d, OutputDecision)
    assert d.ok is True


def test_story_with_banned_word_fails():
    d = check_output("Con thỏ chửi đụ má con rùa.")
    assert d.ok is False
    assert d.reason


def test_empty_output_fails():
    d = check_output("   ")
    assert d.ok is False

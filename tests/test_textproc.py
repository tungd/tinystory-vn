from app.textproc import trim_to_last_sentence


def test_complete_sentence_unchanged():
    s = "The fox shared its food. And they all lived happily."
    assert trim_to_last_sentence(s) == s


def test_trailing_quote_unchanged():
    s = 'The owl said, "Kindness always wins."'
    assert trim_to_last_sentence(s) == s


def test_cut_midsentence_trims_to_last_period():
    s = "The fox found food. She shared it with everyone. But then the wise"
    assert trim_to_last_sentence(s) == "The fox found food. She shared it with everyone."


def test_cut_keeps_closing_quote_after_terminator():
    s = 'The bear smiled. "We are friends now!" The little rabbit ran off toward the'
    assert trim_to_last_sentence(s) == 'The bear smiled. "We are friends now!"'


def test_no_terminator_returns_original():
    s = "once upon a time a fox was walking"
    assert trim_to_last_sentence(s) == s


def test_empty_returns_empty():
    assert trim_to_last_sentence("") == ""


def test_strips_trailing_whitespace_after_trim():
    s = "A complete tale. Then something incomplete began to"
    assert trim_to_last_sentence(s) == "A complete tale."

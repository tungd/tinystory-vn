from app.metrics import distinct_n, self_bleu, flesch_reading_ease


def test_distinct_n_all_unique():
    assert distinct_n(["a b c d"], 1) == 1.0


def test_distinct_n_with_repeats():
    # tokens: a a a a -> 1 unique / 4 total = 0.25
    assert distinct_n(["a a a a"], 1) == 0.25


def test_self_bleu_identical_texts_high():
    v = self_bleu(["the fox ran fast", "the fox ran fast"], n=2)
    assert v > 0.9


def test_self_bleu_disjoint_texts_low():
    v = self_bleu(["alpha beta gamma delta", "one two three four"], n=2)
    assert v < 0.1


def test_flesch_returns_number():
    assert isinstance(flesch_reading_ease("The fox ran. The end."), float)

from scripts.metrics import (
    distinct_n,
    flesch_reading_ease,
    self_bleu,
    aggregate_metrics,
)

STORY = (
    "The clever fox lived in a foggy marsh. He tricked the heron and escaped. "
    "The other animals learned that wit wins where strength fails. The marsh was peaceful again."
)


def test_flesch_reading_ease_runs():
    assert isinstance(flesch_reading_ease(STORY), float)


def test_distinct_n_in_range():
    d1 = distinct_n([STORY, STORY])
    assert 0.0 <= d1 <= 1.0
    assert distinct_n(["a a a", "a a a"]) == 0.0 or distinct_n(["a b c", "d e f"]) > 0


def test_self_bleu_low_for_diverse():
    s = self_bleu([STORY, "A tiny mouse freed a lion from a net and they became friends."])
    assert 0.0 <= s <= 1.0


def test_aggregate_metrics_shape():
    m = aggregate_metrics([STORY])
    assert set(["distinct_1", "distinct_2", "self_bleu", "flesch_reading_ease"]).issubset(m)

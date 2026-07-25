from trieulh.scripts.gen_preference_pairs import make_pair


S_HI = {"grammar": 8, "creativity": 7, "moral_clarity": 8, "prompt_adherence": 9, "overall": 8.0}
S_LO = {"grammar": 6, "creativity": 6, "moral_clarity": 6, "prompt_adherence": 6, "overall": 6.0}
S_MID = {"grammar": 7, "creativity": 7, "moral_clarity": 7, "prompt_adherence": 8, "overall": 7.25}


def test_make_pair_picks_chosen_by_overall():
    p = make_pair("prompt x", "story a", "story b", S_LO, S_HI, min_margin=1.0)
    assert p is not None
    assert p["chosen"] == "story b" and p["rejected"] == "story a"
    assert p["score_chosen"] == 8.0 and p["score_rejected"] == 6.0
    assert p["prompt"] == "prompt x"


def test_make_pair_filters_low_margin():
    # chênh 0.75 < 1.0 -> loại (pair mơ hồ)
    assert make_pair("p", "a", "b", S_MID, S_HI, min_margin=1.0) is None


def test_make_pair_filters_zero_scores():
    # judge lỗi trả overall 0 -> loại
    bad = {**S_LO, "overall": 0.0}
    assert make_pair("p", "a", "b", bad, S_HI, min_margin=1.0) is None


def test_make_pair_filters_identical_stories():
    assert make_pair("p", "same", "same", S_LO, S_HI, min_margin=1.0) is None

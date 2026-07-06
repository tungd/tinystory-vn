from app.judge import parse_scores, build_judge_prompt


def test_build_judge_prompt_mentions_axes():
    p = build_judge_prompt("story...", "prompt...")
    for k in ["grammar", "creativity", "moral", "adherence"]:
        assert k.lower() in p.lower()


def test_parse_scores_from_json():
    raw = '{"grammar": 9, "creativity": 7, "moral_clarity": 8, "prompt_adherence": 10}'
    s = parse_scores(raw)
    assert s["grammar"] == 9 and s["prompt_adherence"] == 10
    assert s["overall"] == round((9+7+8+10)/4, 2)


def test_parse_scores_tolerates_extra_text():
    raw = 'Here are the scores: {"grammar":8,"creativity":6,"moral_clarity":7,"prompt_adherence":9} thanks'
    s = parse_scores(raw)
    assert s["creativity"] == 6


def test_parse_scores_missing_defaults_zero():
    s = parse_scores('{"grammar": 8}')
    assert s["creativity"] == 0

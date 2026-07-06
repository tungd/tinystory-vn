from app.judge import parse_scores, build_judge_prompt


def test_build_judge_prompt_mentions_axes():
    p = build_judge_prompt("story...", "prompt...")
    for k in ["grammar", "creativity", "moral", "adherence"]:
        assert k.lower() in p.lower()


def test_build_judge_prompt_asks_for_reason():
    p = build_judge_prompt("story...", "prompt...")
    assert "reason" in p.lower()
    assert "score" in p.lower()


def test_parse_scores_from_flat_json():
    # Legacy flat shape still supported.
    raw = '{"grammar": 9, "creativity": 7, "moral_clarity": 8, "prompt_adherence": 10}'
    s = parse_scores(raw)
    assert s["grammar"] == 9 and s["prompt_adherence"] == 10
    assert s["overall"] == round((9 + 7 + 8 + 10) / 4, 2)
    assert s["rationale"] == {}


def test_parse_scores_from_nested_json_with_reasons():
    raw = (
        '{"grammar": {"score": 9, "reason": "Clean sentences: \'The fox ran.\'"}, '
        '"creativity": {"score": 7, "reason": "Familiar plot"}, '
        '"moral_clarity": {"score": 8, "reason": "Moral stated at end"}, '
        '"prompt_adherence": {"score": 10, "reason": "Has fox, marsh, heron"}}'
    )
    s = parse_scores(raw)
    assert s["grammar"] == 9 and s["prompt_adherence"] == 10
    assert s["overall"] == round((9 + 7 + 8 + 10) / 4, 2)
    assert s["rationale"]["grammar"].startswith("Clean sentences")
    assert s["rationale"]["prompt_adherence"] == "Has fox, marsh, heron"


def test_parse_scores_tolerates_extra_text_and_nested_braces():
    raw = (
        'Here are the scores: {"grammar": {"score": 8, "reason": "ok {inline} brace"}, '
        '"creativity": {"score": 6, "reason": "meh"}, '
        '"moral_clarity": {"score": 7, "reason": "clear"}, '
        '"prompt_adherence": {"score": 9, "reason": "on topic"}} thanks!'
    )
    s = parse_scores(raw)
    assert s["creativity"] == 6
    assert s["rationale"]["grammar"] == "ok {inline} brace"


def test_parse_scores_recovers_from_malformed_json():
    # Real failure mode: model closes one axis object with ")" instead of "}",
    # which breaks whole-object json.loads. Per-axis regex must still recover all.
    raw = (
        '{"grammar": {"score": 10, "reason": "Clean sentences"}, '
        '"creativity": {"score": 7, "reason": "simple twists"), '
        '"moral_clarity": {"score": 10, "reason": "Moral explicit"}, '
        '"prompt_adherence": {"score": 9, "reason": "Has honeybee and teaching"}}'
    )
    s = parse_scores(raw)
    assert s["grammar"] == 10 and s["creativity"] == 7
    assert s["moral_clarity"] == 10 and s["prompt_adherence"] == 9
    assert s["overall"] == round((10 + 7 + 10 + 9) / 4, 2)
    assert s["rationale"]["creativity"] == "simple twists"
    assert s["rationale"]["prompt_adherence"] == "Has honeybee and teaching"


def test_parse_scores_missing_defaults_zero():
    s = parse_scores('{"grammar": {"score": 8, "reason": "fine"}}')
    assert s["creativity"] == 0
    assert "creativity" not in s["rationale"]
    assert s["rationale"]["grammar"] == "fine"

import random
from trieulh.scripts.prepare_tf1_pretrain import build_record

ROW = {
    "prompt": ("- Main Character: a fox\n- Setting: a marsh\n- Challenge: hunger\n"
               "- Outcome: shares food\n- Teaching: sharing is caring\n"),
    "fable": "A fox found food and shared it. Moral: sharing is caring.",
    "prompt_hash": "abc123",
}

def test_build_record_returns_text_and_cond_len():
    rec = build_record(ROW, random.Random(0))
    assert rec is not None
    assert "text" in rec and "cond_len" in rec
    assert rec["text"][rec["cond_len"]:].startswith("<|story|>")

def test_build_record_skips_empty_fable():
    row = {**ROW, "fable": "   "}
    assert build_record(row, random.Random(0)) is None


def _row(fable):
    return {"prompt": "- Main Character: a fox\n- Teaching: be kind\n",
            "fable": fable, "prompt_hash": "h1"}


def test_build_record_rejects_too_short():
    # 3-word fable, min_words=5 -> rejected
    assert build_record(_row("A short tale."), random.Random(0), min_words=5) is None


def test_build_record_rejects_too_long():
    long_fable = " ".join(["word"] * 400)
    assert build_record(_row(long_fable), random.Random(0), max_words=320) is None


def test_build_record_accepts_in_range():
    ok_fable = " ".join(["word"] * 100) + " moral: be kind."
    rec = build_record(_row(ok_fable), random.Random(0), min_words=60, max_words=320)
    assert rec is not None and "text" in rec and "cond_len" in rec


def _rows_with_owl(n_owl, n_plain):
    base = " ".join(["word"] * 80)
    rows = []
    for i in range(n_owl):
        rows.append({"prompt": "- Main Character: a fox\n",
                     "fable": base + " A wise old owl helped everyone.",
                     "prompt_hash": f"owl{i}"})
    for i in range(n_plain):
        rows.append({"prompt": "- Main Character: a fox\n",
                     "fable": base + " The fox solved it alone.",
                     "prompt_hash": f"plain{i}"})
    return rows


def test_write_split_caps_phrase_fraction(tmp_path):
    from trieulh.scripts.prepare_tf1_pretrain import _write_split
    # 50 owl rows first, then 100 plain rows; cap owl at 10% of written
    rows = _rows_with_owl(50, 100)
    out = tmp_path / "train.jsonl"
    n = _write_split(rows, out, n=100, seed=13, min_words=60, max_words=320,
                     cap_phrase="wise old owl", cap_frac=0.10)
    assert n == 100
    import json as _j
    texts = [_j.loads(l)["text"].lower() for l in out.open()]
    owl = sum(1 for t in texts if "wise old owl" in t)
    assert owl <= 10   # at most 10% of 100


def test_write_split_no_cap_keeps_all(tmp_path):
    from trieulh.scripts.prepare_tf1_pretrain import _write_split
    rows = _rows_with_owl(5, 5)
    out = tmp_path / "train.jsonl"
    n = _write_split(rows, out, n=10, seed=13, min_words=60, max_words=320)
    assert n == 10

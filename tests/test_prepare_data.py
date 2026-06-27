import json
from pathlib import Path

from scripts.prepare_data import clean_text, build_records, split_records

FIX = Path(__file__).parent / "fixtures"


def test_clean_text_collapses_whitespace_and_strips():
    assert clean_text("Ngày  xưa  có\n\n") == "Ngày xưa có"


def test_build_records_dedupes_and_formats_instruction():
    raw = [json.loads(l) for l in (FIX / "sample_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    refusals = [json.loads(l) for l in (FIX / "sample_refusal.jsonl").read_text(encoding="utf-8").splitlines()]
    records = build_records(raw, refusals)
    stories = [r for r in records if r["type"] == "story"]
    refusal_recs = [r for r in records if r["type"] == "refusal"]
    # 2 dòng raw trùng nội dung sau khi clean -> còn 1
    assert len(stories) == 1
    assert stories[0]["instruction"].startswith("Viết một truyện ngụ ngôn cho trẻ em về chủ đề: lòng trung thực")
    assert len(refusal_recs) == 1
    assert refusal_recs[0]["type"] == "refusal"


def test_split_records_is_deterministic_and_partitions():
    records = [{"type": "story", "instruction": f"i{n}", "output": f"o{n}"} for n in range(10)]
    a = split_records(records, seed=42)
    b = split_records(records, seed=42)
    assert a == b
    total = len(a["train"]) + len(a["val"]) + len(a["test"])
    assert total == 10
    assert len(a["val"]) == 1 and len(a["test"]) == 1

import json
from pathlib import Path

from scripts.prepare_tf1 import (
    collect_records,
    format_record,
    parse_prompt_elements,
    passes_filter,
    split_records,
    write_dataset,
)


def test_format_record_uses_narrative_fields():
    raw = {
        "story": "A fox told the truth. Moral: Honesty earns trust.",
        "character": "a fox",
        "setting": "a market",
        "challenge": "he wants to cheat",
        "outcome": "he tells the truth",
        "moral": "honesty earns trust",
    }
    item = format_record(raw)
    assert item is not None
    assert item["instruction"].startswith("Write a short English fable")
    assert "Character: a fox" in item["input"]
    assert "Teaching: honesty earns trust" in item["input"]


def test_format_record_parses_tf1_prompt_fields():
    raw = {
        "prompt": (
            "Create a fable based on the following elements:\n"
            "  - Main Character: a persuasive firefly\n"
            "  - Setting: a canyon\n"
            "  - Challenge: a storm separates friends\n"
            "  - Outcome: the firefly is helped by an owl\n"
            "  - Teaching: timely help earns lasting loyalty\n"
        ),
        "fable": "A firefly was lost in a storm. An owl helped her home.",
    }
    item = format_record(raw)
    assert item is not None
    assert "Character: a persuasive firefly" in item["input"]
    assert "Setting: a canyon" in item["input"]
    assert item["output"].endswith("Moral: timely help earns lasting loyalty")


def test_parse_prompt_elements():
    elements = parse_prompt_elements("- Main Character: a fox\n- Teaching: honesty earns trust")
    assert elements["character"] == "a fox"
    assert elements["teaching"] == "honesty earns trust"


def test_format_record_accepts_minimal_story_with_moral():
    raw = {"text": "The ant helped a bee. Moral: Small kindness matters."}
    item = format_record(raw)
    assert item is not None
    assert "Teaching: Small kindness matters." in item["input"]


def test_filter_requires_word_bounds_and_moral():
    good = {"input": "", "output": " ".join(["word"] * 65) + " Moral: Be kind."}
    bad_short = {"input": "", "output": "Moral: Be kind."}
    bad_no_moral = {"output": " ".join(["word"] * 80)}
    assert passes_filter(good, min_words=60, max_words=100)
    assert not passes_filter(bad_short, min_words=60, max_words=100)
    assert not passes_filter({**bad_no_moral, "input": ""}, min_words=60, max_words=100)


def test_filter_removes_child_inappropriate_topics():
    bad = {
        "input": "Challenge: betrayal by a friend\nTeaching: be loyal",
        "output": " ".join(["word"] * 80) + " Moral: be loyal.",
    }
    assert not passes_filter(bad, min_words=60, max_words=100)


def test_collect_dedupes_and_limits():
    story = " ".join(["kind"] * 70) + " Moral: Be kind."
    rows = [{"story": story, "moral": "be kind"}, {"story": story, "moral": "be kind"}]
    assert len(collect_records(rows, target_n=2, min_words=10, max_words=100)) == 1


def test_split_and_write_dataset(tmp_path: Path):
    rows = [
        {
            "instruction": "i",
            "input": f"Character: c{i}",
            "output": " ".join(["word"] * 70) + " Moral: Be kind.",
        }
        for i in range(20)
    ]
    train, valid = split_records(rows, valid_ratio=0.1, seed=42)
    assert len(train) == 18
    assert len(valid) == 2

    out = tmp_path / "sft_20"
    write_dataset(out, rows, valid_ratio=0.1, seed=42)
    assert len((out / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 18
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total"] == 20

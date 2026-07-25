import random
from trieulh.scripts.tf1_pretrain.format import (
    length_bucket, apply_dropout, build_training_text, SEP, END,
)

def test_length_bucket():
    assert length_bucket(100) == "short"
    assert length_bucket(300) == "medium"
    assert length_bucket(700) == "long"

def test_apply_dropout_blanks_some_but_keeps_keys():
    slots = {"character": "a fox", "setting": "a marsh", "challenge": "c",
             "outcome": "o", "teaching": "t"}
    rng = random.Random(0)
    dropped = apply_dropout(slots, rng, p=1.0, p_all=0.0)
    assert set(dropped) == set(slots)           # keys preserved
    assert all(v == "" for v in dropped.values())  # p=1.0 blanks all non-all path
    assert slots["character"] == "a fox"        # original untouched

def test_build_training_text_has_separator_and_fable():
    slots = {"character": "a fox", "setting": "", "challenge": "",
             "outcome": "", "teaching": "be kind"}
    text, cond_len = build_training_text(slots, "Once upon a time. The end.", "short")
    assert SEP in text and text.endswith(END)
    assert text[cond_len:].startswith(SEP)      # fable region begins at the separator
    assert "be kind" in text[:cond_len]         # conditioning contains present slots
    assert "Setting" not in text[:cond_len]     # empty slots omitted by build_fable_prompt


def test_apply_dropout_per_slot_overrides():
    slots = {"character": "a fox", "setting": "a marsh", "challenge": "c",
             "outcome": "o", "teaching": "t"}
    rng = random.Random(0)
    # teaching/outcome never dropped (p=0), others always dropped (p=1)
    dropped = apply_dropout(slots, rng, p=1.0, p_all=0.0,
                            p_overrides={"teaching": 0.0, "outcome": 0.0})
    assert dropped["teaching"] == "t" and dropped["outcome"] == "o"
    assert dropped["character"] == "" and dropped["setting"] == "" and dropped["challenge"] == ""


def test_apply_dropout_default_unchanged_without_overrides():
    slots = {"character": "a fox", "teaching": "t"}
    a = apply_dropout(slots, random.Random(7), p=0.3, p_all=0.05)
    b = apply_dropout(slots, random.Random(7), p=0.3, p_all=0.05, p_overrides=None)
    assert a == b   # back-compat: None overrides = old behavior

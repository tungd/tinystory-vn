import json
from pathlib import Path

from scripts.fable_tokenizer import FableTokenizer
from scripts.prepare_tf1 import (
    format_sample,
    parse_elements,
    prepare_char,
    rec_is_valid,
)

FIX = Path(__file__).parent / "fixtures"
RECORDS = [json.loads(l) for l in (FIX / "sample_tf1.jsonl").read_text().splitlines()]


def test_parse_elements_structured():
    char, moral = parse_elements(RECORDS[0])
    assert char == "a clever fox"
    assert moral == "cleverness beats brute force"


def test_parse_elements_alt_keys():
    char, moral = parse_elements(RECORDS[1])
    assert char == "a brave little mouse"
    assert moral == "kindness returns to those who give it"


def test_parse_elements_from_tf1_prompt():
    char, moral = parse_elements(RECORDS[3])
    assert char == "a persuasive firefly"
    assert moral == "timely help earns lasting loyalty"


def test_rec_is_valid_requires_english():
    assert rec_is_valid({"prompt": "Main Character: x\nTeaching: y", "fable": "z" * 90, "language": "fr"}) is False
    assert rec_is_valid({"prompt": "Main Character: x\nTeaching: y", "fable": "z" * 90, "language": "en"}) is True


def test_format_sample_has_prefix_tags():
    s = format_sample(RECORDS[0])
    assert s.startswith("<char> a clever fox </char>")
    assert "<moral> cleverness beats brute force </moral>" in s
    assert "<story>" in s and "</story>" in s


def test_rec_is_valid_filters_empty_and_short():
    assert rec_is_valid(RECORDS[0]) is True
    assert rec_is_valid(RECORDS[2]) is False  # empty character/moral


def test_fable_tokenizer_roundtrip():
    tok = FableTokenizer()
    tok.train(["<char> fox </char>", "<moral> honesty </moral>"])
    ids = tok.encode("<char> fox </char>")
    assert tok.decode(ids) == "<char> fox </char>"
    assert tok.vocab_size > 0


def test_prepare_char_writes_bins(tmp_path):
    meta = prepare_char(RECORDS, tmp_path, seed=1)
    assert (tmp_path / "train.bin").exists()
    assert (tmp_path / "val.bin").exists()
    assert (tmp_path / "vocab.json").exists()
    assert meta["mode"] == "char"
    assert meta["n_samples"] == 3  # 3 valid records (HF-style one included)


def test_prepare_char_bin_is_uint16(tmp_path):
    prepare_char(RECORDS, tmp_path, seed=1)
    data = (tmp_path / "train.bin").read_bytes()
    assert len(data) % 2 == 0  # uint16 packed

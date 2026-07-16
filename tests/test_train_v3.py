import pytest

from scripts.train_v3 import encode_story_only, validate_output_path


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def encode(self, text):
        offset = 10 if text.startswith("<char>") else 20
        return FakeEncoding([offset + i for i, _ in enumerate(text.split())])


def test_encode_story_only_masks_prompt():
    encoded = encode_story_only(
        FakeTokenizer(), "<char> fox </char>", "A useful story </story>", 20, eos_id=99
    )
    assert encoded["labels"][:3] == [-100, -100, -100]
    assert encoded["labels"][3:] == encoded["input_ids"][3:]


def test_encode_story_only_forces_eos_when_truncated():
    encoded = encode_story_only(
        FakeTokenizer(), "<char> fox </char>", "one two three four five six", 6, eos_id=99
    )
    assert encoded["input_ids"][-1] == 99
    assert encoded["labels"][-1] == 99


def test_v3_output_must_not_overlap_v2(tmp_path):
    base = tmp_path / "v2" / "hf"
    base.mkdir(parents=True)
    with pytest.raises(ValueError):
        validate_output_path(base, base / "v3")
    with pytest.raises(ValueError):
        validate_output_path(base, tmp_path / "v2")


def test_v3_output_must_be_empty(tmp_path):
    base = tmp_path / "v2"
    base.mkdir()
    out = tmp_path / "v3"
    out.mkdir()
    (out / "checkpoint").write_text("occupied")
    with pytest.raises(FileExistsError):
        validate_output_path(base, out)

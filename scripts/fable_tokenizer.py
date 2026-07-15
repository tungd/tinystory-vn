"""Char-level tokenizer for the keyword-guided fable model.

Pure-Python, zero-dependency. The model is trained on a flat character stream where
each sequence is formatted as:

    <char> {character} </char>
    <moral> {moral} </moral>
    <story>
    {story}
    </story>

Tag strings ("<char>", etc.) are encoded literally as their constituent characters,
so the network learns them as delimiters without any special-token machinery.
"""

import json
from pathlib import Path


class FableTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None):
        # vocab: char -> id
        self.vocab: dict[str, int] = dict(vocab) if vocab else {}
        self.id2char: dict[int, str] = {i: c for c, i in self.vocab.items()}

    def train(self, texts: list[str]) -> None:
        chars: set[str] = set()
        for t in texts:
            chars.update(t)
        for c in sorted(chars):
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
        self.id2char = {i: c for c, i in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.vocab[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id2char.get(i, "") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"vocab": self.vocab}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "FableTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab=data["vocab"])

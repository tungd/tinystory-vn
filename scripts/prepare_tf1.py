"""Prepare TF1-EN-3M for training a ~200M fable transformer (from scratch).

Keyword guidance = seed elements only: main character + moral lesson (no RAG).

Two output modes:
  - `bpe`  (default, Colab): train a BPE tokenizer via `tokenizers`, write
           `fables.jsonl` (formatted texts) + `tokenizer.json`. The training
           notebook tokenizes with `datasets.map` and trains with
           `transformers.Trainer` (GPT2LMHeadModel, ~200M params).
  - `char` (offline fallback / local tests): char-level tokenizer, write
           nanoGPT-style `train.bin` / `val.bin` + `vocab.json`.

Common steps (pure-python, testable): parse the two keyword seeds and format
each fable as a prefixed string:

    <char> {character} </char>
    <moral> {moral} </moral>
    <story>
    {story}
    </story>
"""

import argparse
import json
import random
import struct
from pathlib import Path

from scripts.fable_tokenizer import FableTokenizer

TF1_DATASET = "klusai/ds-tf1-en-3m"

_CHARACTER_KEYS = ("character", "main_character", "char", "protagonist")
_MORAL_KEYS = ("moral", "teaching", "lesson", "theme")

_PREFIX = (
    "<char> {character} </char>\n"
    "<moral> {moral} </moral>\n"
    "<story>\n{story}\n</story>"
)


def _first(d: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _regex(text: str, pat: str) -> str:
    import re

    m = re.search(pat, text)
    return m.group(1).strip() if m else ""


def parse_elements(rec: dict) -> tuple[str, str]:
    """Extract (character, moral).

    Primary: TF1-EN-3M schema — seeds live inside the `prompt` field as
    'Main Character:' / 'Teaching:' lines. Falls back to structured keys or a
    generic regex scan of any text field.
    """
    prompt = rec.get("prompt") or ""
    character = _regex(prompt, r"Main Character:\s*([^\n]+)")
    moral = _regex(prompt, r"Teaching:\s*([^\n]+)")

    if not character:
        character = _first(rec, _CHARACTER_KEYS)
    if not moral:
        moral = _first(rec, _MORAL_KEYS)

    if not character:
        character = _regex(prompt.lower(), r"character[:\-]\s*([^\n.,]{1,60})")
    if not moral:
        moral = _regex(prompt.lower(), r"moral[:\-]\s*([^\n.]{1,80})")
    return character, moral


def _story_of(rec: dict) -> str:
    return (rec.get("fable") or rec.get("story") or rec.get("output")
            or rec.get("text") or "").strip()


def format_sample(rec: dict) -> str:
    character, moral = parse_elements(rec)
    return _PREFIX.format(character=character, moral=moral, story=_story_of(rec))


def rec_is_valid(rec: dict) -> bool:
    # keep only English fables when language is present
    lang = rec.get("language")
    if isinstance(lang, str) and lang.lower() != "en":
        return False
    character, moral = parse_elements(rec)
    story = _story_of(rec)
    return bool(character) and bool(moral) and len(story) >= 80


def iter_tf1(n: int, seed: int = 42):
    """Yield up to `n` valid fable records from the HF dataset (streaming)."""
    from datasets import load_dataset

    ds = load_dataset(TF1_DATASET, split="train", streaming=True)
    rng = random.Random(seed)
    pool: list[dict] = []
    for row in ds:
        pool.append(row)
        if len(pool) >= n * 4:  # oversample before filtering
            break
    rng.shuffle(pool)
    yielded = 0
    for row in pool:
        if rec_is_valid(row):
            yield row
            yielded += 1
            if yielded >= n:
                return


def iter_local(path: str | Path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


# --------------------------------------------------------------------------
# BPE mode (default) — transformers / tokenizers
# --------------------------------------------------------------------------

def build_tokenizer_bpe(texts: list[str], path: str | Path, vocab_size: int = 8192):
    """Train a BPE tokenizer via `tokenizers` and save tokenizer.json."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<char>", "</char>",
                                                                 "<moral>", "</moral>",
                                                                 "<story>", "</story>"])
    tok.train_from_iterator(texts, trainer)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def prepare_bpe(records: list[dict], out_dir: str | Path, vocab_size: int = 8192) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [format_sample(r) for r in records if rec_is_valid(r)]
    if not texts:
        raise ValueError("No valid fable records found.")

    build_tokenizer_bpe(texts, out_dir / "tokenizer.json", vocab_size=vocab_size)
    _write_jsonl(out_dir / "fables.jsonl", texts)
    meta = {"mode": "bpe", "vocab_size": vocab_size, "n_samples": len(texts)}
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------
# Char mode (offline fallback / local tests) — nanoGPT-style bins
# --------------------------------------------------------------------------

def prepare_char(records: list[dict], out_dir: str | Path,
                 val_frac: float = 0.05, seed: int = 42) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = [format_sample(r) for r in records if rec_is_valid(r)]
    if not texts:
        raise ValueError("No valid fable records found.")

    tok = FableTokenizer()
    tok.train(texts)

    rng = random.Random(seed)
    rng.shuffle(texts)  # mix themes; within-sample contiguity preserved
    flat: list[int] = []
    sep = tok.encode("\n")
    for t in texts:
        flat.extend(tok.encode(t))
        flat.extend(sep)

    n_val = int(len(flat) * val_frac)
    _write_bin(out_dir / "val.bin", flat[:n_val])
    _write_bin(out_dir / "train.bin", flat[n_val:])
    tok.save(out_dir / "vocab.json")
    meta = {"mode": "char", "vocab_size": tok.vocab_size,
            "n_train": len(flat) - n_val, "n_val": n_val, "n_samples": len(texts)}
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _write_bin(path: Path, ids: list[int]) -> None:
    with path.open("wb") as f:
        for i in ids:
            f.write(struct.pack("<H", i))


def _write_jsonl(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hf", help="hf | path/to/fables.jsonl")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--mode", choices=["bpe", "char"], default="bpe")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--out", default="data/fable200m")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = (list(iter_tf1(args.n, seed=args.seed))
               if args.source == "hf" else list(iter_local(args.source)))

    if args.mode == "bpe":
        meta = prepare_bpe(records, args.out, vocab_size=args.vocab_size)
    else:
        meta = prepare_char(records, args.out, val_frac=args.val_frac, seed=args.seed)

    print("Wrote dataset to", args.out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

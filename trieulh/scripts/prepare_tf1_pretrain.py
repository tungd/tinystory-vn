"""Prepare a conditional pretraining corpus from klusai/ds-tf1-en-3m.

Streams the dataset, parses the 5 scaffold slots, applies slot dropout,
formats train text with the app's prompt format, dedups by prompt_hash,
and writes JSONL. Training deps (datasets) only needed when actually
streaming; build_record itself is pure and unit-tested.

v3 (2026-07-11): two data interventions from the qualitative eval:
- --cap-phrase/--cap-frac: cap the fraction of examples whose FABLE contains a
  template phrase ("wise old owl" is in 28% of TF1 but sampling amplifies it
  to ~90% of generations); capping the mode at ~10% attacks template collapse.
- --slot-dropout k=v: per-slot dropout override (teaching=0.15 outcome=0.15)
  so the model sees the requested moral/outcome in conditioning ~85% of the
  time and learns to FOLLOW it instead of substituting a generic one.
"""
import argparse
import json
import random
from pathlib import Path

from trieulh.scripts.tf1_pretrain.parse import parse_slots
from trieulh.scripts.tf1_pretrain.format import (
    apply_dropout, build_training_text, length_bucket,
)


def build_record(row: dict, rng: random.Random,
                 min_words: int = 0, max_words: int | None = None,
                 dropout_overrides: dict | None = None) -> dict | None:
    fable = (row.get("fable") or "").strip()
    if not fable:
        return None
    wc = len(fable.split())
    if wc < min_words:
        return None
    if max_words is not None and wc > max_words:
        return None
    slots = parse_slots(row.get("prompt") or "")
    slots = apply_dropout(slots, rng, p_overrides=dropout_overrides)
    length = length_bucket(wc)
    text, cond_len = build_training_text(slots, fable, length)
    return {"text": text, "cond_len": cond_len}


def _write_split(rows, out_path: Path, n: int, seed: int,
                 min_words: int = 0, max_words: int | None = None,
                 cap_phrase: str | None = None, cap_frac: float = 0.10,
                 dropout_overrides: dict | None = None) -> int:
    rng = random.Random(seed)
    seen: set[str] = set()
    written = 0
    phrase_written = 0
    phrase = cap_phrase.lower() if cap_phrase else None
    with out_path.open("w") as f:
        for row in rows:
            h = row.get("prompt_hash")
            if h in seen:
                continue
            seen.add(h)
            if phrase and phrase in (row.get("fable") or "").lower():
                # skip once the phrase quota is used up (keeps its share <= cap_frac)
                if phrase_written >= cap_frac * (written + 1):
                    continue
                has_phrase = True
            else:
                has_phrase = False
            rec = build_record(row, rng, min_words, max_words, dropout_overrides)
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if has_phrase:
                phrase_written += 1
            if written >= n:
                break
    return written


def _parse_slot_dropout(pairs: list[str]) -> dict:
    """Parse repeated k=v args, e.g. ["teaching=0.15","outcome=0.15"]."""
    out: dict[str, float] = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k.strip()] = float(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=600_000)
    ap.add_argument("--test-n", type=int, default=500)
    ap.add_argument("--out", default="data/tf1")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--min-words", type=int, default=60)
    ap.add_argument("--max-words", type=int, default=320)
    ap.add_argument("--cap-phrase", default=None,
                    help='Template phrase to cap in fables, e.g. "wise old owl".')
    ap.add_argument("--cap-frac", type=float, default=0.10,
                    help="Max fraction of examples containing --cap-phrase.")
    ap.add_argument("--slot-dropout", nargs="*", default=[],
                    help="Per-slot dropout overrides, e.g. teaching=0.15 outcome=0.15.")
    args = ap.parse_args(argv)
    overrides = _parse_slot_dropout(args.slot_dropout) or None

    from datasets import load_dataset  # imported lazily (training dep)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = load_dataset("klusai/ds-tf1-en-3m", split="train", streaming=True)
    n_tr = _write_split(train, out / "train.jsonl", args.train_n, args.seed,
                        args.min_words, args.max_words,
                        args.cap_phrase, args.cap_frac, overrides)
    test = load_dataset("klusai/ds-tf1-en-3m", split="test", streaming=True)
    n_te = _write_split(test, out / "test.jsonl", args.test_n, args.seed + 1,
                        args.min_words, args.max_words,
                        None, args.cap_frac, overrides)   # test: no cap (keep the real distribution)
    print(f"wrote train={n_tr} test={n_te} to {out}")


if __name__ == "__main__":
    main()

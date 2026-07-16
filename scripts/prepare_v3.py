"""Prepare the isolated v3 continuation dataset from exact v2 TF1 pairs.

v3 keeps the v2 control prefix and tokenizer, but makes the requested moral an
explicit generation target:

    <char> exact character </char>
    <moral> exact moral </moral>
    <story>
    original story

    Moral: exact moral
    </story>

The output stores prompt and target separately so train_v3.py can mask prompt
tokens from loss.
"""

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_tf1 import _story_of, parse_elements


PREFIX = (
    "<char> {character} </char>\n"
    "<moral> {moral} </moral>\n"
    "<story>\n"
)


def parse_formatted_v2(text: str) -> tuple[str, str, str]:
    """Recover exact controls/story from a v1/v2 formatted training sample."""
    match = re.fullmatch(
        r"<char>\s*(.*?)\s*</char>\n"
        r"<moral>\s*(.*?)\s*</moral>\n"
        r"<story>\n(.*?)\n</story>",
        text.strip(),
        flags=re.DOTALL,
    )
    if not match:
        return "", "", ""
    return tuple(part.strip() for part in match.groups())


def extract_fields(record: dict | str) -> tuple[str, str, str]:
    if isinstance(record, str):
        return parse_formatted_v2(record)
    character, moral = parse_elements(record)
    return character, moral, _story_of(record)


def character_headword(character: str) -> str:
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", character.lower())
    return words[-1] if words else ""


def story_mentions_character(character: str, story: str) -> bool:
    headword = character_headword(character)
    if not headword:
        return False
    variants = {headword}
    if headword.endswith("y") and len(headword) > 1:
        variants.add(headword[:-1] + "ies")
    elif headword.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(headword + "es")
    elif not headword.endswith("s"):
        variants.add(headword + "s")
    return any(re.search(rf"\b{re.escape(word)}\b", story, re.IGNORECASE) for word in variants)


def build_example(record: dict | str) -> dict | None:
    character, moral, story = extract_fields(record)
    if not character or not moral or len(story) < 80:
        return None
    if not story_mentions_character(character, story):
        return None

    prompt = PREFIX.format(character=character, moral=moral)
    target = f"{story.rstrip()}\n\nMoral: {moral}\n</story>"
    return {
        "character": character,
        "moral": moral,
        "prompt": prompt,
        "target": target,
    }


def load_records(path: str | Path) -> Iterable[dict | str]:
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_v3(
    records: Iterable[dict | str],
    out_dir: str | Path,
    tokenizer_source: str | Path,
    *,
    validation_fraction: float = 0.05,
    seed: int = 42,
    limit: int | None = None,
) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    accepted: list[dict] = []
    seen = 0
    for record in records:
        seen += 1
        example = build_example(record)
        if example is not None:
            accepted.append(example)
            if limit is not None and len(accepted) >= limit:
                break
    if len(accepted) < 2:
        raise ValueError("Need at least two valid v3 examples")

    random.Random(seed).shuffle(accepted)
    validation_count = max(1, round(len(accepted) * validation_fraction))
    validation = accepted[:validation_count]
    train = accepted[validation_count:]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "validation.jsonl", validation)
    shutil.copy2(tokenizer_source, out / "tokenizer.json")

    meta = {
        "version": "v3",
        "source_records_seen": seen,
        "accepted": len(accepted),
        "rejected": seen - len(accepted),
        "train": len(train),
        "validation": len(validation),
        "validation_fraction": validation_fraction,
        "seed": seed,
        "format": "exact controls + original story + explicit moral",
        "loss": "target only; prompt masked",
        "tokenizer_source": str(tokenizer_source),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/v2/data/fables.jsonl")
    parser.add_argument("--tokenizer", default="runs/v2/artifacts/hf/tokenizer.json")
    parser.add_argument("--out", default="runs/v3/data")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    meta = prepare_v3(
        load_records(args.source),
        args.out,
        args.tokenizer,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

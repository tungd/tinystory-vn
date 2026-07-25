"""Clean raw story data and format instruction records for fine-tuning."""

import argparse
import json
import re
from pathlib import Path

from app.prompt import build_instruction


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_records(
    raw: list[dict],
    refusals: list[dict],
    max_chars: int | None = None,
    min_chars: int = 0,
) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        story = clean_text(item["story"])
        if not story or story in seen:
            continue
        if min_chars and len(story) < min_chars:
            continue
        if max_chars is not None and len(story) > max_chars:
            continue
        seen.add(story)
        records.append(
            {
                "type": "story",
                "instruction": build_instruction(item["topic"], item["moral"], item["age_range"]),
                "output": story,
            }
        )
    for item in refusals:
        records.append(
            {
                "type": "refusal",
                "instruction": clean_text(item["instruction"]),
                "output": clean_text(item["output"]),
            }
        )
    return records


def split_records(records: list[dict], seed: int) -> dict:
    import random

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, n // 10) if n >= 10 else 0
    n_test = max(1, n // 10) if n >= 10 else 0
    val = shuffled[:n_val]
    test = shuffled[n_val:n_val + n_test]
    train = shuffled[n_val + n_test:]
    return {"train": train, "val": val, "test": test}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/stories.jsonl")
    parser.add_argument("--refusals", default="data/refusal/refusals.jsonl")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--min-chars", type=int, default=0)
    args = parser.parse_args()

    records = build_records(
        _read_jsonl(Path(args.raw)),
        _read_jsonl(Path(args.refusals)),
        max_chars=args.max_chars,
        min_chars=args.min_chars,
    )
    splits = split_records(records, seed=args.seed)
    for name, rows in splits.items():
        _write_jsonl(Path(args.out) / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)} records")


if __name__ == "__main__":
    main()

"""Prepare English fable SFT datasets from TF1-style records.

The script can read either:

- Hugging Face dataset `klusai/ds-tf1-en-3m` with `--source hf`
- a local JSONL file with `--source-jsonl path/to/file.jsonl`

It writes split folders for several dataset sizes so the project can compare
training approaches with the same data budget, for example SFT-100 vs SFT-500.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


DEFAULT_INSTRUCTION = "Write a short English fable for children with a clear moral."
DEFAULT_DATASET = "klusai/ds-tf1-en-3m"

STORY_KEYS = ("output", "story", "fable", "text", "response", "completion")
MORAL_KEYS = ("teaching", "moral", "lesson", "moral_lesson")
CHARACTER_KEYS = ("character", "main_character", "protagonist")
SETTING_KEYS = ("setting", "place", "location")
CHALLENGE_KEYS = ("challenge", "problem", "conflict")
OUTCOME_KEYS = ("outcome", "resolution", "ending")

PROMPT_LABELS = {
    "character": ("Main Character", "Character"),
    "setting": ("Setting",),
    "challenge": ("Challenge",),
    "outcome": ("Outcome",),
    "teaching": ("Teaching", "Moral"),
}

CHILD_UNSAFE_PATTERNS = (
    r"\bfall(?:s|ing)? in love\b",
    r"\blove triangle\b",
    r"\bromance\b",
    r"\bromantic\b",
    r"\bdate\b",
    r"\bdating\b",
    r"\bmarry\b",
    r"\bmarriage\b",
    r"\bbetrayal\b",
    r"\bbetray\b",
    r"\bkill\b",
    r"\bblood\b",
    r"\bweapon\b",
    r"\bwar\b",
    r"\bvicious\b",
    r"\bpoisonous\b",
    r"\babandoned\b",
    r"\bdeserted\b",
    r"\bruined\b",
    r"\bfortress\b",
    r"\bmine\b",
    r"\bscapegoat\w*\b",
    r"\bhidden agenda\b",
    r"\binjustice\b",
    r"\boppression\b",
    r"\brebellion\b",
    r"\bidentity crisis\b",
    r"\bprophecy\b",
    r"\bethical dilemma\b",
    r"\bmoral compromise\b",
    r"\brenounces violence\b",
    r"\bviolence\b",
    r"\bdeceiver\b",
    r"\bdeceived\b",
    r"\bdeception\b",
    r"\bspy\b",
    r"\bsabotage\b",
    r"\bbounty\b",
    r"\bmanipulative\b",
    r"\bancient enemies\b",
    r"\bsworn enemy\b",
    r"\bvillain\b",
    r"\bbanished\b",
    r"\bruthless\b",
    r"\bharm\b",
    r"\bfight\b",
    r"\bfighting\b",
    r"\bbattle of wits\b",
    r"\bvictim\b",
    r"\brebellious\b",
    r"\bforced to confront\b",
    r"\bhaunted\b",
    r"\bseethed\b",
    r"\bchallenge to authority\b",
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip()


def first_text(record: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return clean_text(value)
    return ""


def nested_first_text(record: dict, keys: tuple[str, ...]) -> str:
    value = first_text(record, keys)
    if value:
        return value
    for parent in ("input", "prompt", "metadata", "template", "narrative", "story_elements"):
        child = record.get(parent)
        if isinstance(child, dict):
            value = first_text(child, keys)
            if value:
                return value
    return ""


def parse_prompt_elements(prompt: str) -> dict[str, str]:
    elements: dict[str, str] = {}
    for key, labels in PROMPT_LABELS.items():
        for label in labels:
            pattern = rf"^\s*-\s*{re.escape(label)}\s*:\s*(.+?)\s*$"
            match = re.search(pattern, prompt, flags=re.I | re.M)
            if match:
                elements[key] = clean_text(match.group(1))
                break
    return elements


def infer_moral_from_story(story: str) -> str:
    match = re.search(r"(?:^|\n)\s*(?:Moral|Lesson|Teaching)\s*:\s*(.+)$", story, re.I)
    if match:
        return clean_text(match.group(1))
    sentences = re.split(r"(?<=[.!?])\s+", story)
    if sentences:
        last = clean_text(sentences[-1])
        if len(last.split()) <= 20 and re.search(r"\b(learn|kind|share|honest|patient|wise|friend|help)\w*\b", last, re.I):
            return last.removeprefix("Moral:").strip()
    return ""


def build_input(record: dict, story: str) -> str:
    prompt_elements = parse_prompt_elements(str(record.get("prompt", "")))
    character = nested_first_text(record, CHARACTER_KEYS) or prompt_elements.get("character", "")
    setting = nested_first_text(record, SETTING_KEYS) or prompt_elements.get("setting", "")
    challenge = nested_first_text(record, CHALLENGE_KEYS) or prompt_elements.get("challenge", "")
    outcome = nested_first_text(record, OUTCOME_KEYS) or prompt_elements.get("outcome", "")
    teaching = nested_first_text(record, MORAL_KEYS) or prompt_elements.get("teaching", "") or infer_moral_from_story(story)

    lines = []
    if character:
        lines.append(f"Character: {character}")
    if setting:
        lines.append(f"Setting: {setting}")
    if challenge:
        lines.append(f"Challenge: {challenge}")
    if outcome:
        lines.append(f"Outcome: {outcome}")
    if teaching:
        lines.append(f"Teaching: {teaching}")
    if not lines:
        lines.append("Teaching: include a clear moral at the end")
    return "\n".join(lines)


def format_record(record: dict) -> dict | None:
    story = first_text(record, STORY_KEYS)
    if not story:
        story = nested_first_text(record, STORY_KEYS)
    if not story:
        return None
    story = clean_text(story.replace("**", ""))
    story = re.sub(r"\bThe moral\s*:", "Moral:", story, flags=re.I)
    teaching = nested_first_text(record, MORAL_KEYS) or parse_prompt_elements(str(record.get("prompt", ""))).get("teaching", "")
    if teaching and "moral:" not in story.lower():
        story = f"{story.rstrip()} Moral: {teaching}"
    return {
        "instruction": DEFAULT_INSTRUCTION,
        "input": build_input(record, story),
        "output": story,
    }


def passes_filter(item: dict, min_words: int, max_words: int) -> bool:
    story = item["output"]
    combined = f"{item.get('input', '')} {story}".lower()
    if any(re.search(pattern, combined) for pattern in CHILD_UNSAFE_PATTERNS):
        return False
    words = re.findall(r"[A-Za-z']+", story)
    if len(words) < min_words or len(words) > max_words:
        return False
    if "Teaching:" not in item.get("input", "") and "moral:" not in story.lower():
        return False
    return True


def dedupe_key(item: dict) -> str:
    normalized = clean_text(item["input"] + "\n" + item["output"]).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def read_local_jsonl(path: Path) -> Iterator[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def read_hf_dataset(dataset_name: str, split: str) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with: "
            ".\\.venv\\Scripts\\python.exe -m pip install datasets"
        ) from exc
    return load_dataset(dataset_name, split=split, streaming=True)


def collect_records(
    source: Iterable[dict],
    target_n: int,
    min_words: int,
    max_words: int,
) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for raw in source:
        item = format_record(raw)
        if not item or not passes_filter(item, min_words=min_words, max_words=max_words):
            continue
        key = dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        rows.append(item)
        if len(rows) >= target_n:
            break
    return rows


def split_records(rows: list[dict], valid_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * valid_ratio)) if len(shuffled) >= 10 else max(1, len(shuffled) // 5)
    valid = shuffled[:n_valid]
    train = shuffled[n_valid:]
    return train, valid


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset(out_dir: Path, rows: list[dict], valid_ratio: float, seed: int) -> None:
    train, valid = split_records(rows, valid_ratio=valid_ratio, seed=seed)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    manifest = {
        "total": len(rows),
        "train": len(train),
        "valid": len(valid),
        "instruction": DEFAULT_INSTRUCTION,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_sizes(value: str) -> list[int]:
    sizes = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not sizes or any(size <= 1 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers greater than 1")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["hf", "local"], default="hf")
    parser.add_argument("--source-jsonl", default="")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("100,500,2000"))
    parser.add_argument("--out", default="data/tf1")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=60)
    parser.add_argument("--max-words", type=int, default=450)
    parser.add_argument("--promote-size", type=int, default=0, help="Copy this size to data/train.jsonl and data/valid.jsonl.")
    args = parser.parse_args()

    max_size = max(args.sizes)
    if args.source == "local":
        if not args.source_jsonl:
            raise SystemExit("--source-jsonl is required when --source local")
        source = read_local_jsonl(Path(args.source_jsonl))
    else:
        source = read_hf_dataset(args.dataset, args.split)

    rows = collect_records(source, target_n=max_size, min_words=args.min_words, max_words=args.max_words)
    if len(rows) < max_size:
        raise SystemExit(f"Only collected {len(rows)} usable rows, expected {max_size}.")

    root = Path(args.out)
    for size in args.sizes:
        out_dir = root / f"sft_{size}"
        write_dataset(out_dir, rows[:size], valid_ratio=args.valid_ratio, seed=args.seed)
        print(f"Wrote {out_dir}: {size} rows")

    if args.promote_size:
        if args.promote_size not in args.sizes:
            raise SystemExit("--promote-size must be one of --sizes")
        train = root / f"sft_{args.promote_size}" / "train.jsonl"
        valid = root / f"sft_{args.promote_size}" / "valid.jsonl"
        Path("data/train.jsonl").write_text(train.read_text(encoding="utf-8"), encoding="utf-8")
        Path("data/valid.jsonl").write_text(valid.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Promoted sft_{args.promote_size} to data/train.jsonl and data/valid.jsonl")


if __name__ == "__main__":
    main()

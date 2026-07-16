"""Prepare a wider, cleaner TF1 continuation dataset for v4.

v4 starts after the v2 source slice and the 100 controls used for v3
evaluation. It keeps exact character/moral supervision while selecting new
setting/challenge/outcome combinations. Targets have one canonical moral footer
and no Markdown formatting debris.
"""

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_tf1 import TF1_DATASET, _story_of, parse_elements, rec_is_valid
from scripts.prepare_v3 import PREFIX, story_mentions_exact_character


MORAL_STOPWORDS = {
    "a", "an", "and", "are", "be", "being", "can", "for", "in", "is",
    "it", "its", "of", "on", "or", "over", "that", "the", "this", "to",
    "with",
}
SCAFFOLD_FIELDS = ("setting", "challenge", "outcome")


def parse_scaffold(record: dict) -> dict[str, str]:
    prompt = record.get("prompt") or ""
    result = {}
    for field in SCAFFOLD_FIELDS:
        match = re.search(rf"(?im)^\s*-\s*{field}:\s*(.+?)\s*$", prompt)
        result[field] = html.unescape(match.group(1).strip()) if match else ""
    return result


def clean_story(story: str, moral: str) -> str:
    """Remove Markdown wrappers and any trailing source moral footer."""
    story = html.unescape(story).replace("\r\n", "\n").replace("\r", "\n").strip()
    story = re.sub(r"(?m)^\s*```[^\n]*$", "", story)
    story = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", story)
    story = story.replace("**", "").replace("__", "")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", story) if part.strip()]
    normalized_moral = re.sub(r"[^a-z0-9]+", " ", moral.casefold()).strip()
    while paragraphs:
        normalized = re.sub(r"[^a-z0-9]+", " ", paragraphs[-1].casefold()).strip()
        is_moral_label = bool(re.match(
            r"^(?:the\s+)?moral(?:\s+is(?:\s+clear)?|\s+of\s+[^:]+)?\s*:",
            paragraphs[-1],
            re.I,
        ))
        is_lesson_label = bool(re.match(r"^lesson\s*:", paragraphs[-1], re.I))
        is_meta_note = bool(re.match(
            r"^\(?\s*(?:note|word count|age group)\s*:|^this story\b",
            paragraphs[-1],
            re.I,
        ))
        if is_moral_label or is_lesson_label or is_meta_note or normalized == normalized_moral:
            paragraphs.pop()
        else:
            break
    story = "\n\n".join(paragraphs).strip()
    story = re.sub(r"(?i)\bthe moral\s+is\s+clear\s*:", "the lesson was clear:", story)
    story = re.sub(r"(?i)\bthe moral\s*:", "the lesson that", story)
    story = re.sub(r"(?i)\bmoral\s*:", "lesson:", story)
    return story


def moral_terms(moral: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z]+(?:'[a-z]+)?", moral.casefold())
        if word not in MORAL_STOPWORDS and len(word) >= 3
    }


def tail_supports_moral(story: str, moral: str, tail_words: int = 100) -> bool:
    tail = " ".join(story.casefold().split()[-tail_words:])
    return any(
        re.search(rf"\b{re.escape(term)}\b", tail)
        for term in moral_terms(moral)
    )


def screen_record(
    record: dict,
    *,
    source_index: int | None = None,
    min_words: int = 160,
    max_words: int = 380,
) -> tuple[dict | None, str]:
    character, moral = parse_elements(record)
    raw_story = _story_of(record)
    if not character or not moral or not raw_story:
        return None, "missing_fields"

    story = clean_story(raw_story, moral)
    word_count = len(story.split())
    if not min_words <= word_count <= max_words:
        return None, "length"
    opening = " ".join(story.split()[:120])
    if not story_mentions_exact_character(character, opening):
        return None, "character_not_in_opening"
    if not tail_supports_moral(story, moral):
        return None, "moral_not_supported_near_end"

    scaffold = parse_scaffold(record)
    source = (
        f"{TF1_DATASET}:train-valid-index-{source_index}"
        if source_index is not None else record.get("prompt_hash", "")
    )
    example = {
        "character": character,
        "moral": moral,
        **scaffold,
        "source": source,
        "prompt_hash": record.get("prompt_hash", ""),
        "prompt": PREFIX.format(character=character, moral=moral),
        "target": f"{story}\n\nMoral: {moral}\n</story>",
    }
    return example, "accepted"


def build_example(record: dict, **kwargs) -> dict | None:
    return screen_record(record, **kwargs)[0]


def load_hf_records() -> Iterable[dict]:
    from datasets import load_dataset

    return load_dataset(TF1_DATASET, split="train", streaming=True)


def load_local_records(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _split_is_validation(key: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < fraction


def _control(record: dict, valid_index: int) -> dict:
    character, moral = parse_elements(record)
    return {
        "character": character,
        "moral": moral,
        "prompt": PREFIX.format(character=character, moral=moral),
        "reference_story": _story_of(record),
        "source": f"{TF1_DATASET}:train-valid-index-{valid_index}",
    }


def prepare_v4(
    records: Iterable[dict],
    out_dir: str | Path,
    tokenizer_source: str | Path,
    *,
    skip_valid: int = 200_100,
    limit: int = 250_000,
    validation_fraction: float = 0.02,
    eval_controls: int = 100,
    seed: int = 42,
) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if limit < 2:
        raise ValueError("limit must be at least two")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_paths = [out / name for name in ("train.jsonl", "validation.jsonl", "meta.json")]
    if any(path.exists() for path in final_paths):
        raise FileExistsError(f"Refusing existing v4 dataset in {out}")
    train_tmp = out / "train.jsonl.tmp"
    validation_tmp = out / "validation.jsonl.tmp"
    rejection_counts: Counter[str] = Counter()
    diversity = {field: set() for field in ("character", "moral", *SCAFFOLD_FIELDS)}
    seen_keys: set[str] = set()
    accepted = train_count = validation_count = 0
    valid_index = -1
    training_end_index = None
    controls: list[dict] = []

    try:
        with train_tmp.open("w", encoding="utf-8") as train_out, validation_tmp.open(
            "w", encoding="utf-8"
        ) as validation_out:
            for record in records:
                if not rec_is_valid(record):
                    continue
                valid_index += 1
                if valid_index < skip_valid:
                    if valid_index and valid_index % 25_000 == 0:
                        print(f"skipped {valid_index}/{skip_valid} prior valid rows", flush=True)
                    continue

                if accepted >= limit:
                    if training_end_index is None:
                        training_end_index = valid_index - 1
                    if not eval_controls:
                        break
                    controls.append(_control(record, valid_index))
                    if len(controls) >= eval_controls:
                        break
                    continue

                example, reason = screen_record(record, source_index=valid_index)
                if example is None:
                    rejection_counts[reason] += 1
                    continue
                key = example["prompt_hash"] or hashlib.sha256(
                    example["target"].encode()
                ).hexdigest()
                if key in seen_keys:
                    rejection_counts["duplicate"] += 1
                    continue
                seen_keys.add(key)

                line = json.dumps(example, ensure_ascii=False) + "\n"
                if _split_is_validation(key, seed, validation_fraction):
                    validation_out.write(line)
                    validation_count += 1
                else:
                    train_out.write(line)
                    train_count += 1
                accepted += 1
                if accepted % 25_000 == 0:
                    print(
                        f"accepted {accepted}/{limit}; valid index {valid_index}; "
                        f"rejected {sum(rejection_counts.values())}",
                        flush=True,
                    )
                for field in diversity:
                    if example[field]:
                        diversity[field].add(example[field].casefold())

        if accepted != limit:
            raise ValueError(f"Only accepted {accepted} of requested {limit} examples")
        if eval_controls and len(controls) != eval_controls:
            raise ValueError(f"Only collected {len(controls)} of {eval_controls} controls")
        if not train_count or not validation_count:
            raise ValueError("Need non-empty train and validation splits")

        train_tmp.replace(out / "train.jsonl")
        validation_tmp.replace(out / "validation.jsonl")
        shutil.copy2(tokenizer_source, out / "tokenizer.json")
        (out / "eval_controls.json").write_text(
            json.dumps(controls, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception:
        train_tmp.unlink(missing_ok=True)
        validation_tmp.unlink(missing_ok=True)
        raise

    meta = {
        "version": "v4",
        "dataset": TF1_DATASET,
        "skip_valid": skip_valid,
        "training_end_valid_index": training_end_index,
        "source_valid_rows_scanned": training_end_index - skip_valid + 1,
        "accepted": accepted,
        "rejected": sum(rejection_counts.values()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "train": train_count,
        "validation": validation_count,
        "validation_fraction": validation_fraction,
        "eval_controls": len(controls),
        "eval_start_valid_index": (
            int(controls[0]["source"].rsplit("-", 1)[-1]) if controls else None
        ),
        "seed": seed,
        "diversity": {f"unique_{field}": len(values) for field, values in diversity.items()},
        "format": "exact controls + cleaned source story + one canonical moral",
        "filters": [
            "160-380 words after cleanup",
            "exact character phrase in first 120 words",
            "moral content word in final 100 words before canonical footer",
            "unique prompt hash",
        ],
        "loss": "target only; prompt masked",
        "tokenizer_source": str(tokenizer_source),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="hf", help="hf or local raw TF1 JSONL")
    parser.add_argument("--tokenizer", default="runs/v2/artifacts/hf/tokenizer.json")
    parser.add_argument("--out", default="runs/v4/data")
    parser.add_argument("--skip-valid", type=int, default=200_100)
    parser.add_argument("--limit", type=int, default=250_000)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--eval-controls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_hf_records() if args.source == "hf" else load_local_records(args.source)
    meta = prepare_v4(
        records,
        args.out,
        args.tokenizer,
        skip_valid=args.skip_valid,
        limit=args.limit,
        validation_fraction=args.validation_fraction,
        eval_controls=args.eval_controls,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

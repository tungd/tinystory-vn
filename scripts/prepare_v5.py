"""Build a small quality-weighted v5 continuation set.

Accepted human-authored stories stay unrewritten. Their exact protagonist phrase
and Gemma-inferred causal moral become controls. A small v3 replay slice limits
catastrophic forgetting; repeated real stories give the quality set most weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.label_v5 import latest_by_source, load_jsonl
from scripts.prepare_v3 import PREFIX, story_mentions_exact_character
from scripts.v5_sources import clean_public_story


NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
}
TRAIT_BLOCKLIST = {"misunderstanding"}
CONTENT_BLOCKLIST = re.compile(r"\b(?:Hottentots?|Bushmen)\b", re.IGNORECASE)
MIN_STORY_WORDS = 70
MODERN_MIN_STORY_WORDS = 50
MAX_STORY_WORDS = 250


def _validation(source: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{source}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def inject_character(story: str, anchor: str, trait: str) -> tuple[str, str] | None:
    """Insert one audited trait at the first exact protagonist mention."""
    match = re.search(
        rf"(?<![\w'-]){re.escape(anchor)}(?![\w'-])",
        story,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    original = story[match.start() : match.end()]
    words = original.split()
    if trait.casefold() in {word.casefold() for word in words}:
        character = original
    elif words and words[0].casefold() in {"a", "an", "the"}:
        insert_at = 2 if len(words) > 1 and words[1].casefold() in NUMBER_WORDS else 1
        if words[0].casefold() in {"a", "an"}:
            article = "an" if trait[0].casefold() in "aeiou" else "a"
            words[0] = article.capitalize() if words[0][0].isupper() else article
        words.insert(insert_at, trait)
        character = " ".join(words)
    else:
        character = f"{trait} {original}"
    modified = story[: match.start()] + character + story[match.end() :]
    return character, modified


def build_real_example(row: dict) -> dict | None:
    annotation = row.get("annotation", {})
    if not annotation.get("accepted") or row.get("source_split") == "external_holdout":
        return None
    anchor = " ".join(annotation.get("protagonist_anchor", "").split())
    trait = annotation.get("trait", "").strip().casefold()
    moral = " ".join(annotation.get("moral", "").split())
    story = clean_public_story(row.get("story", ""))
    word_count = len(story.split())
    min_story_words = (
        MODERN_MIN_STORY_WORDS
        if row.get("collection") == "Understanding Fables"
        else MIN_STORY_WORDS
    )
    if (
        not anchor
        or not trait
        or trait in TRAIT_BLOCKLIST
        or not moral
        or not story
        or CONTENT_BLOCKLIST.search(story)
        or not min_story_words <= word_count <= MAX_STORY_WORDS
    ):
        return None
    injected = inject_character(story, anchor, trait)
    if injected is None:
        return None
    character, story = injected
    if not story_mentions_exact_character(character, story):
        raise AssertionError("injected v5 character must appear exactly")
    return {
        "character": character,
        "moral": moral,
        "source": row["source"],
        "source_type": "human-authored",
        "original_protagonist_anchor": anchor,
        "demonstrated_trait": trait,
        "collection": row["collection"],
        "title": row["title"],
        "story_words": word_count,
        "prompt": PREFIX.format(character=character, moral=moral),
        "target": f"{story}\n\nMoral: {moral}\n</story>",
    }


def build_external_control(row: dict) -> dict | None:
    if row.get("source_split") != "external_holdout":
        return None
    training_shape = {**row, "source_split": "train"}
    example = build_real_example(training_shape)
    if example is None:
        return None
    story = example["target"].rsplit("\n\nMoral:", 1)[0]
    return {
        "character": example["character"],
        "moral": example["moral"],
        "prompt": example["prompt"],
        "reference_story": story,
        "source": example["source"],
    }


def _replay_source(row: dict) -> str:
    return "v3-replay:" + hashlib.sha256(row["target"].encode()).hexdigest()[:16]


def prepare_v5(
    annotations: list[dict],
    replay_rows: list[dict],
    out_dir: str | Path,
    tokenizer_source: str | Path,
    eval_controls_source: str | Path,
    *,
    validation_fraction: float = 0.1,
    real_repeats: int = 3,
    replay_ratio: float = 1.0,
    seed: int = 42,
) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    if real_repeats < 1 or replay_ratio < 0:
        raise ValueError("invalid v5 mixing weights")

    latest = latest_by_source(annotations)
    real = [example for row in latest.values() if (example := build_real_example(row))]
    external_controls = [
        control for row in latest.values() if (control := build_external_control(row))
    ]
    if len(real) < 10:
        raise ValueError("Need at least ten accepted public-domain stories")
    real.sort(key=lambda row: row["source"])
    validation = [row for row in real if _validation(row["source"], seed, validation_fraction)]
    train_real = [row for row in real if row not in validation]
    if not validation:
        validation = [train_real.pop()]

    rng = random.Random(seed)
    eligible_replay = [
        row
        for row in replay_rows
        if row["target"].count("Moral:") == 1
        and row["target"].endswith(f"Moral: {row['moral']}\n</story>")
    ]
    replay_count = min(len(eligible_replay), round(len(train_real) * replay_ratio))
    replay = rng.sample(eligible_replay, replay_count)
    replay = [
        {
            **row,
            "source": _replay_source(row),
            "source_type": "v3-replay",
        }
        for row in replay
    ]
    train = [dict(row) for row in train_real for _ in range(real_repeats)] + replay
    rng.shuffle(train)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = [
        out / name
        for name in ("train.jsonl", "validation.jsonl", "external_controls.json", "meta.json")
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Refusing existing v5 dataset in {out}")
    for path, rows in ((out / "train.jsonl", train), (out / "validation.jsonl", validation)):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    shutil.copy2(tokenizer_source, out / "tokenizer.json")
    shutil.copy2(eval_controls_source, out / "eval_controls.json")
    (out / "external_controls.json").write_text(
        json.dumps(sorted(external_controls, key=lambda row: row["source"]), indent=2) + "\n",
        encoding="utf-8",
    )

    collections = Counter(row["collection"] for row in real)
    meta = {
        "version": "v5",
        "annotations_latest": len(latest),
        "real_accepted": len(real),
        "real_train_unique": len(train_real),
        "real_repeats": real_repeats,
        "replay": len(replay),
        "train": len(train),
        "validation_real": len(validation),
        "external_controls": len(external_controls),
        "validation_fraction": validation_fraction,
        "collections": dict(sorted(collections.items())),
        "seed": seed,
        "format": "cleaned human story + one deterministic trait insertion + inferred causal moral",
        "loss": "target only; prompt masked",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def restrict_annotations(annotations: list[dict], candidates: list[dict]) -> list[dict]:
    """Keep annotation history only for the current immutable candidate set."""
    sources = {row["source"] for row in candidates}
    restricted = [row for row in annotations if row["source"] in sources]
    latest = latest_by_source(restricted)
    missing = sorted(sources - latest.keys())
    api_errors = sorted(
        source
        for source, row in latest.items()
        if "api_error" in row["annotation"].get("rejection_reasons", [])
    )
    if missing or api_errors:
        raise ValueError(
            f"Incomplete v5 annotations: missing={missing}, api_errors={api_errors}"
        )
    return restricted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="runs/v5/data/annotations.jsonl")
    parser.add_argument("--candidates", default="runs/v5/data/candidates.jsonl")
    parser.add_argument("--replay", default="runs/v3/data/train.jsonl")
    parser.add_argument("--tokenizer", default="runs/v3/data/tokenizer.json")
    parser.add_argument("--eval-controls", default="runs/v4/data/eval_controls.json")
    parser.add_argument("--out", default="runs/v5/data/prepared")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--real-repeats", type=int, default=3)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    annotations = restrict_annotations(
        load_jsonl(args.annotations), load_jsonl(args.candidates)
    )
    meta = prepare_v5(
        annotations,
        load_jsonl(args.replay),
        args.out,
        args.tokenizer,
        args.eval_controls,
        validation_fraction=args.validation_fraction,
        real_repeats=args.real_repeats,
        replay_ratio=args.replay_ratio,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

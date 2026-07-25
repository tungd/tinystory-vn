"""Build a quality-filtered SFT dataset for fluent English fables.

The goal is different from the earlier TF1 subset scripts: this script keeps a
larger but stricter dataset, with filters aimed at fluency, clean structure,
and reliable final morals. It can stream from Hugging Face or read local JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.prepare_tf1 import DEFAULT_INSTRUCTION, clean_text, format_record


DEFAULT_TF1_DATASET = "klusai/ds-tf1-en-3m"
DEFAULT_CHILDREN_DATASET = "garethpaul/children-stories-dataset"

MORAL_LINE_RE = re.compile(r"(?:^|\n)\s*Moral\s*:\s*(.+?)\s*$", re.I)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

BAD_TEXT_PATTERNS = {
    "mojibake": re.compile(r"[âÃ�]"),
    "markdown": re.compile(r"^\s{0,3}[-*#]\s+", re.M),
    "meta_text": re.compile(r"\b(as an ai|i cannot|here is|the story is about)\b", re.I),
    "chapter_title": re.compile(r"^\s*(chapter|title)\s*[:\d]", re.I),
    "explicit_the_moral": re.compile(r"\bthe moral of the story is\s*:", re.I),
}

UNSAFE_PATTERNS = re.compile(
    r"\b("
    r"blood|bloody|kill|killed|killing|die|died|death|murder|weapon|knife|gun|"
    r"war|battle|violence|violent|poison|romance|romantic|dating|marry|marriage|"
    r"betrayal|revenge|curse|haunted|demon|evil|torture|suicide"
    r")\b",
    re.I,
)

WEAK_FLUENCY_PATTERNS = {
    "run_on": re.compile(r"\b(and then|and so|but then)\b.*\b(and then|and so|but then)\b", re.I),
    "repetition": re.compile(r"\b(\w{4,})\b(?:\W+\1\b){2,}", re.I),
    "double_moral": re.compile(r"\bMoral\s*:", re.I),
}

ASCII_REPLACEMENTS = str.maketrans(
    {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\xa0": " ",
    }
)


@dataclass
class Candidate:
    row: dict[str, str]
    source: str
    score: int
    reasons: list[str]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def load_hf_stream(dataset_name: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with: "
            ".\\.venv\\Scripts\\python.exe -m pip install -e .[train]"
        ) from exc
    return load_dataset(dataset_name, split=split, streaming=True)


def load_hf_jsonl_file(dataset_name: str, filename: str, local_dir: Path) -> Iterator[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with: "
            ".\\.venv\\Scripts\\python.exe -m pip install -e .[train]"
        ) from exc
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=dataset_name,
        repo_type="dataset",
        filename=filename,
        local_dir=str(local_dir),
    )
    return read_jsonl(Path(path))


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text.strip()) if s.strip()]


def normalize_output(story: str, teaching: str) -> str:
    story = clean_text(story.replace("**", "").translate(ASCII_REPLACEMENTS))
    story = re.sub(r"\bThe moral of the story is\s*:", "Moral:", story, flags=re.I)
    story = re.sub(r"\bThe moral\s*:", "Moral:", story, flags=re.I)
    story = re.sub(r"\s*Moral\s*:\s*", "\n\nMoral: ", story, flags=re.I)
    story = re.sub(r"\n{3,}", "\n\n", story).strip()
    if teaching and "moral:" not in story.lower():
        story = f"{story.rstrip()}\n\nMoral: {teaching}"
    return story


def extract_teaching(row: dict[str, str]) -> str:
    input_text = row.get("input", "")
    match = re.search(r"^\s*Teaching\s*:\s*(.+?)\s*$", input_text, flags=re.I | re.M)
    if match:
        return clean_text(match.group(1)).rstrip(".")
    match = MORAL_LINE_RE.search(row.get("output", ""))
    if match:
        return clean_text(match.group(1)).rstrip(".")
    return ""


def normalize_sft_row(row: dict[str, str]) -> dict[str, str]:
    teaching = extract_teaching(row)
    output = normalize_output(row["output"], teaching)
    input_lines = [
        clean_text(line.translate(ASCII_REPLACEMENTS))
        for line in row.get("input", "").replace("\\n", "\n").splitlines()
    ]
    input_text = "\n".join(line for line in input_lines if line)
    return {
        "instruction": row.get("instruction") or DEFAULT_INSTRUCTION,
        "input": input_text,
        "output": output,
    }


def children_story_to_sft(record: dict[str, Any]) -> dict[str, str] | None:
    text = clean_text(record.get("text") or record.get("story") or "")
    if not text:
        return None
    characters = record.get("characters") or record.get("character") or ""
    if isinstance(characters, list):
        character = ", ".join(clean_text(x) for x in characters[:2] if clean_text(x))
    else:
        character = clean_text(characters)
    setting = clean_text(record.get("setting") or record.get("location") or "a child-friendly place")
    tags = record.get("tags") or []
    if isinstance(tags, list):
        tag_words = [clean_text(x).lower() for x in tags if clean_text(x)]
    else:
        tag_words = [clean_text(tags).lower()] if clean_text(tags) else []
    teaching = infer_teaching_from_tags(tag_words)
    if not character:
        character = infer_character_from_story(text)
    if not character:
        return None
    input_text = "\n".join(
        [
            f"Character: {character}",
            f"Setting: {setting}",
            "Challenge: the character faces a small problem and must choose wisely",
            "Outcome: the character learns and makes a kind choice",
            f"Teaching: {teaching}",
        ]
    )
    return {
        "instruction": DEFAULT_INSTRUCTION,
        "input": input_text,
        "output": normalize_output(text, teaching),
    }


def infer_teaching_from_tags(tags: list[str]) -> str:
    joined = " ".join(tags)
    mapping = [
        ("honest", "honesty earns trust"),
        ("friend", "friendship grows through kindness"),
        ("kind", "kindness makes hard days easier"),
        ("share", "sharing turns plenty into friendship"),
        ("patience", "patience helps skill grow"),
        ("team", "teamwork makes hard tasks easier"),
        ("courage", "courage grows with careful steps"),
        ("persist", "perseverance helps us finish"),
        ("empathy", "empathy helps us understand others"),
    ]
    for key, teaching in mapping:
        if key in joined:
            return teaching
    return "kind choices make a better day"


def infer_character_from_story(text: str) -> str:
    patterns = [
        r"\bthere (?:was|lived) (?:a|an|the) ([A-Za-z ]{3,40}?)(?: named| who|\.|,)",
        r"\b(?:a|an|the) ([A-Za-z]+) named ([A-Z][a-z]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = " ".join(part for part in match.groups() if part)
            return clean_text(value).lower()
    return ""


def quality_reasons(row: dict[str, str], min_words: int, max_words: int) -> list[str]:
    reasons: list[str] = []
    output = row["output"].strip()
    input_text = row["input"]
    combined = f"{input_text}\n{output}"
    story_words = words(output)
    story_sentences = sentences(output)
    moral_matches = list(WEAK_FLUENCY_PATTERNS["double_moral"].finditer(output))
    moral_match = MORAL_LINE_RE.search(output)

    if len(story_words) < min_words:
        reasons.append("too_short")
    if len(story_words) > max_words:
        reasons.append("too_long")
    if not moral_match:
        reasons.append("missing_final_moral")
    elif not output.splitlines()[-1].strip().lower().startswith("moral:"):
        reasons.append("moral_not_last_line")
    if len(moral_matches) > 1:
        reasons.append("multiple_morals")
    if len(story_sentences) < 6:
        reasons.append("too_few_sentences")
    if story_sentences and max(len(words(sentence)) for sentence in story_sentences) > 42:
        reasons.append("long_sentence")
    if story_sentences and sum(1 for s in story_sentences if len(words(s)) > 32) >= 3:
        reasons.append("many_long_sentences")
    if UNSAFE_PATTERNS.search(combined):
        reasons.append("unsafe_or_not_child_friendly")
    for name, pattern in BAD_TEXT_PATTERNS.items():
        if pattern.search(combined):
            reasons.append(name)
    if WEAK_FLUENCY_PATTERNS["repetition"].search(output):
        reasons.append("repetition")
    if "Teaching:" not in input_text:
        reasons.append("missing_teaching_input")
    return reasons


def score_candidate(row: dict[str, str]) -> int:
    output = row["output"]
    input_text = row["input"]
    story_words = words(output)
    story_sentences = sentences(output)
    score = 0
    if 120 <= len(story_words) <= 260:
        score += 2
    if output.splitlines()[-1].strip().lower().startswith("moral:"):
        score += 3
    if "Character:" in input_text and "Setting:" in input_text and "Teaching:" in input_text:
        score += 2
    if story_sentences:
        avg_sentence_len = sum(len(words(s)) for s in story_sentences) / len(story_sentences)
        if 8 <= avg_sentence_len <= 24:
            score += 2
    if "\n\n" in output:
        score += 1
    return score


def dedupe_key(row: dict[str, str]) -> str:
    normalized = clean_text(row["output"]).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def collect_from_tf1(source: Iterable[dict[str, Any]], target: int, args: argparse.Namespace) -> tuple[list[Candidate], Counter]:
    accepted: list[Candidate] = []
    rejected = Counter()
    seen: set[str] = set()
    scanned = 0
    for raw in source:
        scanned += 1
        if args.scan_limit and scanned > args.scan_limit:
            break
        formatted = format_record(raw)
        if not formatted:
            rejected["unparseable"] += 1
            continue
        row = normalize_sft_row(formatted)
        key = dedupe_key(row)
        if key in seen:
            rejected["duplicate"] += 1
            continue
        reasons = quality_reasons(row, args.min_words, args.max_words)
        if reasons:
            for reason in reasons:
                rejected[reason] += 1
            continue
        seen.add(key)
        accepted.append(Candidate(row=row, source="tf1", score=score_candidate(row), reasons=[]))
        if len(accepted) >= target:
            break
    rejected["scanned"] = scanned
    return accepted, rejected


def collect_from_children(source: Iterable[dict[str, Any]], target: int, args: argparse.Namespace) -> tuple[list[Candidate], Counter]:
    accepted: list[Candidate] = []
    rejected = Counter()
    seen: set[str] = set()
    scanned = 0
    for raw in source:
        scanned += 1
        row = children_story_to_sft(raw)
        if not row:
            rejected["unparseable"] += 1
            continue
        key = dedupe_key(row)
        if key in seen:
            rejected["duplicate"] += 1
            continue
        reasons = quality_reasons(row, args.min_words, args.max_words)
        if reasons:
            for reason in reasons:
                rejected[reason] += 1
            continue
        seen.add(key)
        accepted.append(Candidate(row=row, source="children_stories", score=score_candidate(row), reasons=[]))
        if len(accepted) >= target:
            break
    rejected["scanned"] = scanned
    return accepted, rejected


def split_rows(rows: list[dict[str, str]], valid_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * valid_ratio))
    return shuffled[n_valid:], shuffled[:n_valid]


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, rows: list[Candidate], train: list[dict[str, str]], valid: list[dict[str, str]], counters: dict[str, Counter], args: argparse.Namespace) -> None:
    source_counts = Counter(row.source for row in rows)
    word_counts = [len(words(row.row["output"])) for row in rows]
    scores = [row.score for row in rows]
    lines = [
        "# Fluency SFT Dataset v1",
        "",
        "Purpose: create a cleaner training set for improving English fluency while preserving fable structure and explicit moral endings.",
        "",
        "## Summary",
        "",
        f"- Total accepted: {len(rows)}",
        f"- Train rows: {len(train)}",
        f"- Valid rows: {len(valid)}",
        f"- Min/max words filter: {args.min_words}-{args.max_words}",
        f"- Valid ratio: {args.valid_ratio}",
        f"- Seed: {args.seed}",
        "",
        "## Source Mix",
        "",
    ]
    for source, count in source_counts.most_common():
        lines.append(f"- {source}: {count}")
    if word_counts:
        lines += [
            "",
            "## Length Statistics",
            "",
            f"- Min words: {min(word_counts)}",
            f"- Max words: {max(word_counts)}",
            f"- Average words: {sum(word_counts) / len(word_counts):.1f}",
            f"- Average quality score: {sum(scores) / len(scores):.2f}/10",
        ]
    lines += ["", "## Filter Rejections", ""]
    for source, counter in counters.items():
        lines.append(f"### {source}")
        for reason, count in counter.most_common(12):
            lines.append(f"- {reason}: {count}")
        lines.append("")
    lines += [
        "## Recommended Use",
        "",
        "Train this as a new LoRA and compare against Base+Repair and Strict+Postprocess. Do not promote it as final unless human evaluation shows fluency improves without losing adherence.",
    ]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/fluency_sft_v1")
    parser.add_argument("--tf1-target", type=int, default=9000)
    parser.add_argument("--children-target", type=int, default=1000)
    parser.add_argument("--tf1-source-jsonl", default="")
    parser.add_argument("--children-source-jsonl", default="")
    parser.add_argument("--tf1-dataset", default=DEFAULT_TF1_DATASET)
    parser.add_argument("--children-dataset", default=DEFAULT_CHILDREN_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-words", type=int, default=110)
    parser.add_argument("--max-words", type=int, default=280)
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str((out_dir / ".hf_cache").resolve()))

    counters: dict[str, Counter] = {}
    candidates: list[Candidate] = []

    if args.tf1_target:
        tf1_source = read_jsonl(Path(args.tf1_source_jsonl)) if args.tf1_source_jsonl else load_hf_stream(args.tf1_dataset, args.split)
        tf1_rows, tf1_counter = collect_from_tf1(tf1_source, args.tf1_target, args)
        candidates.extend(tf1_rows)
        counters["tf1"] = tf1_counter

    if args.children_target:
        children_source = (
            read_jsonl(Path(args.children_source_jsonl))
            if args.children_source_jsonl
            else load_hf_jsonl_file(args.children_dataset, "train.jsonl", out_dir / "raw" / "children_stories")
        )
        child_rows, child_counter = collect_from_children(children_source, args.children_target, args)
        candidates.extend(child_rows)
        counters["children_stories"] = child_counter

    if len(candidates) < 10:
        raise SystemExit(f"Only accepted {len(candidates)} rows; cannot build a useful dataset.")

    candidates.sort(key=lambda item: (-item.score, dedupe_key(item.row)))
    rows = [candidate.row for candidate in candidates]
    train, valid = split_rows(rows, args.valid_ratio, args.seed)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    manifest = {
        "name": "fluency_sft_v1",
        "total": len(rows),
        "train": len(train),
        "valid": len(valid),
        "sources": Counter(candidate.source for candidate in candidates),
        "filters": {
            "min_words": args.min_words,
            "max_words": args.max_words,
            "requires_final_moral": True,
            "rejects_long_sentences": True,
            "rejects_mojibake": True,
            "rejects_child_unsafe_terms": True,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(out_dir / "dataset_report.md", candidates, train, valid, counters, args)

    print(f"Wrote {out_dir}")
    print(f"Accepted total={len(rows)} train={len(train)} valid={len(valid)}")
    for source, count in Counter(candidate.source for candidate in candidates).most_common():
        print(f"{source}: {count}")


if __name__ == "__main__":
    main()

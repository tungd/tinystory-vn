"""Evaluate generated fables with deterministic surface metrics.

The metrics are intentionally simple and reproducible. They do not replace
human scoring, but they catch the failures this project repeatedly observed:
missing moral, weak prompt adherence, unfinished endings, and excessive length.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WORD_RE = re.compile(r"[a-zA-Z']+")
END_RE = re.compile(r"[.!?\"']\s*$")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_words(text: str) -> list[str]:
    return [m.group(0).lower().strip("'") for m in WORD_RE.finditer(text)]


def content_words(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "with",
        "for",
        "is",
        "are",
        "was",
        "were",
        "can",
        "be",
        "by",
        "his",
        "her",
        "their",
        "our",
        "your",
        "my",
    }
    return {w for w in normalize_words(text) if len(w) > 2 and w not in stop}


def phrase_present(phrase: str, story: str) -> bool:
    words = normalize_words(phrase)
    if not words:
        return True
    normalized_story = " ".join(normalize_words(story))
    return " ".join(words) in normalized_story


def moral_text(story: str) -> str:
    match = re.search(r"\bmoral\s*:\s*(.+)$", story, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def has_empty_moral(story: str) -> bool:
    match = re.search(r"\bmoral\s*:\s*(.*)$", story, flags=re.IGNORECASE | re.DOTALL)
    return bool(match and not match.group(1).strip())


def sentence_lengths(story: str) -> list[int]:
    body = re.sub(r"\bmoral\s*:\s*.+$", "", story, flags=re.IGNORECASE | re.DOTALL)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    return [len(normalize_words(sentence)) for sentence in sentences]


def row_metrics(row: dict) -> dict:
    story = row.get("story", "")
    prompt = row.get("input", {})
    words = normalize_words(story)
    moral = moral_text(story)
    lengths = sentence_lengths(story)
    teaching_words = content_words(prompt.get("teaching", ""))
    moral_words = content_words(moral)
    outcome_words = content_words(prompt.get("outcome", ""))
    story_words = set(words)
    has_moral = bool(re.search(r"\bmoral\s*:", story, flags=re.IGNORECASE))
    return {
        "id": row.get("id", ""),
        "status": row.get("status", ""),
        "word_count": len(words),
        "has_moral_label": has_moral,
        "has_moral_footer": bool(moral),
        "empty_moral": has_empty_moral(story),
        "character_exact": phrase_present(prompt.get("character", ""), story),
        "moral_keyword_overlap": len(teaching_words & moral_words),
        "moral_exact": bool(teaching_words) and teaching_words.issubset(moral_words),
        "outcome_keyword_overlap": len(outcome_words & story_words),
        "outcome_covered": bool(outcome_words) and len(outcome_words & story_words) >= max(1, len(outcome_words) // 2),
        "clean_ending": bool(END_RE.search(story.strip())),
        "max_sentence_words": max(lengths) if lengths else 0,
        "has_run_on_sentence": bool(lengths and max(lengths) > 65),
        "latency_ms": row.get("meta", {}).get("latency_ms", 0),
        "output_tokens": row.get("meta", {}).get("output_tokens", 0),
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    success = [r for r in rows if r["status"] == "success"]
    denom = len(success) or 1

    def rate(key: str) -> float:
        return round(sum(1 for r in success if r[key]) / denom, 3)

    latencies = [r["latency_ms"] for r in success if r["latency_ms"]]
    word_counts = [r["word_count"] for r in success]
    return {
        "total": len(rows),
        "success": len(success),
        "has_moral_label_rate": rate("has_moral_label"),
        "has_moral_footer_rate": rate("has_moral_footer"),
        "empty_moral_rate": rate("empty_moral"),
        "character_exact_rate": rate("character_exact"),
        "moral_exact_rate": rate("moral_exact"),
        "outcome_covered_rate": rate("outcome_covered"),
        "clean_ending_rate": rate("clean_ending"),
        "run_on_sentence_rate": rate("has_run_on_sentence"),
        "avg_words": round(sum(word_counts) / len(word_counts), 1) if word_counts else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
    }


def write_markdown(path: Path, model_name: str, summary: dict, rows: list[dict]) -> None:
    lines = [
        f"# Evaluation: {model_name}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Per Prompt",
            "",
            "| ID | Moral | Character | Moral exact | Outcome | Ending | Words | Latency ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {id} | {has_moral_footer} | {character_exact} | {moral_exact} | "
            "{outcome_covered} | {clean_ending} | {word_count} | {latency_ms} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    rows = [row_metrics(row) for row in read_jsonl(Path(args.input))]
    summary = aggregate(rows)
    output = {"model_name": args.model_name, "summary": summary, "rows": rows}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(Path(args.out_md), args.model_name, summary, rows)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

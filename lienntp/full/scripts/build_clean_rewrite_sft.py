"""Build a stricter clean/rewrite SFT dataset from synthetic fables.

The goal is to improve over the previous Clean-3K run by removing examples
with weak prose before training. Optionally, the script can rewrite examples
with a teacher model, but the deterministic filter/export path is fast enough
to run locally.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from app.ollama_client import generate_meta


DEFAULT_INPUTS = [
    "experiments/sft_clean_refiltered_3k/sft_clean_refiltered_dataset/train.jsonl",
    "experiments/sft_clean_refiltered_3k/sft_clean_refiltered_dataset/valid.jsonl",
]
DEFAULT_OUT = "experiments/sft_clean_rewrite_1k"
DEFAULT_INSTRUCTION = "Write a short English fable for children with a clear moral."

REWRITE_SYSTEM = (
    "You are a careful editor of children's fables. Rewrite stories in clear, simple, "
    "natural English for ages 4-7. Preserve the requested character, setting, challenge, "
    "outcome, and teaching. Use 2-4 short paragraphs. Remove run-on sentences, odd titles, "
    "logic jumps, and unrelated details. End with exactly one final line that starts with 'Moral:'. "
    "Return only the rewritten fable."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_input(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def normalize_story(story: str, teaching: str) -> str:
    story = story.strip()
    story = re.sub(r"\r\n?", "\n", story)
    story = re.sub(r"[ \t]+", " ", story)
    story = re.sub(r"\n{3,}", "\n\n", story)
    story = re.sub(r"(?im)^\s*Moral\s*:\s*$", f"Moral: {teaching}", story)
    if "moral:" not in story.lower():
        story = f"{story.rstrip()}\n\nMoral: {teaching}"
    return story.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def sentence_word_lengths(text: str) -> list[int]:
    body = re.sub(r"(?im)^\s*Moral\s*:.*$", "", text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    return [word_count(sentence) for sentence in sentences]


def paragraph_word_lengths(text: str) -> list[int]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [word_count(paragraph) for paragraph in paragraphs]


def first_line_title_like(story: str) -> bool:
    lines = [line.strip() for line in story.splitlines() if line.strip()]
    if not lines:
        return False
    first = lines[0]
    if len(first.split()) <= 7 and not first.endswith((".", "!", "?", ":")):
        return True
    title_words = sum(1 for word in first.split() if word[:1].isupper())
    return len(first.split()) <= 8 and title_words >= max(3, len(first.split()) - 1)


def quality_reasons(row: dict) -> list[str]:
    reasons: list[str] = []
    fields = parse_input(row.get("input", ""))
    teaching = fields.get("teaching", "")
    story = normalize_story(row.get("output", ""), teaching)
    lower = story.lower()
    words = word_count(story)
    paragraphs = [p.strip() for p in story.split("\n\n") if p.strip()]
    sentence_lengths = sentence_word_lengths(story)
    paragraph_lengths = paragraph_word_lengths(story)

    if words < 120:
        reasons.append("too_short")
    if words > 260:
        reasons.append("too_long")
    if len(paragraphs) < 2:
        reasons.append("too_few_paragraphs")
    if len(paragraphs) > 5:
        reasons.append("too_many_paragraphs")
    if sentence_lengths and max(sentence_lengths) > 45:
        reasons.append("run_on_sentence")
    if sentence_lengths and sum(sentence_lengths) / len(sentence_lengths) > 25:
        reasons.append("high_avg_sentence_length")
    if paragraph_lengths and max(paragraph_lengths) > 120:
        reasons.append("paragraph_too_long")
    if not re.search(r"(?im)^\s*Moral\s*:\s*\S+", story):
        reasons.append("missing_moral")
    if not re.search(r"(?im)\n\s*Moral\s*:", story):
        reasons.append("moral_not_final_line")
    if first_line_title_like(story):
        reasons.append("title_like_first_line")
    if any(token in story for token in ("&", "_", "...")):
        reasons.append("bad_punctuation_or_symbols")
    if re.search(r"\b(I|we|my|our)\b", story) and re.search(r"\b(lived|was|were|said|asked)\b", lower):
        # A light signal for first-person drift in mostly third-person stories.
        if lower.count(" i ") + lower.count(" my ") + lower.count(" we ") >= 2:
            reasons.append("possible_pov_drift")
    if teaching and teaching.lower() not in lower:
        reasons.append("teaching_missing")

    for key in ("character", "setting"):
        value = fields.get(key, "")
        keyword = value.split()[-1].lower() if value.split() else ""
        if keyword and keyword not in lower:
            reasons.append(f"{key}_keyword_missing")

    return reasons


def clean_row(row: dict) -> dict:
    fields = parse_input(row.get("input", ""))
    teaching = fields.get("teaching", "")
    return {
        "instruction": row.get("instruction") or DEFAULT_INSTRUCTION,
        "input": row["input"].strip(),
        "output": normalize_story(row["output"], teaching),
    }


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(read_jsonl(Path(path)))
    return rows


def command_filter(args: argparse.Namespace) -> int:
    rows = load_rows(args.inputs)
    accepted: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        cleaned = clean_row(row)
        reasons = quality_reasons(cleaned)
        if reasons:
            rejected.append({**cleaned, "filter_reasons": reasons})
        else:
            accepted.append(cleaned)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "filtered_accepted.jsonl", accepted)
    write_jsonl(out_dir / "filtered_rejected.jsonl", rejected)
    summary: dict[str, int] = {}
    for row in rejected:
        for reason in row["filter_reasons"]:
            summary[reason] = summary.get(reason, 0) + 1
    (out_dir / "filter_summary.json").write_text(
        json.dumps(
            {
                "input": len(rows),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "rejection_reasons": dict(sorted(summary.items(), key=lambda x: (-x[1], x[0]))),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Filtered {len(rows)} rows: accepted={len(accepted)} rejected={len(rejected)}")
    return 0


def rewrite_prompt(row: dict) -> str:
    fields = parse_input(row["input"])
    return (
        "Rewrite the draft fable below into a polished final fable.\n\n"
        "Required elements:\n"
        f"- Character: {fields.get('character', '')}\n"
        f"- Setting: {fields.get('setting', '')}\n"
        f"- Challenge: {fields.get('challenge', '')}\n"
        f"- Outcome: {fields.get('outcome', '')}\n"
        f"- Teaching/Moral: {fields.get('teaching', '')}\n\n"
        "Draft:\n"
        f"{row['output']}\n\n"
        f"Final line must be exactly: Moral: {fields.get('teaching', '')}"
    )


def command_rewrite(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    done = len(read_jsonl(out_path)) if out_path.exists() else 0
    rows = rows[done:]
    for index, row in enumerate(rows, done + 1):
        fields = parse_input(row["input"])
        print(f"[{index}] rewrite {fields.get('character', '')} / {fields.get('teaching', '')}", flush=True)
        result = generate_meta(
            prompt=rewrite_prompt(row),
            system=REWRITE_SYSTEM,
            model=args.model,
            num_predict=args.num_predict,
            seed=args.seed,
            temperature=args.temperature,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        rewritten = {
            "instruction": row.get("instruction") or DEFAULT_INSTRUCTION,
            "input": row["input"],
            "output": normalize_story(result["text"], fields.get("teaching", "")),
            "rewrite_meta": {
                "teacher_model": args.model,
                "latency_ms": result.get("latency_ms", 0),
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            },
        }
        append_jsonl(out_path, rewritten)
    print(f"Wrote {out_path}")
    return 0


def command_export(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input))
    final: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        cleaned = clean_row(row)
        reasons = quality_reasons(cleaned)
        if reasons:
            rejected.append({**cleaned, "filter_reasons": reasons})
            continue
        final.append(cleaned)
    if args.limit:
        final = final[: args.limit]
    if len(final) < 10:
        raise SystemExit(f"Only {len(final)} rows passed final filter.")

    random.Random(args.seed).shuffle(final)
    n_valid = max(1, round(len(final) * args.valid_ratio))
    valid = final[:n_valid]
    train = final[n_valid:]

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "final_rejected.jsonl", rejected)
    manifest = {
        "name": "sft-clean-rewrite-1k",
        "total": len(final),
        "train": len(train),
        "valid": len(valid),
        "source": str(args.input),
        "valid_ratio": args.valid_ratio,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(final)} rows: train={len(train)} valid={len(valid)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser("filter")
    filter_parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    filter_parser.add_argument("--out-dir", default=DEFAULT_OUT)
    filter_parser.set_defaults(func=command_filter)

    rewrite_parser = subparsers.add_parser("rewrite")
    rewrite_parser.add_argument("--input", default=f"{DEFAULT_OUT}/filtered_rejected.jsonl")
    rewrite_parser.add_argument("--out", default=f"{DEFAULT_OUT}/rewritten.jsonl")
    rewrite_parser.add_argument("--model", default="llama3.2:3b-instruct-fp16")
    rewrite_parser.add_argument("--limit", type=int, default=0)
    rewrite_parser.add_argument("--num-predict", type=int, default=360)
    rewrite_parser.add_argument("--temperature", type=float, default=0.4)
    rewrite_parser.add_argument("--seed", type=int, default=42)
    rewrite_parser.set_defaults(func=command_rewrite)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--input", default=f"{DEFAULT_OUT}/filtered_accepted.jsonl")
    export_parser.add_argument("--out-dir", default=f"{DEFAULT_OUT}/dataset")
    export_parser.add_argument("--limit", type=int, default=1000)
    export_parser.add_argument("--valid-ratio", type=float, default=0.1)
    export_parser.add_argument("--seed", type=int, default=42)
    export_parser.set_defaults(func=command_export)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()

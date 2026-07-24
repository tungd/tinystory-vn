"""Create human evaluation templates for a subset of English model outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_MODELS = [
    ("Llama 3.2 3B Instruct FP16", "results/baseline_outputs.jsonl"),
    ("Qwen 2.5 3B", "results/qwen25_3b_outputs.jsonl"),
]

FIELDS = [
    "prompt_id",
    "model_name",
    "model_id",
    "ollama",
    "character",
    "setting",
    "challenge",
    "outcome",
    "teaching",
    "length",
    "story",
    "score_english_fluency_1_5",
    "score_prompt_adherence_1_5",
    "score_fable_structure_1_5",
    "score_moral_clarity_1_5",
    "score_child_safety_1_5",
    "average_score",
    "notes",
]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_rows(limit: int, models: list[tuple[str, Path]]) -> list[dict]:
    rows: list[dict] = []
    for model_name, output_path in models:
        if not output_path.exists():
            continue
        for item in read_jsonl(output_path)[:limit]:
            prompt = item["input"]
            rows.append(
                {
                    "prompt_id": item["id"],
                    "model_name": model_name,
                    "model_id": item.get("model_id", ""),
                    "ollama": item.get("ollama", ""),
                    "character": prompt.get("character", ""),
                    "setting": prompt.get("setting", ""),
                    "challenge": prompt.get("challenge", ""),
                    "outcome": prompt.get("outcome", ""),
                    "teaching": prompt.get("teaching", ""),
                    "length": prompt.get("length", ""),
                    "story": item.get("story", "").strip(),
                    "score_english_fluency_1_5": "",
                    "score_prompt_adherence_1_5": "",
                    "score_fable_structure_1_5": "",
                    "score_moral_clarity_1_5": "",
                    "score_child_safety_1_5": "",
                    "average_score": "",
                    "notes": "",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_prompt: dict[str, list[dict]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], []).append(row)

    lines = [
        "# Human Evaluation",
        "",
        "Scores use 1-5 where 1 is poor and 5 is excellent.",
        "",
        "| Criterion | Meaning |",
        "| --- | --- |",
        "| English fluency | Grammar, wording, and naturalness. |",
        "| Prompt adherence | Uses the requested character, setting, challenge, outcome, and teaching. |",
        "| Fable structure | Has a clear setup, conflict, resolution, and concise fable style. |",
        "| Moral clarity | Ends with or clearly states the intended moral. |",
        "| Child safety | Wholesome and age-appropriate. |",
        "",
    ]

    for prompt_id, prompt_rows in by_prompt.items():
        first = prompt_rows[0]
        lines.extend(
            [
                f"## {prompt_id}",
                "",
                "### Prompt",
                "",
                "```text",
                f"Character: {first['character']}",
                f"Setting: {first['setting']}",
                f"Challenge: {first['challenge']}",
                f"Outcome: {first['outcome']}",
                f"Teaching: {first['teaching']}",
                f"Length: {first['length']}",
                "```",
                "",
            ]
        )

        for row in prompt_rows:
            lines.extend(
                [
                    f"### Model: {row['model_name']}",
                    "",
                    f"- Model ID: `{row['model_id']}`",
                    f"- Ollama: `{row['ollama']}`",
                    "",
                    "#### Story",
                    "",
                    row["story"] or "_No story generated._",
                    "",
                    "#### Scores",
                    "",
                    "| Criterion | Score |",
                    "| --- | --- |",
                    "| English fluency |  |",
                    "| Prompt adherence |  |",
                    "| Fable structure |  |",
                    "| Moral clarity |  |",
                    "| Child safety |  |",
                    "| Average |  |",
                    "",
                    "Notes:",
                    "",
                    "---",
                    "",
                ]
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", default="results/human_eval_template.csv")
    parser.add_argument("--out-md", default="")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add a model output file, e.g. 'Qwen=results/qwen25_3b_outputs.jsonl'.",
    )
    args = parser.parse_args()

    if args.model:
        models: list[tuple[str, Path]] = []
        for spec in args.model:
            if "=" not in spec:
                raise SystemExit(f"Invalid --model value: {spec!r}. Use NAME=PATH.")
            name, path = spec.split("=", 1)
            models.append((name, Path(path)))
    else:
        models = [(name, Path(path)) for name, path in DEFAULT_MODELS]

    rows = build_rows(args.limit, models)
    write_csv(Path(args.out), rows)
    print(f"Wrote {args.out} with {len(rows)} rows")
    if args.out_md:
        write_markdown(Path(args.out_md), rows)
        print(f"Wrote {args.out_md} with {len(rows)} rows")


if __name__ == "__main__":
    main()

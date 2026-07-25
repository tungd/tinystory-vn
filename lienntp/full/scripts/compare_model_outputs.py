"""Create a simple comparison report for two English batch output files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metrics(rows: list[dict]) -> dict:
    successes = [r for r in rows if r.get("status") == "success"]
    blocked = [r for r in successes if r.get("output_safety") != "ok"]
    words: list[str] = []
    bigrams: list[tuple[str, str]] = []
    moral_outputs = 0
    for row in successes:
        story = row.get("story", "")
        if re.search(r"\b(moral|lesson|teaching)\b", story.lower()):
            moral_outputs += 1
        toks = re.findall(r"\w+", story.lower(), flags=re.UNICODE)
        words.extend(toks)
        bigrams.extend(zip(toks, toks[1:]))
    latencies = [r.get("meta", {}).get("latency_ms", 0) for r in successes]
    output_tokens = [r.get("meta", {}).get("output_tokens", 0) for r in successes]
    return {
        "total": len(rows),
        "success": len(successes),
        "errors": len(rows) - len(successes),
        "blocked_outputs": len(blocked),
        "moral_outputs": moral_outputs,
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "avg_output_tokens": round(sum(output_tokens) / len(output_tokens), 1) if output_tokens else 0,
        "distinct_1": round(len(set(words)) / len(words), 4) if words else 0,
        "distinct_2": round(len(set(bigrams)) / len(bigrams), 4) if bigrams else 0,
    }


def write_markdown(path: Path, left_name: str, left: dict, right_name: str, right: dict) -> None:
    rows = [
        ("Success", "success"),
        ("Errors", "errors"),
        ("Blocked outputs", "blocked_outputs"),
        ("Outputs with moral/lesson/teaching", "moral_outputs"),
        ("Avg latency ms", "avg_latency_ms"),
        ("Avg output tokens", "avg_output_tokens"),
        ("Distinct-1", "distinct_1"),
        ("Distinct-2", "distinct_2"),
    ]
    lines = [
        "# English Baseline Model Comparison",
        "",
        f"| Metric | {left_name} | {right_name} |",
        "|---|---:|---:|",
    ]
    for label, key in rows:
        lines.append(f"| {label} | {left[key]} | {right[key]} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Both models were evaluated on the same fixed English prompts from `data/test_prompts.jsonl`.",
            "- These are automatic surface metrics. Human or LLM-as-judge scoring is still needed for English fluency, moral clarity, and prompt adherence.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="results/baseline_outputs.jsonl")
    parser.add_argument("--left-name", default="Llama 3.2 3B Instruct FP16")
    parser.add_argument("--right", default="results/qwen25_3b_outputs.jsonl")
    parser.add_argument("--right-name", default="Qwen 2.5 3B")
    parser.add_argument("--out-json", default="results/model_comparison.json")
    parser.add_argument("--out-md", default="results/model_comparison.md")
    args = parser.parse_args()

    left = metrics(read_jsonl(Path(args.left)))
    right = metrics(read_jsonl(Path(args.right)))
    data = {
        "left": {"name": args.left_name, "metrics": left},
        "right": {"name": args.right_name, "metrics": right},
    }
    Path(args.out_json).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(Path(args.out_md), args.left_name, left, args.right_name, right)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

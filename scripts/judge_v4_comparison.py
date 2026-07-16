"""Strictly judge a reproducible paired subset of v3-full versus v4."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import google_judge_client, judge
from scripts.evaluate_v3_comparison import checks
from scripts.judge_v3_comparison import summarize


def select_sources(rows: list[dict], count: int, seed: int) -> list[str]:
    sources = [row["source"] for row in rows if row["model"] == "v3-full"]
    return random.Random(seed).sample(sources, min(count, len(sources)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    selected = set(args.source or select_sources(data["generations"], args.controls, args.seed))
    selected_rows = [row for row in data["generations"] if row["source"] in selected]
    judged = []
    for index, row in enumerate(selected_rows, 1):
        request = f"Main character: {row['character']}\nTeaching: {row['moral']}"
        started = time.perf_counter()
        scores = judge.evaluate(
            row["story"],
            request,
            model="gemma-4-26b-a4b-it",
            gen=google_judge_client.generate,
        )
        judged.append({
            **row,
            "checks": checks(row),
            "judge": scores,
            "judge_latency_ms": round((time.perf_counter() - started) * 1000),
        })
        print(f"judged {index}/{len(selected_rows)}: {row['model']}", flush=True)

    result = {
        "kind": "v4-judge-comparison",
        "source": args.input,
        "selection": {
            "method": "explicit sources" if args.source else "seeded random paired controls",
            "seed": args.seed,
            "controls": len(selected),
            "stories": len(judged),
        },
        "judge_settings": {
            "backend": "google-genai",
            "model": "gemma-4-26b-a4b-it",
            "thinking_level": google_judge_client.JUDGE_THINKING_LEVEL,
            "response_mime_type": "application/json",
            "prompt_version": judge.PROMPT_VERSION,
        },
        "summary": summarize(judged),
        "judgments": judged,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

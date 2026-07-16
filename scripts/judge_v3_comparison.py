"""Judge a reproducible paired subset of a v2/v3 generation comparison."""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import google_judge_client, judge
from scripts.evaluate_v3_comparison import checks


def select_sources(rows: list[dict], count: int, seed: int) -> list[str]:
    sources = [row["source"] for row in rows if row["model"] == "v2"]
    return random.Random(seed).sample(sources, min(count, len(sources)))


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    result = {}
    for model, model_rows in grouped.items():
        result[model] = {
            "n": len(model_rows),
            "judge_mean": {
                axis: round(
                    sum(row["judge"][axis] for row in model_rows) / len(model_rows), 2
                )
                for axis in judge.AXES + ["overall"]
            },
            "mean_latency_ms": round(
                sum(row["judge_latency_ms"] for row in model_rows) / len(model_rows)
            ),
            "checks_passed": {
                key: sum(int(row["checks"][key]) for row in model_rows)
                for key in model_rows[0]["checks"]
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    selected = set(select_sources(data["generations"], args.controls, args.seed))
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
        judged.append(
            {
                **row,
                "checks": checks(row),
                "judge": scores,
                "judge_latency_ms": round((time.perf_counter() - started) * 1000),
            }
        )
        print(f"judged {index}/{len(selected_rows)}: {row['model']}", flush=True)

    result = {
        "kind": "v3-full-judge-comparison",
        "source": args.input,
        "selection": {
            "method": "seeded random paired controls",
            "seed": args.seed,
            "controls": len(selected),
            "stories": len(judged),
        },
        "judge_settings": {
            "backend": "google-genai",
            "model": "gemma-4-26b-a4b-it",
            "thinking_level": "minimal",
            "response_mime_type": "application/json",
        },
        "summary": summarize(judged),
        "judgments": judged,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

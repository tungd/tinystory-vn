"""Compute deterministic conditioning metrics for matched v2/v3 generations."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def contains_phrase(phrase: str, text: str) -> bool:
    phrase = " ".join(phrase.casefold().split())
    text = " ".join(text.casefold().split())
    return re.search(rf"(?<![\w'-]){re.escape(phrase)}(?![\w'-])", text) is not None


def checks(row: dict) -> dict:
    story = row["story"]
    near_end = " ".join(story.split()[-80:])
    exact_character = contains_phrase(row["character"], story)
    exact_moral = contains_phrase(row["moral"], story)
    return {
        "exact_character": exact_character,
        "exact_moral": exact_moral,
        "exact_moral_near_end": contains_phrase(row["moral"], near_end),
        "exact_both": exact_character and exact_moral,
        "ended": bool(row["ended"]),
    }


def summarize(rows: list[dict]) -> dict:
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    summary = {}
    for model, model_rows in by_model.items():
        n = len(model_rows)
        counts = {
            key: sum(int(row["checks"][key]) for row in model_rows)
            for key in model_rows[0]["checks"]
        }
        summary[model] = {
            "n": n,
            "counts": counts,
            "rates": {key: round(value / n, 4) for key, value in counts.items()},
            "mean_words": round(sum(len(row["story"].split()) for row in model_rows) / n, 1),
            "mean_output_tokens": round(sum(row["output_tokens"] for row in model_rows) / n, 1),
        }
    return summary


def paired_changes(rows: list[dict]) -> dict:
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["source"]][row["model"]] = row
    models = sorted({row["model"] for row in rows})
    if len(models) != 2:
        return {}
    before, after = "v2", "v3-full"
    changes = {}
    for metric in rows[0]["checks"]:
        gained = lost = unchanged = 0
        for pair in pairs.values():
            old = pair[before]["checks"][metric]
            new = pair[after]["checks"][metric]
            gained += int(not old and new)
            lost += int(old and not new)
            unchanged += int(old == new)
        changes[metric] = {"gained": gained, "lost": lost, "unchanged": unchanged}
    return changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    rows = data["generations"]
    for row in rows:
        row["checks"] = checks(row)
    result = {
        "kind": "v3-full-deterministic-evaluation",
        "source": args.input,
        "controls": data["controls"],
        "settings": data["settings"],
        "summary": summarize(rows),
        "paired_changes": paired_changes(rows),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["paired_changes"], indent=2))


if __name__ == "__main__":
    main()

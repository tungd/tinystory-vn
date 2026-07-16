"""Compute deterministic conditioning metrics for v3-full versus v5."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_v3_comparison import checks, summarize


def paired_changes(rows: list[dict]) -> dict:
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["source"]][row["model"]] = row
    changes = {}
    for metric in rows[0]["checks"]:
        gained = lost = unchanged = 0
        for pair in pairs.values():
            old = pair["v3-full"]["checks"][metric]
            new = pair["v5"]["checks"][metric]
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
        "kind": "v5-deterministic-evaluation",
        "source": args.input,
        "controls": data["controls"],
        "settings": data["settings"],
        "summary": summarize(rows),
        "paired_changes": paired_changes(rows),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

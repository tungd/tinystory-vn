"""Summarize V5 train/holdout, length, and moral-ablation diagnostics."""

import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from scripts.evaluate_v3_comparison import contains_phrase


def body(story: str) -> str:
    return re.split(r"\n+\s*Moral\s*:", story, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def similarity(left: str, right: str) -> float:
    normalize = lambda value: " ".join(body(value).casefold().split())
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def summarize_group(rows: list[dict]) -> dict:
    n = len(rows)
    character = sum(contains_phrase(row["character"], row["story"]) for row in rows)
    moral_rows = [row for row in rows if row["requested_moral"]]
    moral = sum(contains_phrase(row["requested_moral"], row["story"]) for row in moral_rows)
    both = sum(
        contains_phrase(row["character"], row["story"])
        and bool(row["requested_moral"])
        and contains_phrase(row["requested_moral"], row["story"])
        for row in rows
    )
    return {
        "n": n,
        "exact_character": round(character / n, 4),
        "exact_requested_moral": round(moral / len(moral_rows), 4) if moral_rows else None,
        "exact_both": round(both / n, 4) if moral_rows else None,
        "ended": round(sum(row["ended"] for row in rows) / n, 4),
        "mean_words": round(sum(len(row["story"].split()) for row in rows) / n, 1),
    }


def evaluate(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    indexed = {}
    for row in rows:
        key = (row["split"], row["condition"], row["max_new_tokens"])
        grouped[key].append(row)
        indexed[(row["split"], row["source"], row["condition"], row["max_new_tokens"])] = row
    groups = {
        f"{split}:{condition}:{length}": summarize_group(members)
        for (split, condition, length), members in sorted(grouped.items())
    }
    sensitivity = {}
    for split in ("train", "holdout"):
        originals = grouped[(split, "original", 180)]
        swaps = []
        blanks = []
        old_moral_retained = requested_moral_followed = 0
        for original in originals:
            swap = indexed[(split, original["source"], "swapped_moral", 180)]
            blank = indexed[(split, original["source"], "blank_moral", 180)]
            swaps.append(similarity(original["story"], swap["story"]))
            blanks.append(similarity(original["story"], blank["story"]))
            requested_moral_followed += contains_phrase(swap["requested_moral"], swap["story"])
            old_moral_retained += contains_phrase(original["original_moral"], swap["story"])
        n = len(originals)
        sensitivity[split] = {
            "mean_body_similarity_original_vs_swapped": round(sum(swaps) / n, 4),
            "mean_body_similarity_original_vs_blank": round(sum(blanks) / n, 4),
            "swapped_requested_moral_rate": round(requested_moral_followed / n, 4),
            "swapped_old_moral_rate": round(old_moral_retained / n, 4),
        }
    return {"groups": groups, "moral_sensitivity": sensitivity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    result = {
        "kind": "v5-conditioning-diagnostic-evaluation",
        "source": args.input,
        "settings": data["settings"],
        **evaluate(data["generations"]),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

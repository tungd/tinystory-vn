"""Summarize latest v5 annotations without tracking raw public-domain text."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.label_v5 import latest_by_source, load_jsonl
from scripts.prepare_v5 import build_external_control, build_real_example, restrict_annotations


def summarize(rows: list[dict]) -> dict:
    latest = latest_by_source(rows)
    collections = defaultdict(
        lambda: {"annotated": 0, "annotation_accepted": 0, "prepared": 0, "external": 0}
    )
    rejection_reasons = Counter()
    annotation_accepted = prepared = external = api_errors = 0
    for row in latest.values():
        annotation = row["annotation"]
        collection = collections[row["collection"]]
        collection["annotated"] += 1
        accepted = bool(annotation.get("accepted"))
        annotation_accepted += int(accepted)
        collection["annotation_accepted"] += int(accepted)
        ready = build_real_example(row) is not None
        prepared += int(ready)
        collection["prepared"] += int(ready)
        external_ready = build_external_control(row) is not None
        external += int(external_ready)
        collection["external"] += int(external_ready)
        rejection_reasons.update(annotation.get("rejection_reasons", []))
        api_errors += int("api_error" in annotation.get("rejection_reasons", []))
    return {
        "kind": "v5-source-annotation-audit",
        "annotated": len(latest),
        "annotation_accepted": annotation_accepted,
        "prepared_after_cleanup": prepared,
        "external_controls_after_cleanup": external,
        "api_errors": api_errors,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "collections": dict(sorted(collections.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/v5/data/annotations.jsonl")
    parser.add_argument("--candidates", default="runs/v5/data/candidates.jsonl")
    parser.add_argument("--out", default="runs/v5/results/source_audit.json")
    args = parser.parse_args()
    source = Path(args.input)
    rows = restrict_annotations(load_jsonl(source), load_jsonl(args.candidates))
    result = summarize(rows)
    result["annotations_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

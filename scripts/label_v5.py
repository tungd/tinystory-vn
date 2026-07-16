"""Annotate and strictly filter v5 public-domain stories with Gemma."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import google_judge_client
from app.judge import _extract_json


SYSTEM_INSTRUCTION = (
    "You are a strict children's-literature data curator. Annotate the supplied "
    "human-authored story; never rewrite it. Reject incomplete, incoherent, unsafe, "
    "or weakly moralized text. Output JSON only."
)

_rate_lock = threading.Lock()
_next_request_at = 0.0
_request_interval = 0.0


def configure_rate_limit(seconds: float) -> None:
    global _next_request_at, _request_interval
    with _rate_lock:
        _request_interval = max(0.0, seconds)
        _next_request_at = 0.0


def _wait_for_rate_slot() -> None:
    global _next_request_at
    with _rate_lock:
        now = time.monotonic()
        delay = max(0.0, _next_request_at - now)
        _next_request_at = max(now, _next_request_at) + _request_interval
    if delay:
        time.sleep(delay)


def build_annotation_prompt(row: dict) -> str:
    supplied_moral = row.get("provided_moral")
    moral_rule = (
        f'- The supplied moral is fixed exactly as: "{supplied_moral}". Judge whether the '
        "events causally demonstrate it; return it unchanged.\n"
        if supplied_moral else
        "- moral must be one concise general lesson causally demonstrated by the outcome.\n"
    )
    return (
        "Annotate this complete story for controlled fable training. Do not rewrite or "
        "continue it. Judge only its actual events.\n\n"
        "Return exactly this JSON shape:\n"
        '{"protagonist_anchor":"exact article+noun phrase copied from story",'
        '"trait":"one lowercase adjective demonstrated by the protagonist actions",'
        '"moral":"one concise general lesson causally demonstrated by the outcome",'
        '"coherence":1,"moral_causality":1,"complete":true,'
        '"child_suitable":true,"accept":true,"reason":"brief evidence"}\n\n'
        "Rules:\n"
        "- protagonist_anchor must be an exact contiguous phrase already in STORY, "
        "normally beginning a/an/the, and identify the active protagonist.\n"
        "- trait must describe behavior visibly demonstrated in the plot, not an invented trait.\n"
        + moral_rule +
        "- coherence and moral_causality use 1-5; accept requires both >=4.\n"
        "- reject excerpts, framing text, disconnected events, archaic text too hard for "
        "children, or stories whose lesson is unclear.\n\n"
        f"COLLECTION: {row['collection']}\nTITLE: {row['title']}\n\n"
        f"STORY:\n{row['story']}\n\nJSON:"
    )


def _find_exact_casefold(haystack: str, needle: str) -> tuple[int, int] | None:
    match = re.search(re.escape(needle), haystack, flags=re.IGNORECASE)
    return match.span() if match else None


def parse_annotation(raw: str, story: str, provided_moral: str | None = None) -> dict:
    data = _extract_json(raw)
    anchor = " ".join(str(data.get("protagonist_anchor", "")).split())
    trait = str(data.get("trait", "")).strip().casefold()
    moral = " ".join(str(provided_moral or data.get("moral", "")).split()).strip(" \"'")
    reason = " ".join(str(data.get("reason", "")).split())
    try:
        coherence = int(data.get("coherence", 0))
        moral_causality = int(data.get("moral_causality", 0))
    except (TypeError, ValueError):
        coherence = moral_causality = 0

    failures = []
    if not anchor or _find_exact_casefold(story, anchor) is None:
        failures.append("anchor_not_exact")
    if not re.fullmatch(r"[a-z]+(?:-[a-z]+)?", trait):
        failures.append("invalid_trait")
    if not 3 <= len(moral.split()) <= 20 or len(moral) > 140:
        failures.append("invalid_moral")
    if not 1 <= coherence <= 5 or not 1 <= moral_causality <= 5:
        failures.append("invalid_scores")
    complete = data.get("complete") is True
    child_suitable = data.get("child_suitable") is True
    model_accept = data.get("accept") is True
    accepted = (
        model_accept
        and complete
        and child_suitable
        and coherence >= 4
        and moral_causality >= 4
        and not failures
    )
    if not model_accept:
        failures.append("model_rejected")
    if not complete:
        failures.append("incomplete")
    if not child_suitable:
        failures.append("not_child_suitable")
    if coherence < 4:
        failures.append("low_coherence")
    if moral_causality < 4:
        failures.append("weak_moral_causality")
    return {
        "protagonist_anchor": anchor,
        "trait": trait,
        "moral": moral,
        "coherence": coherence,
        "moral_causality": moral_causality,
        "complete": complete,
        "child_suitable": child_suitable,
        "accepted": accepted,
        "reason": reason,
        "rejection_reasons": sorted(set(failures)),
    }


def stratified_rows(rows: list[dict], seed: int) -> list[dict]:
    groups: dict[str, deque] = defaultdict(deque)
    rng = random.Random(seed)
    for collection, members in _group_rows(rows).items():
        rng.shuffle(members)
        groups[collection].extend(members)
    result = []
    while groups:
        for collection in list(groups):
            result.append(groups[collection].popleft())
            if not groups[collection]:
                del groups[collection]
    return result


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["collection"]].append(row)
    return groups


def annotate(row: dict, model: str, retries: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            _wait_for_rate_slot()
            raw = google_judge_client.generate(
                prompt=build_annotation_prompt(row),
                system=SYSTEM_INSTRUCTION,
                model=model,
                num_predict=800,
                temperature=0.0,
            )
            return {
                **row,
                "annotation": parse_annotation(raw, row["story"], row.get("provided_moral")),
            }
        except Exception as exc:  # API errors are retried and persisted as failures.
            last_error = exc
            retry_match = re.search(r"retry in ([\d.]+)s|retryDelay[^\d]+(\d+)s", str(exc))
            retry_after = max(float(value) for value in retry_match.groups() if value) \
                if retry_match else 0
            time.sleep(max(2**attempt, retry_after + 1))
    return {
        **row,
        "annotation": {
            "accepted": False,
            "rejection_reasons": ["api_error"],
            "reason": f"{type(last_error).__name__}: {last_error}",
        },
    }


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def latest_by_source(rows: list[dict]) -> dict[str, dict]:
    return {row["source"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/v5/data/candidates.jsonl")
    parser.add_argument("--out", default="runs/v5/data/annotations.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--request-interval", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gemma-4-26b-a4b-it")
    args = parser.parse_args()

    rows = stratified_rows(load_jsonl(args.input), args.seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        latest = latest_by_source(load_jsonl(output))
        completed = {
            source for source, row in latest.items()
            if "api_error" not in row["annotation"].get("rejection_reasons", [])
        }
    pending = [row for row in rows if row["source"] not in completed]
    if args.limit:
        pending = pending[: max(0, args.limit - len(completed))]
    if not pending:
        print(f"nothing pending; {len(completed)} already annotated")
        return

    configure_rate_limit(args.request_interval)
    lock = threading.Lock()
    accepted = 0
    with output.open("a", encoding="utf-8") as destination:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(annotate, row, args.model): row["source"] for row in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                with lock:
                    destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                    destination.flush()
                accepted += int(result["annotation"]["accepted"])
                print(
                    f"annotated {len(completed) + index}/{len(completed) + len(pending)}; "
                    f"new accepted {accepted}",
                    flush=True,
                )


if __name__ == "__main__":
    main()

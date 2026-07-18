"""Build a cleaner synthetic SFT dataset from the baseline model.

Pipeline:

1. make-prompts: create structured fable prompts.
2. generate: ask a teacher model for multiple candidate stories and keep only
   candidates that pass deterministic quality filters.
3. export: split accepted stories into train/valid JSONL files for SFT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from app.ollama_client import OllamaError, generate_meta
from app.prompt_en import SYSTEM_PROMPT_EN, build_fable_prompt


DEFAULT_INSTRUCTION = "Write a short English fable for children with a clear moral."
DEFAULT_ROOT = Path("experiments/sft_clean_5k")

CHARACTERS = [
    "a small turtle",
    "a clever fox",
    "a shy rabbit",
    "a young owl",
    "a hungry squirrel",
    "a proud rooster",
    "a lost firefly",
    "a little ant",
    "a gentle bear",
    "a curious kitten",
    "a young crane",
    "a spotted ladybug",
    "a patient snail",
    "a brave mouse",
    "a careful hedgehog",
    "a playful dolphin",
    "a quiet deer",
    "a busy bee",
    "a sleepy puppy",
    "a nervous duckling",
    "a lonely lamb",
    "a tiny frog",
    "a cheerful sparrow",
    "a stubborn goat",
    "a helpful raccoon",
]

SETTINGS = [
    "a quiet pond",
    "a forest market",
    "a school garden",
    "an old library tree",
    "a snowy park",
    "a sunny farmyard",
    "a dark meadow",
    "a picnic field",
    "a mountain village",
    "a quiet bakery",
    "a misty riverbank",
    "a vegetable garden",
    "a windy hill",
    "a warm barn",
    "a little seaside town",
    "a spring orchard",
    "a moonlit path",
    "a busy playground",
    "a green valley",
    "a small classroom",
]

CHALLENGES = [
    "wants to do everything alone",
    "is tempted to take more than needed",
    "is afraid to ask for help",
    "thinks being fast is more important than being careful",
    "forgets a promise to a friend",
    "judges another animal by appearance",
    "is too proud to listen",
    "tries to carry something too heavy",
    "hides a mistake instead of admitting it",
    "wants to win by being unfair",
    "feels too small to make a difference",
    "rushes before learning the right way",
    "keeps useful things instead of sharing",
    "gets lost after ignoring good advice",
    "speaks unkindly without thinking",
    "gives up when the task becomes difficult",
    "copies another animal instead of being honest",
    "does not want to wait for a turn",
    "is scared of trying something new",
    "boasts about a skill without practicing",
]

OUTCOMES = [
    "learns to work with friends",
    "chooses to share fairly",
    "asks for help and finds the way home",
    "slows down and succeeds safely",
    "keeps the promise and repairs the friendship",
    "discovers kindness in an unexpected friend",
    "listens carefully and learns something new",
    "gets help and finishes the task together",
    "admits the mistake and helps fix it",
    "chooses honesty and earns trust",
    "helps in a small but important way",
    "practices patiently and improves",
    "shares with someone who needs help",
    "follows friendly guidance back to safety",
    "apologizes and speaks kindly",
    "keeps trying and completes the work",
    "tells the truth and feels proud",
    "waits patiently and gets a fair chance",
    "tries carefully and gains confidence",
    "practices humbly and becomes better",
]

TEACHINGS = [
    "teamwork makes hard tasks easier",
    "sharing turns plenty into friendship",
    "asking for help is wise",
    "carefulness is better than rushing",
    "promises matter",
    "do not judge by appearances",
    "wisdom begins with humility",
    "big tasks become easier together",
    "responsibility follows mistakes",
    "honesty earns trust",
    "small helpers can do important work",
    "patience helps skill grow",
    "kindness makes hard days easier",
    "good advice can guide us home",
    "kind words can mend hurt feelings",
    "perseverance helps us finish",
    "truth is better than pretending",
    "patience makes fairness possible",
    "courage grows with careful steps",
    "practice matters more than boasting",
]

UNSAFE_PATTERNS = (
    r"\bkill\b",
    r"\bblood\b",
    r"\bweapon\b",
    r"\bwar\b",
    r"\bviolent\b",
    r"\bscary\b",
    r"\bhaunted\b",
    r"\bdeath\b",
    r"\bdie\b",
    r"\bromance\b",
    r"\bdating\b",
    r"\bmarry\b",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_prompt_id(row: dict) -> str:
    raw = "|".join(row[key] for key in ("character", "setting", "challenge", "outcome", "teaching"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def make_prompts(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    prompts: list[dict] = []
    seen: set[str] = set()
    while len(prompts) < count:
        row = {
            "character": rng.choice(CHARACTERS),
            "setting": rng.choice(SETTINGS),
            "challenge": rng.choice(CHALLENGES),
            "outcome": rng.choice(OUTCOMES),
            "teaching": rng.choice(TEACHINGS),
            "length": "short",
        }
        row["id"] = f"syn_{make_prompt_id(row)}"
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        prompts.append(row)
    return prompts


def story_key(story: str) -> str:
    normalized = re.sub(r"\s+", " ", story).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def sentence_lengths(story: str) -> list[int]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", story) if s.strip()]
    return [len(re.findall(r"[A-Za-z']+", sentence)) for sentence in sentences]


def quality_reasons(story: str, prompt: dict) -> list[str]:
    reasons: list[str] = []
    clean = story.strip()
    compact = re.sub(r"\s+", " ", clean).strip()
    lower = clean.lower()
    words = re.findall(r"[A-Za-z']+", compact)

    if len(words) < 110:
        reasons.append("too_short")
    if len(words) > 280:
        reasons.append("too_long")
    if "moral:" not in lower:
        reasons.append("missing_moral")
    elif not re.search(r"(?:^|\n)\s*Moral\s*:", story, flags=re.I):
        reasons.append("moral_not_final_line_style")
    if re.search(r"^\s*[-*]\s+", story, flags=re.M):
        reasons.append("contains_bullets")
    if any(re.search(pattern, lower) for pattern in UNSAFE_PATTERNS):
        reasons.append("unsafe_terms")
    if clean.count("\n\n") == 0 and len(words) > 220:
        reasons.append("long_single_paragraph")
    lengths = sentence_lengths(clean)
    if lengths and max(lengths) > 55:
        reasons.append("run_on_sentence")
    final_line = clean.splitlines()[-1].strip() if clean.splitlines() else ""
    if compact and not final_line.lower().startswith("moral:") and compact.rstrip("\"'”").rstrip()[-1] not in ".!?":
        reasons.append("incomplete_ending")

    content_checks = [
        ("character_missing", prompt["character"].split()[-1]),
        ("setting_missing", prompt["setting"].split()[-1]),
    ]
    for reason, keyword in content_checks:
        if keyword.lower() not in lower:
            reasons.append(reason)
    return reasons


def normalize_story(story: str, prompt: dict) -> str:
    clean = story.strip()
    if "moral:" not in clean.lower():
        clean = f"{clean.rstrip()}\n\nMoral: {prompt['teaching']}"
    clean = re.sub(
        r"(?im)^\s*Moral\s*:\s*$",
        f"Moral: {prompt['teaching']}",
        clean,
    )
    return clean.strip()


def to_sft_row(prompt: dict, story: str) -> dict:
    input_text = "\n".join(
        [
            f"Character: {prompt['character']}",
            f"Setting: {prompt['setting']}",
            f"Challenge: {prompt['challenge']}",
            f"Outcome: {prompt['outcome']}",
            f"Teaching: {prompt['teaching']}",
        ]
    )
    return {
        "instruction": DEFAULT_INSTRUCTION,
        "input": input_text,
        "output": story.strip(),
    }


def command_make_prompts(args: argparse.Namespace) -> int:
    rows = make_prompts(args.count, args.seed)
    write_jsonl(Path(args.out), rows)
    print(f"Wrote {args.out}: {len(rows)} prompts")
    return 0


def command_generate(args: argparse.Namespace) -> int:
    prompts = read_jsonl(Path(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]

    candidates_path = Path(args.candidates)
    accepted_path = Path(args.accepted)
    rejected_path = Path(args.rejected)

    existing_candidate_keys = {
        (row.get("prompt_id"), row.get("candidate_index"))
        for row in read_jsonl(candidates_path)
    }
    existing_story_keys = {story_key(row["output"]) for row in read_jsonl(accepted_path)}
    accepted_count = len(existing_story_keys)

    for prompt_index, prompt in enumerate(prompts, 1):
        for candidate_index in range(args.candidates_per_prompt):
            if accepted_count >= args.target:
                print(f"Reached target: {accepted_count} accepted rows")
                return 0
            key = (prompt["id"], candidate_index)
            if key in existing_candidate_keys:
                continue

            prompt_text = build_fable_prompt(
                prompt["character"],
                prompt["setting"],
                prompt["challenge"],
                prompt["outcome"],
                prompt["teaching"],
                "Keep it short and polished, about 140-220 words. Use 2-4 short paragraphs.",
            )
            prompt_text += f"\nThe final line must be exactly: Moral: {prompt['teaching']}"
            print(
                f"[{prompt_index}/{len(prompts)}] {prompt['id']} candidate {candidate_index + 1} "
                f"accepted={accepted_count}/{args.target}",
                flush=True,
            )
            try:
                result = generate_meta(
                    prompt=prompt_text,
                    system=SYSTEM_PROMPT_EN,
                    model=args.model,
                    num_predict=args.num_predict,
                    seed=args.seed + candidate_index,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repeat_penalty=args.repeat_penalty,
                )
                raw_story = result["text"].strip()
                story = normalize_story(raw_story, prompt)
                candidate_row = {
                    "prompt_id": prompt["id"],
                    "candidate_index": candidate_index,
                    "prompt": prompt,
                    "story": raw_story,
                    "normalized_story": story,
                    "teacher_model": args.model,
                    "meta": result,
                }
                append_jsonl(candidates_path, candidate_row)

                reasons = quality_reasons(story, prompt)
                sft_row = to_sft_row(prompt, story)
                sft_key = story_key(sft_row["output"])
                if reasons or sft_key in existing_story_keys:
                    append_jsonl(
                        rejected_path,
                        {
                            **candidate_row,
                            "filter_reasons": reasons or ["duplicate_story"],
                        },
                    )
                    continue

                append_jsonl(accepted_path, sft_row)
                existing_story_keys.add(sft_key)
                accepted_count += 1
            except OllamaError as exc:
                append_jsonl(
                    rejected_path,
                    {
                        "prompt_id": prompt["id"],
                        "candidate_index": candidate_index,
                        "prompt": prompt,
                        "teacher_model": args.model,
                        "error": str(exc),
                        "filter_reasons": ["generation_error"],
                    },
                )

    print(f"Accepted rows: {accepted_count}")
    return 0


def command_export(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.accepted))
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) < 10:
        raise SystemExit("Need at least 10 accepted rows to export a train/valid split.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_valid = max(1, round(len(rows) * args.valid_ratio))
    valid = rows[:n_valid]
    train = rows[n_valid:]

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    manifest = {
        "total": len(rows),
        "train": len(train),
        "valid": len(valid),
        "source": "synthetic_clean_teacher_model",
        "instruction": DEFAULT_INSTRUCTION,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir}: train={len(train)} valid={len(valid)}")
    return 0


def command_refilter(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.candidates))
    accepted_path = Path(args.accepted)
    rejected_path = Path(args.rejected)
    if accepted_path.exists() and not args.force:
        raise SystemExit(f"{accepted_path} exists. Pass --force to overwrite.")
    if rejected_path.exists() and not args.force:
        raise SystemExit(f"{rejected_path} exists. Pass --force to overwrite.")

    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_path.write_text("", encoding="utf-8")
    rejected_path.write_text("", encoding="utf-8")

    seen: set[str] = set()
    accepted = 0
    rejected = 0
    for row in rows:
        prompt = row["prompt"]
        story = normalize_story(row.get("story", ""), prompt)
        reasons = quality_reasons(story, prompt)
        sft_row = to_sft_row(prompt, story)
        key = story_key(sft_row["output"])
        if key in seen:
            reasons.append("duplicate_story")
        if reasons:
            append_jsonl(rejected_path, {**row, "normalized_story": story, "filter_reasons": reasons})
            rejected += 1
            continue
        append_jsonl(accepted_path, sft_row)
        seen.add(key)
        accepted += 1
    print(f"Refiltered {len(rows)} candidates: accepted={accepted} rejected={rejected}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make-prompts")
    make.add_argument("--count", type=int, default=1700)
    make.add_argument("--seed", type=int, default=42)
    make.add_argument("--out", default=str(DEFAULT_ROOT / "prompts.jsonl"))
    make.set_defaults(func=command_make_prompts)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--prompts", default=str(DEFAULT_ROOT / "prompts.jsonl"))
    generate.add_argument("--candidates", default=str(DEFAULT_ROOT / "candidates.jsonl"))
    generate.add_argument("--accepted", default=str(DEFAULT_ROOT / "accepted.jsonl"))
    generate.add_argument("--rejected", default=str(DEFAULT_ROOT / "rejected.jsonl"))
    generate.add_argument("--model", default="llama3.2:3b-instruct-fp16")
    generate.add_argument("--target", type=int, default=5000)
    generate.add_argument("--limit", type=int, default=0)
    generate.add_argument("--candidates-per-prompt", type=int, default=3)
    generate.add_argument("--num-predict", type=int, default=360)
    generate.add_argument("--seed", type=int, default=5410)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.add_argument("--repeat-penalty", type=float, default=1.2)
    generate.set_defaults(func=command_generate)

    export = subparsers.add_parser("export")
    export.add_argument("--accepted", default=str(DEFAULT_ROOT / "accepted.jsonl"))
    export.add_argument("--out-dir", default=str(DEFAULT_ROOT / "dataset"))
    export.add_argument("--limit", type=int, default=5000)
    export.add_argument("--valid-ratio", type=float, default=0.1)
    export.add_argument("--seed", type=int, default=42)
    export.set_defaults(func=command_export)

    refilter = subparsers.add_parser("refilter")
    refilter.add_argument("--candidates", default=str(DEFAULT_ROOT / "candidates.jsonl"))
    refilter.add_argument("--accepted", default=str(DEFAULT_ROOT / "accepted.jsonl"))
    refilter.add_argument("--rejected", default=str(DEFAULT_ROOT / "rejected.jsonl"))
    refilter.add_argument("--force", action="store_true")
    refilter.set_defaults(func=command_refilter)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()

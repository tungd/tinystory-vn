"""Batch scientific evaluation (ADR-0002): SLMs vs Qwen on held-out TF1 test.

Cite: Nadas et al. (2025), TF1-EN-3M, arXiv:2504.20605.

Pure functions (unit-tested): aggregate_axis_scores, conclude_by_rank, panel_agreement.
Integration functions (deferred to Phase P4 when models exist): build_summary, main.

Usage (Phase P4 only):
    python -m trieulh.scripts.eval_slm \\
        --test data/tf1/test.jsonl \\
        --out results/eval_summary.json \\
        --limit 100 \\
        [--perplexity-json results/perplexity.json] \\
        [--checkpoint-json results/checkpoint_curve.json] \\
        [--loss-json results/loss_log.json] \\
        [--params slm-10m=10 slm-30m=30 qwen3-4b=4000]
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

from app import judge, ollama_client
from app.agreement import cohen_kappa_weighted, kendall_tau
from app.metrics import distinct_n, flesch_reading_ease, self_bleu

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

AXES = ["grammar", "creativity", "moral_clarity", "prompt_adherence"]

DEFAULT_MODELS: list[str] = ["slm-10m", "slm-30m", "qwen3-4b"]
DEFAULT_JUDGES: list[str] = ["qwen3:4b", "gemma2:2b", "llama3.2:3b"]
DEFAULT_PARAMS_M: dict[str, int] = {"slm-10m": 10, "slm-30m": 30, "qwen3-4b": 4000}

# System prompt used for story generation
GENERATION_SYSTEM = (
    "You are a creative fable author. Write engaging, morally instructive fables "
    "suitable for children. Keep each story focused and complete."
)

STORY_SEPARATOR = "<|story|>"
END_TOKEN = "<|end|>"

# ─── Pure functions (unit-tested) ─────────────────────────────────────────────


def aggregate_axis_scores(panel: dict[str, list[dict]]) -> dict:
    """Mean score per axis (+ overall) across all judges and all stories.

    Args:
        panel: mapping judge_name -> list of score dicts
                (each dict has grammar, creativity, moral_clarity, prompt_adherence, overall).

    Returns:
        dict with keys = AXES + ["overall"], values = rounded means.
    """
    rows = [r for judge_rows in panel.values() for r in judge_rows]
    if not rows:
        return {a: 0.0 for a in AXES} | {"overall": 0.0}
    out = {a: round(sum(r[a] for r in rows) / len(rows), 3) for a in AXES}
    out["overall"] = round(sum(out[a] for a in AXES) / len(AXES), 3)
    return out


def panel_agreement(overall_by_judge: dict[str, list[float]]) -> dict:
    """Inter-judge agreement averaged over all judge pairs on overall scores.

    Args:
        overall_by_judge: mapping judge_name -> list[float] overall scores (one per story).

    Returns:
        {"cohen_kappa": float, "kendall_tau": float}
    """
    judges = list(overall_by_judge)
    kappas: list[float] = []
    taus: list[float] = []
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            a = [int(round(x)) for x in overall_by_judge[judges[i]]]
            b = [int(round(x)) for x in overall_by_judge[judges[j]]]
            kappas.append(cohen_kappa_weighted(a, b))
            taus.append(kendall_tau(overall_by_judge[judges[i]], overall_by_judge[judges[j]]))

    def avg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {"cohen_kappa": avg(kappas), "kendall_tau": avg(taus)}


def conclude_by_rank(model_overall: dict[str, float]) -> dict:
    """Produce conclusion dict by ranking models on their overall score.

    Args:
        model_overall: mapping model_id -> mean overall score.

    Returns:
        {"winner": str, "by_rank": str, "notes": str}
    """
    ranked = sorted(model_overall.items(), key=lambda kv: kv[1], reverse=True)
    winner = ranked[0][0]
    order = " > ".join(f"{m} ({s:.2f})" for m, s in ranked)
    return {
        "winner": winner,
        "by_rank": f"By rank: {order}",
        "notes": "Conclusion by rank across judges, not absolute single-judge scores.",
    }


# ─── Integration helpers ───────────────────────────────────────────────────────


def _resolve_model_tag(model_id: str) -> str:
    """Resolve a model_id to its Ollama tag, falling back to the id itself."""
    try:
        from app import models_registry
        return models_registry.resolve_ollama(model_id)
    except KeyError:
        return model_id


def _load_test_prompts(path: Path, limit: int) -> list[dict]:
    """Load held-out prompts from a JSONL file (TF1 format).

    Each line must have {"text", "cond_len"}.
    The conditioning prompt = text[:cond_len].rstrip("\\n").
    The reference fable = text after <|story|> separator up to <|end|>.

    Returns list of {"prompt": str, "reference": str}.
    """
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text: str = obj["text"]
            cond_len: int = obj["cond_len"]
            prompt = text[:cond_len].rstrip("\n")
            # Extract reference: part after <|story|> up to <|end|>
            story_start = text.find(STORY_SEPARATOR)
            if story_start != -1:
                story_part = text[story_start + len(STORY_SEPARATOR):]
                end_pos = story_part.find(END_TOKEN)
                reference = story_part[:end_pos].strip() if end_pos != -1 else story_part.strip()
            else:
                reference = ""
            records.append({"prompt": prompt, "reference": reference})
            if len(records) >= limit:
                break
    return records


def _compute_objective(
    model_id: str,
    generations: list[str],
    perplexity_map: dict[str, float] | None,
) -> dict:
    """Compute objective metrics for one model's set of generations.

    Perplexity is included only if present in perplexity_map (not fabricated).
    """
    obj: dict = {
        "distinct_1": round(distinct_n(generations, 1), 4),
        "distinct_2": round(distinct_n(generations, 2), 4),
        "self_bleu": round(self_bleu(generations), 4),
        "flesch_reading_ease": round(
            statistics.mean(flesch_reading_ease(g) for g in generations) if generations else 0.0,
            2,
        ),
    }
    if perplexity_map and model_id in perplexity_map:
        obj["perplexity"] = perplexity_map[model_id]
    return obj


def build_summary(
    *,
    models: list[str],
    judges: list[str],
    test_prompts: list[dict],
    params_m: dict[str, int],
    perplexity_map: dict[str, float] | None = None,
    checkpoint_data: dict | None = None,
    loss_data: list | None = None,
    gen_fn=None,
    judge_fn=None,
) -> dict:
    """Generate full eval_summary structure.

    Args:
        models: list of model IDs to evaluate (e.g. ["slm-10m","slm-30m","qwen3-4b"]).
        judges: list of judge Ollama model tags.
        test_prompts: list of {"prompt": str, "reference": str} from _load_test_prompts.
        params_m: mapping model_id -> parameter count in millions.
        perplexity_map: optional mapping model_id -> float (from --perplexity-json).
        checkpoint_data: optional dict model_id -> list[{"step":int,"overall":float}].
        loss_data: optional list[{"step":int,"loss":float}].
        gen_fn: callable matching ollama_client.generate signature (injectable for testing).
        judge_fn: callable matching judge.evaluate signature (injectable for testing).

    Returns:
        Full eval_summary dict matching the ADR-0002 schema.
    """
    gen_fn = gen_fn or ollama_client.generate
    judge_fn = judge_fn or judge.evaluate

    logger.info("Starting eval: %d models x %d judges x %d prompts",
                len(models), len(judges), len(test_prompts))

    # ── Generate stories for each model ───────────────────────────────────────
    # generations[model_id] = list[str] (one story per prompt)
    generations: dict[str, list[str]] = {}
    for model_id in models:
        ollama_tag = _resolve_model_tag(model_id)
        logger.info("Generating with model %s (tag: %s)", model_id, ollama_tag)
        stories: list[str] = []
        for item in test_prompts:
            try:
                story = gen_fn(
                    prompt=item["prompt"],
                    system=GENERATION_SYSTEM,
                    model=ollama_tag,
                )
            except Exception as exc:
                logger.warning("Generation failed for %s: %s", model_id, exc)
                story = ""
            stories.append(story)
        generations[model_id] = stories

    # ── Score each (model, story, judge) triplet ───────────────────────────────
    # raw_scores[model_id][judge_tag] = list[dict] (one score dict per story)
    raw_scores: dict[str, dict[str, list[dict]]] = {m: {j: [] for j in judges} for m in models}
    for model_id in models:
        for judge_tag in judges:
            logger.info("Judging %s with %s", model_id, judge_tag)
            for story, item in zip(generations[model_id], test_prompts):
                try:
                    scores = judge_fn(story, item["prompt"], model=judge_tag)
                except Exception as exc:
                    logger.warning("Judge %s failed for %s: %s", judge_tag, model_id, exc)
                    scores = {a: 0 for a in AXES} | {"overall": 0.0, "rationale": {}}
                raw_scores[model_id][judge_tag].append(scores)

    # ── Aggregate per model ────────────────────────────────────────────────────
    # judge_panel[model_id] = aggregated axis scores across all judges
    judge_panel_agg: dict[str, dict] = {}
    model_overall: dict[str, float] = {}

    for model_id in models:
        agg = aggregate_axis_scores(raw_scores[model_id])
        judge_panel_agg[model_id] = agg
        model_overall[model_id] = agg["overall"]

    # ── Panel agreement ────────────────────────────────────────────────────────
    # Build per-judge overall score lists across all models x stories
    overall_by_judge: dict[str, list[float]] = {j: [] for j in judges}
    for model_id in models:
        for judge_tag in judges:
            for s in raw_scores[model_id][judge_tag]:
                overall_by_judge[judge_tag].append(float(s.get("overall", 0.0)))

    agreement = panel_agreement(overall_by_judge)

    # ── Objective metrics ──────────────────────────────────────────────────────
    objective: dict[str, dict] = {}
    for model_id in models:
        gens = [g for g in generations[model_id] if g]
        objective[model_id] = _compute_objective(model_id, gens, perplexity_map)

    # ── Size ladder ───────────────────────────────────────────────────────────
    size_ladder = [
        {"model": m, "params_m": params_m.get(m, 0), "overall": model_overall.get(m, 0.0)}
        for m in models
        if m in params_m
    ]
    size_ladder.sort(key=lambda x: x["params_m"])

    # ── Conclusion ────────────────────────────────────────────────────────────
    conclusion = conclude_by_rank(model_overall)

    # ── Assemble summary ──────────────────────────────────────────────────────
    summary: dict = {
        "models": models,
        "objective": objective,
        "judge_panel": {"judges": judges} | judge_panel_agg,
        "agreement": agreement,
        "conclusion": conclusion,
        "size_ladder": size_ladder,
    }

    # Optional sections — only included when input data is provided
    if checkpoint_data:
        summary["checkpoint_curve"] = checkpoint_data

    if loss_data:
        summary["loss_curve"] = loss_data

    return summary


# ─── CLI entrypoint (integration, deferred to Phase P4) ──────────────────────


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for batch eval (Phase P4 — requires trained models via Ollama).

    NOT run during unit tests. Deferred until slm-10m, slm-30m, and qwen3-4b
    are available as Ollama models.
    """
    parser = argparse.ArgumentParser(
        description="Batch scientific eval: SLMs vs Qwen on held-out TF1 test (ADR-0002)."
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/tf1/test.jsonl"),
        help="Path to held-out test JSONL (each line: {text, cond_len}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/eval_summary.json"),
        help="Output path for eval_summary.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of test prompts to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model IDs to evaluate (space-separated).",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=DEFAULT_JUDGES,
        help="Ollama judge model tags (space-separated).",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=[],
        metavar="MODEL=PARAMS_M",
        help="Parameter counts in millions, e.g. slm-10m=10 slm-30m=30 qwen3-4b=4000.",
    )
    parser.add_argument(
        "--perplexity-json",
        type=Path,
        default=None,
        help="Optional JSON file mapping model_id -> perplexity float.",
    )
    parser.add_argument(
        "--checkpoint-json",
        type=Path,
        default=None,
        help="Optional JSON file with checkpoint curve data (model_id -> list[{step,overall}]).",
    )
    parser.add_argument(
        "--loss-json",
        type=Path,
        default=None,
        help="Optional JSON file with loss curve data (list[{step,loss}]).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Parse --params overrides
    params_m: dict[str, int] = dict(DEFAULT_PARAMS_M)
    for token in args.params:
        if "=" in token:
            k, v = token.split("=", 1)
            params_m[k.strip()] = int(v.strip())

    # Load optional JSON files
    perplexity_map: dict[str, float] | None = None
    if args.perplexity_json and args.perplexity_json.exists():
        perplexity_map = json.loads(args.perplexity_json.read_text())

    checkpoint_data: dict | None = None
    if args.checkpoint_json and args.checkpoint_json.exists():
        checkpoint_data = json.loads(args.checkpoint_json.read_text())

    loss_data: list | None = None
    if args.loss_json and args.loss_json.exists():
        loss_data = json.loads(args.loss_json.read_text())

    # Load test prompts
    logger.info("Loading test prompts from %s (limit=%d)", args.test, args.limit)
    test_prompts = _load_test_prompts(args.test, args.limit)
    logger.info("Loaded %d test prompts", len(test_prompts))

    # Run full evaluation
    summary = build_summary(
        models=args.models,
        judges=args.judges,
        test_prompts=test_prompts,
        params_m=params_m,
        perplexity_map=perplexity_map,
        checkpoint_data=checkpoint_data,
        loss_data=loss_data,
    )

    # Write output — ensure parent directory exists
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote eval_summary to %s", out_path)
    print(f"Done. Winner: {summary['conclusion']['winner']}")
    print(f"Ranking: {summary['conclusion']['by_rank']}")


if __name__ == "__main__":
    main()

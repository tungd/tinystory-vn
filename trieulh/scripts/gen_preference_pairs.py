"""Sinh preference pairs (RLAIF) cho ORPO từ SLM qua Ollama local.

Với mỗi prompt: sinh 2 truyện từ model SLM (temp 0.8, seed khác nhau), judge
local chấm rubric 4 trục (app.judge, theo TF1-EN-3M), giữ pair khi chênh
overall >= min-margin. Resume-safe: prompt đã có trong file output sẽ bỏ qua.

Usage (T4 của plan ORPO, chạy local qua đêm):
    python -m trieulh.scripts.gen_preference_pairs \
        --prompts data/orpo/prompts.jsonl \
        --out data/orpo/pairs.jsonl \
        --model slm-30m-p2 --judge qwen3:4b --min-margin 1.0

Spec: docs/superpowers/specs/2026-07-11-slm-orpo-alignment-design.md
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def make_pair(prompt: str, story_a: str, story_b: str,
              scores_a: dict, scores_b: dict,
              min_margin: float = 1.0) -> dict | None:
    """Ghép 2 truyện + điểm judge thành 1 preference pair, hoặc None nếu loại.

    Loại khi: 2 truyện trùng nhau, judge lỗi (overall 0), hoặc chênh lệch
    overall < min_margin (pair mơ hồ chỉ thêm nhiễu cho ORPO).
    """
    oa = float(scores_a.get("overall", 0.0))
    ob = float(scores_b.get("overall", 0.0))
    if story_a.strip() == story_b.strip():
        return None
    if oa <= 0.0 or ob <= 0.0:
        return None
    if abs(oa - ob) < min_margin:
        return None
    if ob > oa:
        chosen, rejected, oc, orj = story_b, story_a, ob, oa
    else:
        chosen, rejected, oc, orj = story_a, story_b, oa, ob
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "score_chosen": oc, "score_rejected": orj}


def _generate(prompt: str, model: str, seed: int) -> str:
    """Sinh 1 truyện qua Ollama (Modelfile TEMPLATE tự thêm <|story|>)."""
    import urllib.request
    req = json.dumps({"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.8, "top_p": 0.9,
                                  "repeat_penalty": 1.1, "seed": seed}}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        "http://localhost:11434/api/generate", req,
        {"Content-Type": "application/json"}), timeout=180)
    return json.loads(r.read())["response"].strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/orpo/prompts.jsonl")
    ap.add_argument("--out", default="data/orpo/pairs.jsonl")
    ap.add_argument("--model", default="slm-30m-p2")
    ap.add_argument("--judge", default="qwen3:4b")
    ap.add_argument("--min-margin", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="Dừng khi tổng số pairs GIỮ (kể cả từ lần trước) đạt mức này. Resume-safe.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from app import judge as judge_mod

    prompts = [json.loads(l)["prompt"] for l in open(args.prompts)]
    if args.limit:
        prompts = prompts[:args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    kept_existing = 0                    # số pairs đã giữ từ các lần chạy trước (để resume tới max-pairs)
    if out.exists():                     # resume: bỏ qua prompt đã xử lý
        for l in out.open():
            d = json.loads(l)
            done.add(d["prompt"])
            if not d.get("filtered"):
                kept_existing += 1
    logger.info("prompts=%d, đã xong=%d, pairs sẵn có=%d", len(prompts), len(done), kept_existing)
    if args.max_pairs and kept_existing >= args.max_pairs:
        logger.info("đã đủ %d pairs (>= max %d) - không cần chạy thêm", kept_existing, args.max_pairs)
        return

    kept = skipped = 0
    with out.open("a") as f:
        for i, prompt in enumerate(prompts):
            if prompt in done:
                continue
            if args.max_pairs and kept_existing + kept >= args.max_pairs:
                logger.info("ĐẠT MỐC %d pairs - dừng.", kept_existing + kept)
                break
            try:
                a = _generate(prompt, args.model, seed=11)
                b = _generate(prompt, args.model, seed=97)
                sa = judge_mod.evaluate(a, prompt, model=args.judge)
                sb = judge_mod.evaluate(b, prompt, model=args.judge)
            except Exception as exc:
                logger.warning("prompt %d lỗi: %s", i, exc)
                continue
            pair = make_pair(prompt, a, b, sa, sb, args.min_margin)
            if pair is None:
                skipped += 1
                # vẫn ghi marker để resume không chấm lại prompt bị loại
                f.write(json.dumps({"prompt": prompt, "filtered": True},
                                   ensure_ascii=False) + "\n")
            else:
                kept += 1
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            f.flush()
            if (kept + skipped) % 25 == 0:
                logger.info("tiến độ: %d chấm | %d giữ | %d loại",
                            kept + skipped, kept, skipped)
    logger.info("XONG: giữ %d pairs, loại %d (file: %s)", kept, skipped, out)


if __name__ == "__main__":
    main()

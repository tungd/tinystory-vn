# ORPO Alignment Implementation Plan

> **For agentic workers:** thực thi inline trong session (executing-plans style); T1-T3 làm được ngay, T4-T6 phụ thuộc model p2 (user chạy notebook tối 2026-07-11).

**Goal:** Chương alignment: ORPO trên preference pairs RLAIF cho SLM 30M-p2, đo delta từng nấc.

**Architecture:** pair-gen + judge chạy local (Ollama); ORPO train trên Colab T4 (TRL); eval cuối bằng panel 3 judge --limit 15. Spec: `docs/superpowers/specs/2026-07-11-slm-orpo-alignment-design.md`.

**Tech Stack:** Ollama API, app.judge (rubric TF1), TRL ORPOTrainer, transformers.

## Global Constraints
- Prompt ORPO lấy từ TF1 split test, OFFSET sau 500 record đầu (giữ nguyên bộ eval).
- Lọc pair: chênh overall >= 1.0. Format pair đúng template train: `cond + "\n" + <|story|>` prompt, completion `story + <|end|>`.
- Artifact hậu tố `-orpo`, không đè p2. Không emoji, không em dash.
- Guard hồi quy: perplexity held-out sau ORPO tăng <= 10% so p2.

### T1: Bộ prompt ORPO (data/orpo/prompts.jsonl, 1500 prompt, offset 500) — làm ngay
### T2: scripts/gen_preference_pairs.py (TDD hàm lọc/format; gen+judge mockable, resume-safe) — làm ngay
### T3: scripts/orpo_train.py cho Colab (TRL ORPOTrainer, load 30M-p2, LR 1e-5, 2 epoch, save 30M-orpo + GGUF) — làm ngay, compile-check
### T4: (sau pha 2, local) chạy gen_preference_pairs qua đêm -> data/orpo/pairs.jsonl (>= 800 pairs)
### T5: (Colab) chạy orpo_train -> 30M-orpo, GGUF, ollama create slm-30m-orpo
### T6: đánh giá: ppl guard + định tính 10 prompt + batch eval panel --limit 15 -> eval_summary.json -> Results tab; cập nhật training log

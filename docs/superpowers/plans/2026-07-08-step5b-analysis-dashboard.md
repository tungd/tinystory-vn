# Step 5B Analysis Dashboard Implementation Plan

> **For agentic workers:** notebook-only change on the Drive copy `1SUsz...` via colab-mcp; edit cells then Run all.

**Goal:** Thay phần Step 5b của notebook 30M bằng dashboard phân tích A+B+C có baseline fable thật, render inline trong run.

**Architecture:** 1 cell harness `collect_analysis()` -> dict + dump JSON Drive; 3 cell vẽ figure matplotlib (`fig_training`, `fig_quality`, `fig_lm_behavior`). Dùng lại `app/metrics.py` + `app/perplexity.py`.

**Tech Stack:** transformers 5.x, torch, matplotlib, textstat, colab-mcp.

## Global Constraints
- KHÔNG emoji, KHÔNG em dash trong mọi text/nhãn.
- Nhãn biểu đồ tiếng Anh.
- Dùng lại metric có sẵn: `distinct_n`, `self_bleu`, `flesch_reading_ease`, `aggregate_nll`, `perplexity_from_nll`.
- `SEP="<|story|>"`, `END="<|end|>"` khớp train/serve.
- Tham số ở đầu harness: `N_GEN=30`, `N_PPL=200`, temp 0.8, max_new_tokens 320.
- Không phá train recipe (TRAIN_N 400k, STEPS 3000, ARCH 30M).

---

### Task 1: Harness cell `collect_analysis()`
- Xoá cell curves cũ (`8cwDqYD0vvDm`) + cell application-analysis cũ (`Un02r9iTvyGh`).
- Thêm cell: setup (`pip install textstat`; `sys.path` repo; import metrics), đọc `trainer30.state.log_history`, load `out/30M`, sinh `N_GEN` fable từ test prompt, đọc `N_GEN`/`N_PPL` fable thật, tính B, forward perplexity + per-position loss + Zipf counts. Trả `analysis` + dump `{DRIVE}/analysis_30M.json`.
- Guard thiếu artifact -> phần đó `None`.

### Task 2: `fig_training(analysis)` cell
- 2x2: train loss; loss-gain; LR; throughput. hline mốc v1 loss ~1.8. Lưu `{DRIVE}/fig_training_30M.png`, `plt.show()`.

### Task 3: `fig_quality(analysis)` cell
- 2x2: Distinct-1/2 (SLM vs real); Self-BLEU (SLM vs real); Flesch (SLM vs real); histogram độ dài chồng. Lưu `fig_quality_30M.png`, show.

### Task 4: `fig_lm_behavior(analysis)` cell
- 1x3: perplexity bar (SLM, mốc v1 nếu có); per-position mean loss; Zipf log-log (SLM output vs real vocab). Lưu `fig_lm_30M.png`, show.

### Task 5: Cập nhật markdown Step 5b + Run all
- Sửa markdown mô tả 3 figure.
- Reconnect colab; Run all: prepare data -> train 30M -> harness -> 3 figure -> export GGUF Drive. Monitor tới khi model + GGUF trên Drive.

# Design: Step 5B — Application Analysis Dashboard (notebook drive/1SUsz)

Ngày: 2026-07-08
Trạng thái: Approved
Branch: `feat/slm-pretrain-tf1`
Ngữ cảnh: mở rộng phần "Step 5b" của notebook 30M (bản Drive `1SUszrhTVm2bP-Lx3af1xEAfkb9_9oxRA`) từ 2 biểu đồ đơn giản thành một dashboard phân tích ứng dụng đầy đủ, hiển thị trực quan inline ngay trong lần chạy.

## 1. Mục tiêu

Sau khi train xong SLM 30M, cung cấp bộ biểu đồ + số liệu để **quan sát và biện luận khoa học** chất lượng model, bám nền tảng môn IT5410 (Week4 Transformer, Week6 autoregressive MLE/perplexity) và literature đánh giá intrinsic reference-free (Zhu 2018 Self-BLEU; Li 2016 distinct-n; open-ended LM eval). Tất cả render inline (matplotlib) trong run, không phụ thuộc dịch vụ ngoài (không LLM-judge trong notebook).

### Non-goals (YAGNI)
- KHÔNG attention heatmap / scaffold-adherence (nhóm D đã loại: cần `output_attentions`, nặng, phức tạp).
- KHÔNG LLM-as-judge trong notebook (đã có `scripts/eval_slm.py` cho batch judge ngoài).
- KHÔNG đổi kiến trúc/train recipe; chỉ thêm phần phân tích sau train.

## 2. Phạm vi đã chốt (grill 2026-07-08)

| Chủ đề | Quyết định |
|---|---|
| Nhóm phân tích | **A (training dynamics) + B (intrinsic quality) + C (LM behavior)** |
| Baseline tham chiếu | **Có** — so fable SLM sinh vs fable thật held-out (`data/tf1/test.jsonl`) |
| Số mẫu | `N_GEN=30` generation; `N_PPL=200` doc test; length/Zipf trên 30 cặp SLM-vs-thật |
| Sampling | temp 0.8 (khớp default app) khi sinh fable đánh giá |
| Metric | dùng lại `app/metrics.py` (distinct_n/self_bleu/flesch_reading_ease) + `app/perplexity.py` (aggregate_nll/perplexity_from_nll) |

## 3. Kiến trúc

Ba đơn vị tách biệt, giao tiếp qua một dict `analysis` + file JSON trên Drive:

```
trainer30.state.log_history ─┐
out/30M (model+tokenizer) ───┼─► collect_analysis() ─► analysis(dict) ─► {DRIVE}/analysis_30M.json
data/tf1/test.jsonl ─────────┘                                          │
                                                    ┌───────────────────┼───────────────────┐
                                              fig_training(analysis) fig_quality(analysis) fig_lm_behavior(analysis)
```

- **Cell 1 — harness** `collect_analysis()`: thu thập mọi số liệu (train log, sinh 30 fable, đọc 30/200 fable thật, tính metric B, forward perplexity + per-position loss + đếm token cho Zipf). Trả về `analysis` và **dump `{DRIVE}/analysis_30M.json`** (survive recycle). Không vẽ.
- **Cell 2 — `fig_training(analysis)`**: 1 figure 2x2 (train loss; loss-gain; LR; throughput). `plt.show()`.
- **Cell 3 — `fig_quality(analysis)`**: 1 figure 2x2 (Distinct-1/2; Self-BLEU; Flesch; histogram độ dài) — mỗi bar/hist đặt SLM cạnh fable thật.
- **Cell 4 — `fig_lm_behavior(analysis)`**: 1 figure 1x3 (perplexity SLM vs mốc v1; per-position mean loss; Zipf log-log SLM output vs vocab thật).

Lý do tách harness khỏi vẽ: nếu runtime recycle sau khi đã collect, chỉ cần load lại JSON và gọi 3 hàm vẽ, không phải sinh/forward lại.

## 4. Chi tiết số liệu

### 4.1 Nhóm A — Training dynamics (từ `trainer30.state.log_history`)
- `loss` theo `step` (điểm có key `loss`).
- `loss_gain[t] = loss[0] - loss[t]` (đường "reward gain" của pretraining — cải thiện so với baseline đầu run).
- `learning_rate` theo step (thể hiện dạng WSD warmup-stable-decay).
- throughput: `train_samples_per_second` nếu có trong log cuối, hoặc suy từ runtime; in dạng bar/annotate.
- Vẽ đường mốc v1 loss ~1.8 (hline) để so.

### 4.2 Nhóm B — Intrinsic quality (dùng `app/metrics.py`)
- Sinh `N_GEN=30` fable từ 30 prompt held-out (`cond + "\n" + SEP`, temp 0.8, `max_new_tokens` ~ 320). Cắt tại `END`.
- Lấy `N_GEN` fable thật khớp từ `test.jsonl` (trường `fable`).
- `distinct_n(gen, 1)`, `distinct_n(gen, 2)` vs `distinct_n(real, 1/2)`.
- `self_bleu(gen, 4)` vs `self_bleu(real, 4)`.
- Flesch: trung bình `flesch_reading_ease(t)` trên gen vs real.
- Histogram độ dài (số từ) gen vs real, chồng bán trong suốt.

### 4.3 Nhóm C — LM behavior (forward trên test)
- **Perplexity**: forward `N_PPL=200` doc test có loss-mask conditioning; cộng NLL phần truyện qua `aggregate_nll`, ra `perplexity_from_nll(total_nll, total_tokens)`. Bar so mốc v1 (nếu có `perplexity_v1`, else chỉ hiện 30M).
- **Per-position mean loss**: CE trung bình theo vị trí token truyện (bin ~64 vị trí đầu) — thấy model tự tin ở đầu/cuối.
- **Zipf**: đếm tần suất token trên output SLM (30 gen) vs vocab fable thật (30 real); vẽ log-log rank-frequency 2 đường.

## 5. Error handling
- Mỗi `fig_*` bọc try/except: thiếu key trong `analysis` thì in cảnh báo + skip subplot đó, không vỡ cell.
- `collect_analysis` kiểm tra tồn tại `out/30M`, `data/tf1/test.jsonl`, `trainer30`; thiếu cái nào thì set phần đó là `None` và ghi chú trong dict.
- `textstat` cần cài (`pip install textstat`); repo root thêm vào `sys.path` để `from app.metrics import ...`.

## 6. Quy ước trình bày
- KHÔNG emoji, KHÔNG em dash trong text/nhãn (theo quy ước app).
- Nhãn biểu đồ tiếng Anh (khớp domain fable tiếng Anh).
- Lưu mỗi figure ra `{DRIVE}/fig_training_30M.png` / `fig_quality_30M.png` / `fig_lm_30M.png` để dùng lại trong báo cáo.

## 7. Deliverables
- 4 cell mới trong section Step 5b của notebook Drive (thay cell curves + application-analysis cũ).
- `analysis_30M.json` + 3 PNG trên Drive sau khi chạy.

## 8. Tiêu chí thành công
- Chạy Run all: sau train, 3 figure hiển thị inline không lỗi.
- Số liệu B/C đặt SLM cạnh fable thật cho thấy khoảng cách rõ ràng.
- Artifact JSON + PNG nằm trên Drive để tái dùng.

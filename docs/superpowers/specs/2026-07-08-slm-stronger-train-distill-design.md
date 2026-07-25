# Design: Stronger SLM training (full-Chinchilla 30M) + distillation to 10M

Ngày: 2026-07-08
Trạng thái: Approved (chờ review spec)
Branch: `feat/slm-pretrain-tf1`
Tiền thân: `docs/superpowers/specs/2026-07-08-slm-pretrain-tf1-design.md` (v1). Bản này (v2) SỬA cách train cho mạnh hơn, đúng scaling law.

## 1. Mục tiêu & cơ sở khoa học

Lần train v1 (reduced, 150k, 900 step ≈ ~50M token cho 30M) bị **undertrain nặng** (~1.7 token/tham số so với Chinchilla 20×) → SLM sinh fable mạch lạc nhưng thua Qwen-4B xa trên judge (2.5 vs 9.75). Bản v2 khắc phục bằng cách **train đúng ngân sách token theo scaling law**, rồi **chưng cất (distillation)** xuống 10M để củng cố luận điểm "model nhỏ vẫn tốt".

Cơ sở (bám bài giảng môn IT5410 + literature):
- **Autoregressive LM = MLE trên chain-rule** `log p(x)=Σ log p(xᵢ|x_<ᵢ)` (Materials Week6). Loss-mask conditioning = chỉ tối đa hoá log-likelihood phần truyện.
- **Scaling laws (Kaplan et al. 2020)** (Materials Week4): test loss giảm power-law khi tăng đồng thời N (tham số), D (token xử lý), C (compute). Sửa undertrain = tăng D tới mức tương xứng N.
- **Chinchilla (Hoffmann et al. 2022)**: compute-optimal ≈ **20 token/tham số** → 30M ≈ 600M token.
- **Data-constrained scaling (Muennighoff et al. 2023, NeurIPS)**: lặp dữ liệu **≤4 epoch ≈ tương đương data mới**; sau ~16 epoch mới bão hoà → được phép repeat subset tới 4 epoch.
- **Knowledge Distillation (Hinton et al. 2015)**: student học phân bố "mềm" của teacher (KL trên logits) → student nhỏ vượt xa bản from-scratch cùng kích thước.
- Đánh giá bám **ADR-0002**; dataset + eval methodology: Nadas et al. (2025), TF1-EN-3M, arXiv:2504.20605.

### Non-goals (YAGNI)
- KHÔNG train model > 30M (giữ tinh thần SLM; 50M/100M đã cân nhắc và loại vì tốn compute).
- KHÔNG dùng Colab Pro/GPU trả phí (chốt free T4 + resume).
- KHÔNG thay kiến trúc/tokenizer/pipeline dữ liệu cốt lõi của v1 (tái dùng `scripts/tf1_pretrain/*`, tokenizer BPE 12k, format conditional + slot dropout).
- KHÔNG dùng toàn bộ 2.8M train (dùng subset ~600k).

## 2. Quyết định đã chốt (grill 2026-07-08)

| Chủ đề | Quyết định |
|---|---|
| Phạm vi | **Teacher 30M (from scratch, full Chinchilla) → distill xuống student 10M** |
| Ngân sách teacher | **~600M token** (Chinchilla 20× cho 30M) |
| Dữ liệu | **subset unique ~600k**, dedup + lọc chất lượng, **epoch ≤4** |
| Distillation | token-level KD từ teacher 30M: `α·KL(student‖teacher,T)·T² + (1-α)·CE`, T≈2, α≈0.5, chỉ trên token truyện |
| Ngân sách student | ~200M token (~1.5 epoch) |
| Compute | **Colab free T4 + checkpoint/resume Drive** (`slm_tf1/`) |
| Đánh giá | batch judge panel 3 model khác họ (qwen3:4b + gemma2:2b + llama3.2:3b) + κ/τ (ADR-0002) |

## 3. Dữ liệu (v2)

Tái dùng `scripts/prepare_tf1_pretrain.py` (parse slot, slot dropout, format conditional, dedup `prompt_hash`) với **bổ sung lọc chất lượng** và subset lớn hơn:
- **Subset ~600k** record unique (từ split `train` 2.8M).
- **Lọc chất lượng**: giữ fable có độ dài hợp lệ (vd 60–320 từ — tránh cụt/lan man), loại record `fable` rỗng/trùng. Thêm tham số CLI (vd `--min-words`, `--max-words`) — mở rộng hàm `build_record`/`_write_split` sẵn có, không phá interface cũ.
- 600k × ~250 tok ≈ **150M token/epoch**; held-out test (dataset split `test`) cho perplexity + batch eval.
- Tokenizer: **giữ nguyên** artifact BPE 12k v1 (hoặc train lại trên subset mới — không bắt buộc; nếu train lại thì đúng quy trình `scripts/train_tokenizer.py`).

## 4. Teacher 30M — from scratch, full Chinchilla

Kiến trúc **không đổi** (Llama-style): 8 layers, hidden 512, 8 heads, GQA 2 kv, SwiGLU, RoPE, tied embeddings, vocab ~12k, seq 512.

Recipe:
- **Token budget ~600M = 4 epoch × 150M** (Chinchilla 20×; ≤4 epoch theo Muennighoff).
- **Steps ~7,900**: `per_device_batch=32 × grad_accum=8 = 256 seq/step` (~76k token/step). (Cân lại theo token thực tế; miễn tổng ≈ 600M token.)
- Optimizer: AdamW β=(0.9, 0.95), weight_decay=0.1, grad-clip=1.0.
- **WSD** LR schedule: warmup ~2% bước → stable → decay 20% cuối. Peak LR ~**2e-3** (hạ nhẹ so với 3e-3 của v1 cho ổn định ở run dài; tinh chỉnh nếu loss không giảm).
- fp16 (T4); loss-mask phần conditioning; log loss định kỳ (`logging_steps` nhỏ để theo dõi đường cong).
- **Checkpoint định kỳ vào Drive** (vd mỗi ~10% bước) + **resume** — bắt buộc vì run dài dễ bị recycle. Lưu step-0 làm baseline "before".
- Kỳ vọng: val loss tiến về ~1.3–1.5 (thấp hơn ~1.8 của v1), chất lượng judge cao hơn rõ.

## 5. Student 10M — distillation từ teacher 30M

Kiến trúc student **giữ như v1 10M**: 6 layers, hidden 320, GQA 2 kv, cùng tokenizer + seq 512.

Distillation:
- Teacher = model 30M đã train (đóng băng, `eval()`), forward mỗi batch để lấy logits (rẻ vì 30M nhỏ).
- **Loss** trên mỗi token truyện (conditioning bị mask): `L = α·T²·KL(softmax(z_s/T) ‖ softmax(z_t/T)) + (1-α)·CE(z_s, y_true)`, với **T≈2, α≈0.5** (tinh chỉnh nếu cần).
- Ngân sách student ~**200M token** (~1.5 epoch trên subset 600k) — distill hội tụ nhanh hơn from-scratch.
- Optimizer/schedule như teacher (AdamW + WSD), peak LR có thể cao hơn chút (student nhỏ).
- Checkpoint/resume Drive; lưu `slm-10m-distilled`.
- Kỳ vọng: 10M-distilled vượt rõ 10M-from-scratch v1, tiến gần 30M.

## 6. Đánh giá (bám ADR-0002)

Tái dùng `scripts/eval_slm.py`:
- **Metric khách quan**: perplexity (test held-out), Distinct-1/2, Self-BLEU, Flesch.
- **Judge panel 3 model khác họ** (qwen3:4b, gemma2:2b, llama3.2:3b) chấm 4 trục (grammar/creativity/moral_clarity/prompt_adherence) cho: `slm-10m-distilled`, `slm-30m`, `qwen3-4b`.
- **Agreement**: weighted Cohen's κ + Kendall's τ; kết luận **theo thứ hạng**.
- Xuất `results/eval_summary.json` (schema mở rộng: models N-way + size_ladder + checkpoint_curve) → Results tab.
- Đây là phép đo near-parity ĐÚNG (không chỉ 1-judge như quick eval trên UI).

## 7. Tích hợp app

- Export cả hai model → **GGUF q8** (kèm 2 fix đã biết: `tokenizer_class` cho transformers 5.x + patch llama.cpp pre-tokenizer `gpt-2`) + Modelfile có **TEMPLATE** khớp train/serve → `ollama create`.
- Registry `config/models.json`: `slm-30m` (train lại, cùng id) + **thêm `slm-10m-distilled`** (kind `scratch-slm`, desc nêu "distilled from 30M").
- Compare mode: đặt SLM cạnh Qwen; Results tab đọc `eval_summary.json`.
- Notebook: mở rộng bản sạch (`notebooks/pretrain_slm_tf1.ipynb`) thành v2 — config full-Chinchilla 30M + **cell distillation** 30M→10M; giữ markdown chú thích tham số.

## 8. Deliverables & phân pha

| Pha | Deliverable |
|---|---|
| **S1 — Data v2** | Mở rộng `scripts/prepare_tf1_pretrain.py`: lọc chất lượng (`--min-words/--max-words`) + subset 600k. TDD hàm lọc trên fixture. |
| **S2 — Teacher 30M** | Notebook cell/config full-Chinchilla (600M token, WSD, checkpoint/resume Drive). Chạy Colab T4 (tương tác). Export GGUF `slm-30m`. |
| **S3 — Distill 10M** | Module/hàm distillation (KD loss) + cell notebook: teacher 30M → student 10M. TDD hàm KD-loss (mock logits). Export GGUF `slm-10m-distilled`. |
| **S4 — Eval + tích hợp** | Pull 2 judge, chạy `scripts/eval_slm.py` → `eval_summary.json`; registry + Results; Compare. |

Thứ tự: S1 → S2 (train thật, tương tác) → S3 (distill, tương tác) → S4. S2/S3 phụ thuộc Colab nên tách rõ.

## 9. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Colab free T4 recycle giữa run dài | Checkpoint/resume Drive bắt buộc; chia phiên; subset 600k giữ mỗi epoch vừa phải |
| Vẫn chưa near-parity sau full Chinchilla | Vẫn có giá trị khoa học (đường cong loss/size cải thiện rõ vs v1); kết luận theo rank; nêu rõ khoảng cách |
| Distillation không cải thiện 10M | TDD KD-loss; thử α/T khác; fallback: so 10M-distilled vs 10M-from-scratch để chứng minh distillation có tác dụng |
| Teacher forward làm distill chậm | Teacher 30M rất nhỏ; batch teacher ở `no_grad`/fp16 |
| Overfit khi 4 epoch | Theo dõi val loss; giảm epoch/tăng subset nếu val tăng |

## 10. Tiêu chí thành công

- Teacher 30M đạt val loss thấp hơn v1 rõ rệt (kỳ vọng ~1.3–1.5) và điểm judge cao hơn hẳn 2.5 của v1.
- 10M-distilled vượt 10M-from-scratch v1 (chứng minh distillation có tác dụng).
- `results/eval_summary.json` cho số liệu panel + κ/τ, kết luận theo rank cho 10M-distilled / 30M / Qwen.
- App Compare + Results trưng được thang chất lượng và tốc độ (SLM nhanh hơn Qwen nhiều lần).
- Tái lập được: script + notebook v2 + tokenizer + Modelfile trên Drive & repo.

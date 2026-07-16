# Design: Pretrain an SLM from scratch on TF1-EN-3M

Ngày: 2026-07-08
Trạng thái: Approved (chờ review spec)
Branch: `feat/slm-pretrain-tf1`

## 1. Mục tiêu & luận điểm khoa học

Pretrain **từ đầu** (không fine-tune base mạnh) một/hai **SLM (Small Language Model)** kiểu Llama trên bộ `klusai/ds-tf1-en-3m`, để chứng minh: **một model nhỏ (~10M và ~30M tham số) vẫn sinh được truyện ngụ ngôn tiếng Anh chất lượng tốt, gần ngang một LLM lớn (Qwen3-4B) trên domain fable** — dù nhỏ hơn ~130-400 lần. Đây là luận điểm gốc của TinyStories (Eldan & Li, 2023) và dòng follow-up TF1-EN-3M / TF3-RO-50M.

"Cải thiện/giá trị của việc train" được trưng qua **bốn góc nhìn**, tất cả hiển thị được trong app:
1. **Before/after**: model khởi tạo ngẫu nhiên (nói nhảm) → sau train (fable mạch lạc).
2. **Đường cong theo checkpoint**: chất lượng tăng theo số bước train.
3. **Đường cong theo kích thước**: ~10M vs ~30M ("nhỏ tới đâu vẫn tốt").
4. **SLM vs LLM**: SLM đạt near-parity với Qwen3-4B trên 4 trục chất lượng.

Ràng buộc khoa học: bám **ADR-0002** (không tự chế tiêu chí; dùng đúng bộ phương pháp paper TF1 + metric kinh điển). Trích dẫn: Nadas et al. (2025), arXiv:2504.20605.

### Non-goals (YAGNI)
- KHÔNG train LLM lớn / fine-tune model mạnh.
- KHÔNG theo đuổi SOTA tuyệt đối; chỉ cần chất lượng "tốt" trên domain fable trẻ em.
- KHÔNG train nhiều hơn 2 size (chốt 10M + 30M).
- KHÔNG dùng toàn bộ 3M fable (dùng subset).

## 2. Quyết định đã chốt (grill 2026-07-08)

| Chủ đề | Quyết định |
|---|---|
| Paradigm | Pretrain **từ đầu** (from scratch) |
| Compute | **Colab free T4 (16GB)** |
| Conditioning | **Theo scaffold** (character/setting/challenge/outcome/teaching) + **slot dropout** |
| Kích thước | **Thang 2 size: ~10M + ~30M** |
| Tokenizer | **BPE riêng, vocab nhỏ ~12k** (train trên corpus fable) |
| Data | **Subset ~500k fable**, multi-epoch tới ngân sách Chinchilla |
| Framework | **HF Transformers `LlamaForCausalLM` + Trainer** → GGUF → Ollama |
| Judge eval | **Panel 3 model khác họ** (qwen3:4b + gemma2:2b + llama3.2:3b) + Cohen's κ / Kendall's τ |

## 3. Kiến trúc model (Llama-style hiện đại)

Decoder-only kiểu Llama: **RoPE**, **GQA**, **RMSNorm**, **SwiGLU**, tied input/output embeddings. Hai cấu hình (số chính xác chốt ở plan; dưới đây là mục tiêu):

| Tham số | SLM-10M | SLM-30M |
|---|---|---|
| layers | 6 | 8 |
| d_model | 320 | 512 |
| attention heads | 8 | 8 |
| GQA kv-heads | 2 | 2 |
| FFN hidden (SwiGLU) | ~1280 | ~2048 |
| seq len | 512 | 512 |
| vocab | ~12k | ~12k |
| params (xấp xỉ, tied emb) | ~10-12M | ~28-32M |

Ghi chú: fable ngắn (~120-350 từ) nên seq len 512 là đủ. Vocab nhỏ giữ tỉ lệ embedding/tham số hợp lý ở scale này.

## 4. Data pipeline

**Nguồn**: `klusai/ds-tf1-en-3m` (streaming). **Subset ~500k** record sau lọc.

**Chuẩn bị** (`scripts/prepare_tf1_pretrain.py`):
- Đọc streaming, lấy các trường scaffold + fable text.
- **Dedup** theo `prompt_hash` (hoặc hash của fable) để tránh trùng.
- **Split** train/val/test (vd 96% / 2% / 2%); test giữ để tính perplexity held-out + batch eval.
- **Format có điều kiện** mỗi record thành chuỗi huấn luyện:
  - Dựng phần điều kiện từ các slot **có mặt** (giống `build_fable_prompt` của app), rồi nối phần fable.
  - Dùng **special tokens** phân tách phần điều kiện và phần truyện (vd `<|cond|> ... <|story|> ... <|end|>`) để loss tập trung vào phần truyện (train-on-response-style: mask phần điều kiện khỏi loss).
  - **Slot dropout**: mỗi slot bị bỏ độc lập với xác suất ~0.3; một tỉ lệ nhỏ record bỏ **toàn bộ** slot (điều kiện rỗng → free-gen). Nhờ vậy 5 ô tùy chọn của app (kể cả để trống) đều hoạt động.

**Tokenizer** (train một lần, lưu artifact):
- BPE (HF `tokenizers` hoặc SentencePiece), vocab ~12k, train trên tập fable text (+ mẫu phần điều kiện).
- Gồm special tokens ở trên. Lưu để tái dùng cho train + inference + export.

## 5. Phương pháp train (recipe)

`notebooks/pretrain_slm_tf1.ipynb` (Colab T4), chạy tuần tự cho từng size:

- **Objective**: next-token prediction; **mask loss** trên phần điều kiện (chỉ tính loss phần truyện).
- **Optimizer**: AdamW, β=(0.9, 0.95), weight decay 0.1, grad clip 1.0.
- **LR schedule**: **WSD** (Warmup-Stable-Decay) — warmup ~2% bước, stable, decay ~20% bước cuối. Peak LR dò nhanh (bậc ~3e-3 cho model nhỏ theo SmolLM2; tinh chỉnh).
- **Precision**: fp16/bf16 (T4 → fp16); gradient accumulation để đạt global batch ~0.25-1M token.
- **Token budget**: ~**Chinchilla 20 tok/param** → 10M ≈ 200M token, 30M ≈ 600M token; đạt bằng **multi-epoch** trên subset ~500k (mỗi epoch ~125M token với ~250 tok/fable → ~10M cần ~2 epoch, ~30M cần ~5 epoch). Theo dõi val loss để tránh overfit.
- **Checkpointing**: lưu checkpoint định kỳ (vd mỗi ~10-20% tiến trình) + **checkpoint step-0 (random init)** làm baseline "before". Log loss (train/val) để vẽ loss curve.
- **LR**: dùng peak LR theo SmolLM2 cho model nhỏ (bậc ~3e-3), tinh chỉnh nhanh nếu loss không giảm. (Không dùng μP — giữ pipeline gọn.)
- **Resume**: hỗ trợ resume từ Google Drive (Colab hay ngắt phiên).

**Export**: mỗi model → **GGUF q8** qua `llama.cpp convert` → `ollama create <name> -f Modelfile`. (LlamaForCausalLM + tokenizer chuẩn → llama.cpp chuyển được.)

## 6. Đánh giá (bám ADR-0002)

`scripts/eval_slm.py` chấm và xuất `results/eval_summary.json`.

- **Metric khách quan (reference-free)**: perplexity trên test held-out (per model + per checkpoint), Distinct-1/2, Self-BLEU, Flesch Reading Ease.
- **LLM-judge panel** (3 model khác họ qua Ollama: qwen3:4b, gemma2:2b, llama3.2:3b) chấm 4 trục paper (Grammar, Creativity, Moral Clarity, Prompt Adherence, 1-10) cho: SLM-10M, SLM-30M, và Qwen3-4B (reference).
- **Agreement**: weighted **Cohen's κ** + **Kendall's τ** giữa các judge.
- **Kết luận theo THỨ HẠNG** (đa số judge xếp cao hơn) + delta metric khách quan.
- **Các đường cong** cho luận điểm: (a) chất lượng ⟶ checkpoint (before/after), (b) chất lượng ⟶ size (10M vs 30M), (c) SLM vs Qwen (near-parity).
- TDD: hàm tính perplexity (mock logprobs), distinct_n, self_bleu, κ/τ trên fixture; phần gọi judge là integration.

## 7. Tích hợp vào app

- **Registry**: thêm SLM-10M, SLM-30M vào `config/models.json` với `kind` mới (vd `scratch-slm`) + `desc` (param count). App hiện `kind`/`desc` qua tooltip; Compare bật khi ≥2 model.
- **Compare mode**: đặt SLM cạnh Qwen3-4B để xem near-parity trực tiếp.
- **Results tab**: đọc `eval_summary.json`. Cần **mở rộng nhẹ schema + ResultsPanel** để vẽ: bảng SLM-10M / SLM-30M / Qwen (objective + judge), **ladder chất lượng theo size**, **loss/quality curve theo checkpoint**. Giữ tương thích ngược (render phòng thủ, thiếu section thì bỏ qua) như hiện tại.
- **Guardrail/observability**: dùng lại pipeline hiện có; SLM chạy như model Ollama bình thường.

## 8. Deliverables & phân pha

| Pha | Deliverable |
|---|---|
| **P1 — Data & tokenizer** | `scripts/prepare_tf1_pretrain.py` (subset, dedup, split, format có điều kiện + slot dropout) + train tokenizer BPE ~12k. TDD hàm format/dropout/split trên fixture. |
| **P2 — Train notebook** | `notebooks/pretrain_slm_tf1.ipynb`: config 10M/30M, WSD, mask-loss, checkpoint + step-0, log loss, resume Drive, export GGUF. Chạy trên Colab T4 (tương tác). |
| **P3 — Batch eval** | `scripts/eval_slm.py` → `results/eval_summary.json` (objective + panel + κ/τ + ladders). TDD phần tổng hợp. |
| **P4 — Tích hợp app** | Registry + `kind` mới; mở rộng ResultsPanel/schema cho ladder size + checkpoint; Compare SLM vs Qwen. |

Thứ tự: P1 → P2 (train thật, tương tác) → P3 → P4. P2 phụ thuộc compute Colab nên tách rõ.

## 9. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| SLM 10M/30M không đạt near-parity với Qwen | Vẫn có giá trị khoa học (định vị đường cong size); kết luận theo rank + nêu rõ khoảng cách. Có thể tăng epoch/subset hoặc nhích size. |
| Colab T4 ngắt phiên khi train lâu | Checkpoint + resume qua Drive; subset ~300k giữ thời gian mỗi run vừa phải. |
| Export GGUF lỗi với config lạ | Bám kiến trúc Llama chuẩn (llama.cpp hỗ trợ); test convert sớm trên checkpoint nhỏ. |
| Judge panel tốn thời gian local | Panel model nhỏ (2-4B); giới hạn test set (vd ~100-200 prompt held-out). |
| Overfit khi multi-epoch trên subset | Theo dõi val loss; cân subset-size vs epoch; early-ish stop theo val. |

## 10. Tiêu chí thành công

- Có 2 SLM (~10M, ~30M) chạy được trong app qua Ollama, sinh fable mạch lạc bám scaffold.
- `results/eval_summary.json` cho số liệu: objective + judge panel + κ/τ, với **kết luận theo rank** so SLM vs Qwen.
- App trưng được: before/after, ladder size, và SLM vs Qwen trong Compare + Results.
- Toàn bộ tái lập được (script + notebook + tokenizer artifact + Modelfile).

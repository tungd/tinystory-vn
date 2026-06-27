# Thiết kế: Fine-tune đa tầng hiện đại cho Qwen3-4B (truyện ngụ ngôn tiếng Việt)

- **Ngày**: 2026-06-27
- **Bối cảnh**: Cải tiến phần fine-tune của đồ án IT5410. Bản trước dùng Qwen3-1.7B (quá yếu). Ưu tiên: **chất lượng truyện tốt nhất có thể** với dữ liệu hiện có.
- **Quyết định nền (đã chốt với user)**:
  - Đổi base sang **Qwen3-4B** (mạnh hơn 1.7B rõ rệt).
  - **Dùng dữ liệu hiện có**, KHÔNG sinh dữ liệu tổng hợp từ LLM khác.
  - Pipeline **2 tầng**: SFT (QLoRA + DoRA + NEFTune) → ORPO (preference alignment, reference-free).
  - Sau khi xong: **web app chỉ dùng 2 model 4B** — `qwen3:4b` (base) và `fable-tuned` (4B fine-tuned). **Bỏ 1.7B** khỏi app.

---

## 1. Mục tiêu & phạm vi

Nâng chất lượng sinh truyện ngụ ngôn bằng:
1. Base mạnh hơn: **Qwen3-4B**.
2. Kỹ thuật fine-tune hiện đại (2026): **QLoRA + DoRA + NEFTune** cho SFT, rồi **ORPO** căn chỉnh ưu tiên.
3. Tinh chỉnh hyperparameter "nhẹ tay" để không phá khả năng nghe lệnh của base mạnh.

Tất cả chạy trên **Colab T4 free**, dùng `colab-mcp`.

### Phi mục tiêu (YAGNI)
- ❌ Sinh dữ liệu tổng hợp từ teacher LLM ngoài (user đã loại).
- ❌ RL/GRPO, RAG, DPO có reference model nặng (ORPO reference-free nhẹ hơn, hợp T4).
- ❌ Giữ 1.7B trong web app (chỉ còn dùng offline cho báo cáo nếu cần).

### Cơ sở phương pháp (nghiên cứu 2026)
- Dữ liệu nhỏ → "chất lượng > số lượng"; đòn bẩy data-centric bị loại nên tập trung vào **base mạnh + kỹ thuật train**.
- **DoRA**: phân tách weight thành magnitude+direction, +1–4% so với LoRA; khuyến nghị 2026 `r=16, DoRA, target_modules="all-linear"`.
- **NEFTune**: thêm nhiễu embedding khi SFT → cải thiện instruction-following trên data nhỏ.
- **ORPO**: gộp tín hiệu SFT + preference, **reference-free** (không cần model tham chiếu trong VRAM) → hợp dataset nhỏ + T4.
- Preference data không cần teacher: `chosen` = truyện thật (dataset), `rejected` = output base tự sinh.

---

## 2. Kiến trúc pipeline

```
[Dữ liệu hiện có: train/val/test.jsonl + refusals]
        │
        ├─(a) SFT dataset (instruction→truyện + refusal)
        │
        ▼
  Stage 1 — SFT trên Qwen3-4B
   QLoRA 4-bit + DoRA + NEFTune + train-on-responses-only (nhẹ tay)
        │  → lưu adapter/checkpoint "4B-SFT"
        ▼
  (b) Xây Preference dataset (trong notebook, cần GPU):
      với mỗi prompt: chosen = truyện thật, rejected = base-Qwen3-4B sinh
        │
        ▼
  Stage 2 — ORPO trên model 4B-SFT
   reference-free preference alignment
        │  → lưu "4B-SFT+ORPO" (model cuối)
        ▼
  Merge LoRA → GGUF q8_0 (q4 nếu cần) → Ollama `fable-tuned`
```

**Checkpoint cho báo cáo (so sánh nhiều mức):** 1.7B-base (offline) → 4B-base → 4B-SFT → 4B-SFT+ORPO.

---

## 3. Dữ liệu

### 3.1. SFT (như hiện tại)
- `data/processed/{train,val,test}.jsonl` từ `prepare_data` (có bộ lọc độ dài `--max-chars`, mặc định 6000 — để mở chỉnh).
- Format chat template (system + user instruction + assistant truyện), `enable_thinking=False`.
- Train-on-responses-only (chỉ tính loss phần assistant).

### 3.2. Preference dataset cho ORPO (xây trong notebook)
- Đầu vào: các prompt từ `train.jsonl` (phần `type=story`).
- Với mỗi prompt:
  - `chosen` = `output` truyện thật trong dataset.
  - `rejected` = sinh bằng **Qwen3-4B base** (chưa fine-tune) cho cùng prompt (1 lần, có repetition_penalty để tránh degenerate).
- Xuất `preference.jsonl`: `{"prompt": ..., "chosen": ..., "rejected": ...}` đúng format TRL ORPO.
- (Tùy chọn) refusal: `chosen` = câu từ chối lịch sự, `rejected` = output base nếu base tuân theo yêu cầu xấu.

---

## 4. Tham số (để mở trong ô HYPERPARAMETERS)

### Stage 1 — SFT
| Tham số | Giá trị khởi đầu | Ghi chú |
|---|---|---|
| MODEL_NAME | `unsloth/Qwen3-4B-Instruct-2507` | bản non-thinking, mạnh |
| MAX_SEQ_LENGTH | 2048 | |
| use_dora | True | DoRA |
| LORA_R / ALPHA | 16 / 16 | |
| target_modules | all-linear (7 proj) | |
| neftune_noise_alpha | 5 | NEFTune |
| LEARNING_RATE | 5e-5 (có thể 3e-5) | nhẹ tay vì base mạnh |
| EPOCHS | 1–2 | tránh overfit |
| BATCH_SIZE / GRAD_ACCUM | 1 / 8 | VRAM 4B |
| train_on_responses_only | True | |

### Stage 2 — ORPO
| Tham số | Giá trị khởi đầu | Ghi chú |
|---|---|---|
| LEARNING_RATE | 8e-6 | ORPO dùng LR rất thấp |
| beta (orpo lambda) | 0.1 | |
| EPOCHS | 1 | |
| MAX_SEQ_LENGTH | ≤2048 (hạ 1024 nếu OOM) | ORPO xử lý chosen+rejected → tốn hơn |
| BATCH_SIZE / GRAD_ACCUM | 1 / 8 | |

---

## 5. Files thay đổi

- `notebooks/finetune_qwen3_qlora.ipynb` (hoặc bản mới `finetune_qwen3_4b_orpo.ipynb`):
  - Đổi MODEL_NAME → Qwen3-4B; thêm `use_dora=True`, `neftune_noise_alpha`.
  - **Thêm nhóm cell (b)**: sinh preference dataset bằng base 4B.
  - **Thêm nhóm cell Stage 2**: train ORPO (TRL `ORPOTrainer` / Unsloth).
  - Cell sinh thử + merge + export GGUF (q8) như cũ, cập nhật tên.
- `ollama/Modelfile`: `FROM ../models/<gguf 4B mới>`; giữ `repeat_penalty 1.3`, `num_ctx 2048`.
- `app/config.py`: `BASE_MODEL` mặc định `qwen3:4b` (đổi từ `qwen3:1.7b`); `TUNED_MODEL` giữ `fable-tuned`.
- `app/main.py` `MODEL_INFO`: cập nhật mô tả nếu cần (vẫn 2 lựa chọn base/tuned, giờ đều 4B).
- (Local sau khi train) `ollama create fable-tuned -f ollama/Modelfile` với gguf 4B; app chạy mặc định 2 model 4B.

---

## 6. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| 4B + data cổ nhỏ → SFT phá coherence (như 1.7B lần đầu) | LR thấp + 1–2 epoch + responses-only + DoRA (targeted) + NEFTune |
| ORPO đẩy về văn phong cổ/lủng củng | beta vừa phải, 1 epoch, LR rất thấp; đánh giá định tính trước khi merge |
| VRAM T4 sát giới hạn (4B + ORPO chosen+rejected) | batch=1, grad accum, gradient checkpointing; dự phòng seq=1024 hoặc gguf q4 |
| Build preference data tốn thời gian (sinh base cho ~73 prompt) | giới hạn max_new_tokens, repetition_penalty; chạy 1 lần, cache |
| Delta before/after hẹp (base 4B đã khá) | chấp nhận; báo cáo nhấn vào style/an toàn + so sánh đa mức gồm 1.7B |

---

## 7. Tiêu chí hoàn thành (Definition of Done)

- [ ] Notebook 4B chạy được trên T4: SFT (DoRA+NEFTune) hoàn tất, loss giảm.
- [ ] Preference dataset được sinh + lưu đúng format ORPO.
- [ ] ORPO chạy được; sinh thử cho truyện mạch lạc, đúng thể loại, an toàn.
- [ ] Export GGUF q8 (4B) → `ollama create fable-tuned`.
- [ ] `app/config.py` đổi base mặc định `qwen3:4b`; app chạy với **2 model 4B** (base + fine-tuned), không còn 1.7B.
- [ ] Lưu các checkpoint/output để so sánh nhiều mức cho báo cáo.

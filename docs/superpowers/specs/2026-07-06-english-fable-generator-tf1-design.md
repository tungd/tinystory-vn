# Thiết kế: English Fable Generator (TF1-EN-3M) — app React/Astryx + đánh giá

- **Ngày**: 2026-07-06
- **Bối cảnh**: Pivot toàn diện từ "ngụ ngôn tiếng Việt" sang **ngụ ngôn TIẾNG ANH**, dùng dataset **`klusai/ds-tf1-en-3m`** (3M fable, có moral, 5-element narrative structure). Redesign app chuyên nghiệp (React + Astryx) với streaming, logging, và đánh giá chất lượng.
- **Lý do pivot**: dữ liệu ngụ ngôn tiếng Việt quá ít/khó nhất quán; TF1-EN-3M dồi dào, đồng loại, có moral (khớp thiết kế app), văn phong phong phú hơn TinyStories. Xem [ADR-0001] (consistency over volume) và `CONTEXT.md`.

---

## 1. Mục tiêu & phạm vi

Ứng dụng web sinh **truyện ngụ ngôn tiếng Anh cho trẻ em (4–7 tuổi)** từ model fine-tune trên TF1-EN-3M, với:
1. Nhập theo **Narrative Structure** (5 yếu tố).
2. **Stream** truyện theo chunk + **cột logging** quan sát quá trình sinh.
3. **Đánh giá** chất lượng đầu ra (LLM-as-judge 4 trục, theo paper TF1).
4. **Model registry** cấu hình được (thêm model sau chỉ cần sửa config).
5. Giao diện chuyên nghiệp bằng **Astryx** (React + StyleX).

### Phi mục tiêu (YAGNI)
- ❌ Đa ngôn ngữ / tiếng Việt (đã pivot sang English-only).
- ❌ Tài khoản, database, lưu lịch sử.
- ❌ Deploy production — chạy local.
- ❌ RL/GRPO. Fine-tune = SFT (QLoRA).

---

## 2. Kiến trúc tổng thể

```
React SPA (Astryx)  ──HTTP + SSE──▶  FastAPI backend  ──▶  Ollama (models từ registry)
  input · story-stream · log · eval      orchestration · guardrail · LLM-judge
```
- Frontend: **làm mới hoàn toàn** bằng React + Astryx (thay frontend vanilla cũ).
- Backend: **giữ FastAPI**, adapt sang English + endpoints mới.
- Model: **Qwen3-4B** fine-tune trên TF1, phục vụ qua Ollama.

---

## 3. Dữ liệu (TF1-EN-3M)

- Nguồn: HF `klusai/ds-tf1-en-3m`. Dùng `datasets` **streaming** để chỉ lấy subset (không tải cả 3M).
- Schema nguồn: `prompt` (narrative elements), `fable`, `prompt_hash`, `system_message`, + metadata sinh.
- **Xử lý**: khử trùng qua `prompt_hash`; lọc độ dài hợp lý (bỏ quá ngắn/quá dài); bỏ record có cờ nội dung.
- **Narrative Structure (5 yếu tố)** = input vocabulary (xem `CONTEXT.md`): Main Character, Setting, Challenge, Outcome, Teaching.
- **Định dạng instruction**: dựng từ 5 yếu tố → `fable`. App dùng CÙNG khuôn prompt này khi sinh (nhất quán train/inference). Yếu tố để trống → prompt bỏ qua, model tự quyết.

---

## 4. Chiến lược Fine-tune (Qwen3-4B trên Colab T4 free)

- **Base**: `unsloth/Qwen3-4B-Instruct-2507`, QLoRA 4-bit (Unsloth), `max_seq_length=2048`.
- **Đòn bẩy throughput: sample packing ON** — gói nhiều fable (~450 token) vào chuỗi 2048 → ~4× throughput.
- **Kỹ thuật**: NEFTune (`neftune_noise_alpha=5`), train-on-responses-only, LR 1e-4, batch 1 + grad accum 8, 1–2 epoch. (KHÔNG dùng DoRA — vỡ export GGUF của Unsloth, bài học đã ghi.)
- **Tối đa khuyến nghị 1 phiên free**: **~20k fable × 1 epoch** (~2.5–3h nhờ packing).
- **Vượt 20k → resumable đa phiên**: checkpoint adapter ra **Google Drive mỗi ~150 step** + `resume_from_checkpoint`; chạy nối nhiều phiên (~3h/phiên) → đạt 40–60k. Keep-alive tránh ngắt idle.
- **Export**: merge → **GGUF q8** → `ollama create` → thêm entry vào `config/models.json` (kind=finetuned, mô tả rõ subset/epoch).
- **Model nền để so sánh**: `qwen3:4b` (base).

---

## 5. Model registry (`config/models.json`)

Danh sách model khả dụng; thêm model sau = thêm 1 dòng + `ollama create`.
```json
[
  {"id":"base-qwen3-4b","name":"Qwen3-4B (base)","ollama":"qwen3:4b","kind":"base","desc":"Chưa fine-tune"},
  {"id":"tf1-sft-20k","name":"Fable-TF1 SFT 20k","ollama":"fable-tf1","kind":"finetuned","desc":"SFT Qwen3-4B trên 20k TF1, packing, 1 epoch"}
]
```
Mỗi entry: `id`, `name` (hiển thị), `ollama` (tên model Ollama), `kind` (`base`|`finetuned`), `desc` (cách fine-tune). Backend `GET /models` đọc file → UI dropdown hiện name + kind + desc.

---

## 6. Backend (FastAPI)

- `GET /models` — trả registry từ `config/models.json`.
- `POST /generate/stream` — SSE. Body: `{character, setting, challenge, outcome, teaching, length, model_id, guardrail_enabled}`.
  - Dựng prompt từ 5 yếu tố (ô trống → bỏ qua).
  - Guardrail 4 lớp (English) nếu bật: lọc input → system prompt ràng buộc → model từ chối → lọc output; regenerate ≤1 lần rồi refuse.
  - Sự kiện SSE: `step` (log pipeline), `token` (chunk truyện, chỉ khi guardrail tắt hoặc sau khi qua lọc — giữ bất biến "không lộ nội dung chưa lọc khi guardrail bật"), `done` (story), `error`.
  - `model_id` → resolve sang tên Ollama qua registry.
- `POST /evaluate` — Body `{story, prompt, judge_model_id?}` → **LLM-judge** chấm 4 trục **Grammar, Creativity, Moral Clarity, Prompt Adherence** (0–10) + nhận xét ngắn → JSON. Judge mặc định là model local cấu hình được, nên **khác model sinh** (giảm self-bias).

---

## 7. Frontend (React + Astryx) — bố cục 3 cột

- **Cột trái — Input**: 5 ô narrative (free-text + placeholder gợi ý), chọn độ dài (Short/Medium/Long), dropdown **model** (từ registry, hiện name+kind), toggle **guardrail**, nút **Generate**.
- **Cột giữa — Story**: khung chính stream **fable theo chunk** (loading/empty/error/refused states); khi xong render truyện + nút **Evaluate** → hiện bảng điểm 4 trục.
- **Cột phải — Logging**: activity feed realtime từ SSE `step` events (guardrail Lớp 1→4, gọi model nào, token count, thời gian). Cho phép observe quá trình sinh.
- Astryx components + theming; dark mode; responsive (hẹp → xếp dọc). Dự phòng: nếu Astryx 0.1.x lỗi → shadcn/ui.

---

## 8. Đánh giá

- **Trên UI (per-generation)**: sau khi sinh, gọi `/evaluate` → hiển thị điểm 4 trục + tổng.
- **Batch offline** (`scripts/eval_tf1.py`): chấm **base vs fine-tuned** trên test set TF1 (held-out), 4 trục + trung bình + so sánh → số liệu before/after cho báo cáo. Theo phương pháp **panel LLM-judge** của paper TF1 (có thể dùng ≥1 judge; ghi rõ judge nào).

---

## 9. Guardrail (English)

4 lớp chuyển sang tiếng Anh: (1) lọc input (banned words/off-scope EN), (2) system prompt ràng buộc "chỉ viết fable trẻ em", (3) model đã học từ chối (nếu có refusal data), (4) lọc output. Toggle bật/tắt để demo năng lực bảo vệ.

---

## 10. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Astryx 0.1.x còn non, API đổi | Cô lập lớp UI; dự phòng shadcn/ui nếu vỡ |
| Colab free ngắt phiên khi train lớn | Checkpoint Drive mỗi ~150 step + resume; giới hạn ~20k/phiên |
| Judge model "dễ dãi"/self-bias | Judge khác model sinh; rubric chặt; batch dùng ≥1 judge |
| Streaming lộ nội dung chưa lọc khi guardrail bật | Khi guardrail bật: chỉ stream sau lọc/không stream token thô (giữ bất biến cũ) |
| TF1 rập khuôn/bias fable phương Tây | Chấp nhận (dataset synthetic); nêu trong báo cáo |

---

## 11. Tiêu chí hoàn thành (DoD)

- [ ] Pipeline tải + lọc + định dạng subset TF1 (streaming từ HF).
- [ ] Notebook fine-tune Qwen3-4B (packing, NEFTune, responses-only) + chiến lược checkpoint/resume Drive.
- [ ] Export GGUF → `ollama create` → entry trong `config/models.json`.
- [ ] Backend: `/models`, `/generate/stream` (SSE, guardrail, model_id), `/evaluate` (4 trục).
- [ ] Frontend React+Astryx: 3 cột (input / story-stream / logging) + bảng đánh giá.
- [ ] `scripts/eval_tf1.py`: batch base vs fine-tuned trên test set (4 trục) cho báo cáo.

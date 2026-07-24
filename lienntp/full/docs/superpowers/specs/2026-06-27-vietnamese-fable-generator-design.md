# Thiết kế: Ứng dụng sinh truyện ngụ ngôn tiếng Việt cho trẻ em bằng LLM

- **Ngày**: 2026-06-27
- **Bối cảnh**: Đồ án cuối kỳ môn IT5410 (cần chấm điểm). Trọng tâm: phương pháp đúng đắn, demo chạy được, giải thích được — không cần chuẩn production.
- **Tác giả**: trieu.le

---

## 1. Mục tiêu & phạm vi

Xây dựng ứng dụng sinh **truyện ngụ ngôn (hư cấu) tiếng Việt cho trẻ em** bằng một LLM được fine-tune từ model nền có sẵn.

Ba trụ cột học thuật của đồ án:

1. **Transfer learning trên kiến trúc transformer**: fine-tune một LLM tiếng Việt có sẵn (đã là transformer) bằng QLoRA — *không* train transformer from scratch.
2. **Guardrail an toàn nội dung đa tầng**: đảm bảo hệ thống chỉ tạo truyện ngụ ngôn trẻ em, từ chối nội dung tục tĩu/bậy bạ, nội dung không phù hợp, và yêu cầu ngoài chức năng.
3. **Đánh giá định lượng before/after**: chứng minh fine-tune cải thiện chất lượng và guardrail nâng cao độ an toàn — bằng số liệu, không chỉ cảm tính.

### Phi mục tiêu (YAGNI — KHÔNG làm)

- ❌ Train transformer from scratch (không khả thi với GPU free, chất lượng kém).
- ❌ Tài khoản người dùng, đăng nhập, database, lưu lịch sử truyện.
- ❌ Deploy production / cloud hosting — ứng dụng chạy local trên MacBook là đủ.
- ❌ Frontend cầu kỳ — ưu tiên rõ ràng, đủ 4 trạng thái UI (loading/success/empty/error).

---

## 2. Ràng buộc & quyết định nền

| Yếu tố | Quyết định | Lý do |
|--------|-----------|-------|
| Tài nguyên train | Google Colab / Kaggle (T4 ~16GB, free) | Ràng buộc cứng → bắt buộc PEFT (QLoRA), không full fine-tune |
| Model nền | **Qwen3-4B-Instruct-2507** (bản *non-thinking*) | Mới (06/2026), Unsloth có notebook QLoRA chạy free trên T4, giảm ~70% VRAM, context dài hơn (hợp truyện dài), độ phủ tiếng Việt tốt |
| Phương pháp fine-tune | QLoRA 4-bit qua **Unsloth** | Khả thi trên T4, tốc độ cao, ổn định |
| Máy chạy ứng dụng | MacBook M3 Pro / 36GB RAM | Dư sức chạy model 4B local ở độ chính xác cao (q8/fp16) |
| Serve model local | **Ollama** | Gọn nhất trên Apple Silicon |
| Backend | **FastAPI** | Chứa logic guardrail + orchestration |
| Frontend | Web đơn giản | Form nhập + hiển thị truyện + toggle guardrail |

Model nền dự phòng để đối chiếu baseline: VyLinh-3B (nền Qwen2.5-3B, cũ hơn) — tùy chọn, không bắt buộc.

---

## 3. Kiến trúc tổng thể — pipeline 5 giai đoạn

```
[1] Dữ liệu        →  [2] Fine-tune (Colab T4)  →  [3] Xuất GGUF
 (instruction)         (QLoRA / Unsloth)            (merge + quantize)
                                                          │
                                                          ▼
[5] Đánh giá  ◄──  [4] Chạy local trên Mac:
 (before/after)        Ollama → FastAPI (guardrail 4 lớp) → Frontend
```

Cloud chỉ dùng cho giai đoạn train (nặng). Toàn bộ ứng dụng vận hành local, tránh điểm yếu Colab không host backend lâu dài.

---

## 4. Giai đoạn 1 — Dữ liệu

> Nội dung dữ liệu để ngỏ (người dùng cung cấp sau). Phần này chốt **cấu trúc & pipeline xử lý** để cắm dữ liệu vào lúc nào cũng được.

### 4.1. Định dạng huấn luyện (instruction tuning)

Mỗi mẫu là một cặp instruction → output:

```json
{
  "instruction": "Viết một truyện ngụ ngôn cho trẻ em về chủ đề: {chủ đề}. Bài học đạo đức: {bài học}. Độ tuổi: {độ tuổi}.",
  "output": "<truyện ngụ ngôn hoàn chỉnh: mở đầu → mâu thuẫn → cao trào → kết + bài học>"
}
```

Cấu trúc input truyện thống nhất 3 thành phần: **chủ đề + bài học đạo đức + độ tuổi**.

### 4.2. Pipeline xử lý dữ liệu

1. **Thu thập** (linh hoạt): truyện thật do người dùng cung cấp, hoặc dữ liệu tổng hợp do LLM lớn sinh ra.
2. **Làm sạch**: khử trùng lặp, loại nội dung không phù hợp trẻ em ngay trong tập train, chuẩn hóa định dạng.
3. **Sinh dữ liệu từ chối** (refusal data): tập các yêu cầu xấu / ngoài phạm vi (tục tĩu, bạo lực, chủ đề người lớn, yêu cầu không phải tạo truyện) → output là câu **từ chối lịch sự, đúng vai trò**. Đây là cách dạy model "biết nói không" — trụ cột của lớp guardrail thứ 3.
4. **Chia tập**: train / validation / test (test giữ cố định để so sánh before/after công bằng).

### 4.3. Yêu cầu tối thiểu

- Ước lượng cần ~vài trăm cặp truyện chất lượng + ~50–100 mẫu refusal để thấy khác biệt before/after rõ ràng. (Con số tinh chỉnh trong giai đoạn thực thi.)

---

## 5. Giai đoạn 2 — Fine-tuning

- **Notebook**: Unsloth QLoRA cho Qwen3-4B-Instruct-2507 trên Colab T4.
- **Kỹ thuật**: nạp model 4-bit, gắn LoRA adapter, train trên tập instruction (gồm cả mẫu refusal).
- **Theo dõi**: training loss + validation loss để tránh overfit.
- **Sản phẩm ra**: LoRA adapter → **merge vào base model**.
- **Tham số gợi ý** (chốt khi thực thi): rank LoRA, learning rate, số epoch, max_seq_len đủ dài cho truyện.

---

## 6. Giai đoạn 3 — Xuất model

- Merge adapter → convert sang **GGUF**.
- Lượng tử hóa: **q8_0** (chất lượng cao, máy M3 Pro/36GB dư sức) hoặc q4_K_M (nhẹ hơn). Mặc định q8_0.
- Nạp vào **Ollama** qua `Modelfile` (định nghĩa model + system prompt mặc định).

---

## 7. Giai đoạn 4 — Ứng dụng local

### 7.1. Serving
- **Ollama** chạy model GGUF, expose API local.

### 7.2. Backend — FastAPI
Endpoint chính: `POST /generate`

Tham số: `{ topic, moral, age_range, guardrail_enabled: bool }`

Luồng xử lý khi `guardrail_enabled = true` — **4 lớp guardrail**:

1. **Lớp 1 — Lọc đầu vào**: chặn từ cấm + phân loại chủ đề (chỉ nhận yêu cầu tạo truyện ngụ ngôn trẻ em). Yêu cầu ngoài phạm vi → trả về thông báo từ chối, không gọi model.
2. **Lớp 2 — System prompt**: ràng buộc vai trò cứng ("Bạn là người kể truyện ngụ ngôn cho trẻ em…") khi gọi Ollama.
3. **Lớp 3 — Model đã học từ chối**: nhờ refusal data ở phần 4.2, model tự từ chối các yêu cầu lọt qua lớp 1.
4. **Lớp 4 — Lọc đầu ra**: quét truyện sinh ra (từ cấm / nội dung không phù hợp); nếu vi phạm → sinh lại hoặc từ chối.

Khi `guardrail_enabled = false`: bỏ qua lớp 1 và lớp 4 (model thô + system prompt tối thiểu) — phục vụ demo so sánh năng lực bảo vệ.

### 7.3. Frontend
- Form nhập: **chủ đề + bài học đạo đức + độ tuổi**.
- **Toggle bật/tắt guardrail** (gửi `guardrail_enabled` xuống backend) → minh họa trực quan hệ thống bị qua mặt thế nào khi tắt bảo vệ.
- Hiển thị truyện kết quả.
- Đủ 4 trạng thái UI: loading / success / empty / error (gồm cả trạng thái "yêu cầu bị từ chối").

---

## 8. Giai đoạn 5 — Đánh giá before/after (lõi báo cáo)

So sánh **base model (chưa fine-tune)** vs **model đã fine-tune**, trên tập test cố định.

### 8.1. Chất lượng sinh truyện
- **Định tính**: bảng side-by-side cùng prompt (base vs fine-tuned).
- **LLM-as-judge**: dùng một model mạnh chấm điểm theo rubric — (a) đúng thể loại ngụ ngôn (cấu trúc + có bài học), (b) văn phong & độ trôi chảy tiếng Việt, (c) phù hợp trẻ em — và so sánh cặp (pairwise preference).
- **Định lượng**: perplexity trên tập truyện held-out (model fine-tune kỳ vọng khớp văn phong ngụ ngôn hơn → perplexity thấp hơn).

### 8.2. Hiệu quả guardrail
- Tập prompt đối kháng: yêu cầu tục tĩu / bạo lực / chủ đề người lớn / ngoài chức năng.
- Đo **tỷ lệ từ chối đúng** (precision / recall) ở 3 cấu hình:
  - Base model, không guardrail.
  - Fine-tuned model, không guardrail (chỉ lớp 3).
  - Fine-tuned model, đủ 4 lớp guardrail.
- Tận dụng đúng toggle ở frontend → số liệu định lượng cho năng lực bảo vệ từng tầng.

---

## 9. Cấu trúc dự án (đề xuất)

```
Final/
├── data/                 # dữ liệu thô + đã xử lý (gitignore phần lớn)
│   ├── raw/
│   ├── processed/        # *.jsonl instruction format
│   └── refusal/          # mẫu từ chối
├── notebooks/
│   └── finetune_qwen3_qlora.ipynb   # train trên Colab
├── scripts/
│   ├── prepare_data.py   # làm sạch + định dạng
│   ├── export_gguf.py    # merge + quantize
│   └── evaluate.py       # before/after + guardrail metrics
├── backend/              # FastAPI
│   ├── main.py
│   ├── guardrail/        # 4 lớp
│   └── ollama_client.py
├── frontend/             # web đơn giản
├── models/               # adapter, gguf (gitignore)
└── docs/
    └── superpowers/specs/2026-06-27-vietnamese-fable-generator-design.md
```

---

## 10. Rủi ro & câu hỏi mở

| Rủi ro | Ảnh hưởng | Giảm thiểu |
|--------|-----------|-----------|
| Dữ liệu chưa có | Cao — quyết định chất lượng fine-tune | Chuẩn bị pipeline trước; có phương án dữ liệu tổng hợp dự phòng |
| Tập train quá nhỏ → before/after không rõ | Trung bình | Bổ sung dữ liệu tổng hợp; đặt kỳ vọng đúng trong báo cáo |
| Phiên Colab giới hạn thời gian | Thấp | QLoRA 4B nhanh; lưu checkpoint thường xuyên |
| Guardrail lọc nhầm yêu cầu hợp lệ (false positive) | Trung bình | Tinh chỉnh ngưỡng; báo cáo precision/recall trung thực |

### Câu hỏi mở (chốt khi thực thi)
- Nguồn dữ liệu cụ thể (truyện thật vs tổng hợp).
- Số lượng mẫu train tối thiểu.
- Danh sách từ cấm + bộ phân loại chủ đề cho lớp 1 (luật đơn giản hay model nhỏ).

---

## 11. Tiêu chí hoàn thành (Definition of Done)

- [ ] Pipeline dữ liệu xử lý được dữ liệu thô → instruction `*.jsonl`.
- [ ] Fine-tune thành công trên Colab T4, có adapter + log loss.
- [ ] Export GGUF chạy được trên Ollama (Mac local).
- [ ] FastAPI `/generate` hoạt động với toggle guardrail.
- [ ] Frontend tạo truyện + bật/tắt guardrail.
- [ ] Báo cáo đánh giá before/after (chất lượng + guardrail) với số liệu.

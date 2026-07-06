# CONTEXT — Trình tạo truyện ngắn cho trẻ em (English)

Glossary các thuật ngữ nghiệp vụ của dự án. Không chứa chi tiết triển khai.

> **PIVOT NGÔN NGỮ (2026-07-01):** Đầu ra chuyển sang **TIẾNG ANH** (trước đây tiếng Việt). Lý do: dữ liệu huấn luyện tiếng Việt quá ít; kho dữ liệu truyện ngắn tiếng Anh (TinyStories, v.v.) dồi dào & chất lượng. App/guardrail/system prompt/khung đồ án đổi sang tiếng Anh. Dữ liệu tiếng Việt (`fables_all.jsonl`, `fairytales.jsonl`, `verse_fables.jsonl`) giữ lại để tham chiếu, không train nữa.

## Fable (English) — VĂN PHONG ĐÍCH
Đầu ra mục tiêu: **truyện ngụ ngôn tiếng Anh** cho trẻ em (4–7 tuổi), có bài học đạo đức, nhân vật (thường là con vật). Nguồn dữ liệu chuẩn: **`klusai/ds-tf1-en-3m`** (TF1-EN-3M).

## Narrative Structure (5 yếu tố) — VOCABULARY ĐẦU VÀO
Cấu trúc kể chuyện của TF1, cũng là **5 ô input** người dùng nhập (đều tùy chọn, free-text):
1. **Main Character** — nhân vật chính (vd "a clever fox").
2. **Setting** — bối cảnh (vd "a foggy marsh").
3. **Challenge** — vấn đề/xung đột nhân vật gặp.
4. **Outcome** — cách giải quyết/kết cục.
5. **Teaching** — bài học/moral truyện truyền tải.
Ô trống → model tự quyết. (Thay cho bộ input cũ topic/moral/age.)

## Truyện cổ tích (Fairy tale) — KHÔNG phải văn phong đích
Truyện kể **dài, nhiều tình tiết, văn phong hoa mỹ/cổ** (vd Grimm, Andersen; median ~1066 từ). Có mặt trong dữ liệu thô nhưng **không** phải văn phong đích; nếu trộn vào tập train sẽ làm văn phong đầu ra lộn xộn, kém mạch lạc.
- **Phân biệt**: "ngụ ngôn" ≠ "cổ tích". Khi nói "truyện" trong dự án này, mặc định là **ngụ ngôn** (văn phong đích).

## Model nền (Base model)
`qwen3:4b` (Qwen3-4B-Instruct-2507) chưa fine-tune. Đã mạch lạc, dùng làm mốc so sánh "trước khi train".

## Model đã fine-tune (Fable-tuned)
`fable-tuned` — Qwen3-4B đã fine-tune trên dữ liệu ngụ ngôn (SFT + ORPO). Kết quả "sau khi train".

## Evaluation axes (LLM-as-judge)
4 trục chấm chất lượng fable đầu ra (thang 0–10), theo paper TF1-EN-3M:
- **Grammar** — độ đúng ngữ pháp/mạch lạc câu chữ.
- **Creativity** — độ sáng tạo/hấp dẫn của truyện.
- **Moral Clarity** — bài học có rõ ràng, truyền tải tốt không.
- **Prompt Adherence** — bám sát các yếu tố narrative người dùng nhập.
`overall` = trung bình 4 trục. Judge là một LLM khác model sinh (giảm self-bias).

## Base vs Fine-tuned (before/after)
So sánh cốt lõi của đồ án: **base** (`qwen3:4b`, chưa train) vs **fine-tuned** (train trên TF1). App phải phản ánh khác biệt này (Single mode để xem từng model; Compare mode để đặt cạnh nhau + delta điểm).

## Guardrail
Cơ chế bảo vệ nhiều lớp đảm bảo app chỉ tạo truyện ngụ ngôn trẻ em an toàn, từ chối nội dung/yêu cầu ngoài phạm vi.

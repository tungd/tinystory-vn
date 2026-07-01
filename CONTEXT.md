# CONTEXT — Trình tạo truyện ngụ ngôn tiếng Việt

Glossary các thuật ngữ nghiệp vụ của dự án. Không chứa chi tiết triển khai.

## Truyện ngụ ngôn (Fable) — VĂN PHONG ĐÍCH
Thể loại **đầu ra mục tiêu** của ứng dụng: truyện **ngắn** (~150–300 từ), thường có nhân vật con vật, cốt truyện đơn giản, và **một bài học đạo đức rõ ràng ở cuối**. Văn phong súc tích, trong sáng, phù hợp trẻ em.
- **Mẫu chuẩn**: Aesop, La Fontaine.
- Đây là văn phong mà model fine-tune cần học và app cần sinh ra.

## Truyện cổ tích (Fairy tale) — KHÔNG phải văn phong đích
Truyện kể **dài, nhiều tình tiết, văn phong hoa mỹ/cổ** (vd Grimm, Andersen; median ~1066 từ). Có mặt trong dữ liệu thô nhưng **không** phải văn phong đích; nếu trộn vào tập train sẽ làm văn phong đầu ra lộn xộn, kém mạch lạc.
- **Phân biệt**: "ngụ ngôn" ≠ "cổ tích". Khi nói "truyện" trong dự án này, mặc định là **ngụ ngôn** (văn phong đích).

## Model nền (Base model)
`qwen3:4b` (Qwen3-4B-Instruct-2507) chưa fine-tune. Đã mạch lạc, dùng làm mốc so sánh "trước khi train".

## Model đã fine-tune (Fable-tuned)
`fable-tuned` — Qwen3-4B đã fine-tune trên dữ liệu ngụ ngôn (SFT + ORPO). Kết quả "sau khi train".

## Guardrail
Cơ chế bảo vệ nhiều lớp đảm bảo app chỉ tạo truyện ngụ ngôn trẻ em an toàn, từ chối nội dung/yêu cầu ngoài phạm vi.

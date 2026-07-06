# 2. Phương pháp đánh giá bám chuẩn khoa học (không tự chế tiêu chí)

Ngày: 2026-07-06
Trạng thái: Accepted

## Bối cảnh
Đồ án ưu tiên **tính khoa học của việc train model**. Đánh giá chất lượng sinh truyện phải dựa trên **phương pháp đã công bố**, không tự vẽ tiêu chí. Nguồn chuẩn: paper **TF1-EN-3M** (arXiv 2504.20605) — dataset ta dùng.

## Quyết định
Đánh giá **base vs fine-tuned** dùng đúng bộ phương pháp của paper + metric kinh điển, KHÔNG bịa trục mới:

1. **Metric khách quan (reference-free / tự động):**
   - **Perplexity** trên tập test held-out (kinh điển cho fine-tune LM — kỳ vọng tuned < base).
   - **Distinct-1/2** + **Self-BLEU** (đa dạng từ vựng — như paper).
   - **Flesch Reading Ease** (độ đọc dễ — như paper).
2. **LLM-as-judge panel** (theo paper): ≥2–3 model **khác họ** chấm 4 trục **Grammar & Style, Creativity, Moral Clarity, Prompt Adherence** (thang 1–10). KHÔNG thêm/bớt trục.
3. **Độ tin cậy judge**: báo cáo **weighted Cohen's κ** + **Kendall's τ**. Vì paper cho thấy item-level κ thấp nhưng **rank-order (τ) cao**, KẾT LUẬN before/after **ưu tiên theo THỨ HẠNG** (model nào được đa số judge xếp cao hơn) + delta metric khách quan, KHÔNG dựa điểm tuyệt đối của 1 judge.

## Hệ quả
- (+) Kết luận đánh giá phòng thủ được về mặt khoa học, tái lập được.
- (+) Kết hợp khách quan (perplexity/diversity/readability) + chủ quan (judge panel) → toàn diện.
- (−) Chạy panel nhiều judge tốn compute (local M3 Pro: dùng 2–3 model open khác họ qua Ollama, hoặc 1 tham chiếu mạnh).
- App: eval per-generation trên UI chỉ là **chỉ báo nhanh (1 judge)**; số liệu nghiêm chỉnh nằm ở **batch eval + Results panel**.

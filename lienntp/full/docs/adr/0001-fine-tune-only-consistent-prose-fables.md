# 1. Fine-tune chỉ trên ngụ ngôn văn xuôi nhất quán

Ngày: 2026-07-01
Trạng thái: Accepted

## Bối cảnh
Dữ liệu thô ban đầu (161 truyện) trộn 4 văn phong: Grimm + Andersen (cổ tích dài, hoa mỹ — 80%) và Aesop + La Fontaine (ngụ ngôn ngắn). Mục tiêu sản phẩm là **truyện ngụ ngôn ngắn, văn phong trau chuốt, mạch lạc xuyên suốt**. Fine-tune trên tập nhỏ + hổ lốn văn phong làm model học ra văn phong trung bình/lộn xộn, GIẢM mạch lạc (đã quan sát ở các lần train trước). Ràng buộc: không sinh dữ liệu tổng hợp bằng LLM khác.

## Quyết định
Fine-tune model ngụ ngôn CHỈ trên **ngụ ngôn văn xuôi ngắn nhất quán** (Aesop, La Fontaine, "108 Truyện Ngụ Ngôn", "Ngụ Ngôn Việt Nam Chọn Lọc" = 483 truyện, median ~759 ký tự).
- **Loại trừ cổ tích** (Grimm/Andersen) → tách sang `data/raw/fairytales.jsonl`, giữ để dùng sau.
- **Loại trừ ngụ ngôn thơ** (Tolstoy) — khác văn phong (thơ), bỏ hẳn.
- Tăng dữ liệu bằng cách **thu thập thêm ngụ ngôn thật public-domain** (33 → 483), KHÔNG sinh tổng hợp.

## Hệ quả
- (+) Văn phong đầu ra nhất quán hơn, kỳ vọng mạch lạc/trau chuốt tốt hơn.
- (+) Kích thước dữ liệu đủ lớn cho SFT chuyển văn phong (483 >> mức tối thiểu ~150–300).
- (−) Bỏ khả năng sinh cổ tích dài (chấp nhận — ngoài phạm vi; data cổ tích để dành).
- Khi đổi tập dữ liệu này, **học lại từ base**, không "học tiếp" trên adapter cũ (tránh trôi dạt tích lũy).

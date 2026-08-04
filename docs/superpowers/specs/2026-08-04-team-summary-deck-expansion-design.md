# Spec: Mở rộng deck tóm tắt nhóm (tinystory-vn-summary)

Ngày: 2026-08-04 · Người duyệt: trieulh (E1) · Phạm vi: `output/presentation/tinystory-vn-summary.pptx`.

## Mục tiêu

Mở rộng và viết lại deck tóm tắt nhóm 16 từ 15 slide lên **23 slide** với ba cải thiện:
1. Phần mở đầu nói rõ đề tài, dữ liệu + cách cài cắm điều kiện, và hướng giải quyết.
2. Mỗi hướng E1–E5 có 3 slide đồng đều: kiến trúc & hướng nghiên cứu, tiến trình
   (kèm biểu đồ mốc), kết quả & phân tích — rút từ báo cáo cá nhân của mỗi người.
3. Văn phong báo cáo học thuật, câu hoàn chỉnh, khẳng định có mức độ, tránh cụt lủn/
   quá tự tin; giữ trung thực số liệu và caveat từ báo cáo gốc.

## Bố cục 23 slide

**Mở đầu (5):**
1. Bìa (giữ).
2. Đề tài & bài toán — tác vụ sinh truyện có điều kiện; điều kiện phải *chi phối diễn
   biến* chứ không chỉ xuất hiện; RQ1–RQ4.
3. Dữ liệu & cách cài cắm điều kiện — TF1-EN-3M (`klusai/ds-tf1-en-3m`); định dạng
   `<5 trường> <|story|> truyện <|end|>`; hai giao diện (5 trường cho E1/E3/E4/E5;
   character+moral cho E2); loss-masking phần điều kiện (from-scratch).
4. Hướng giải quyết — ba nhóm phương pháp (tiền huấn luyện từ đầu / PEFT-QLoRA / kiểm
   soát đầu ra) → E1–E5 + hạ tầng app + giám khảo Gemma + 25 đề chung.
5. Ba lớp bằng chứng — độ giống dữ liệu / mức độ dùng điều kiện / độ tin cậy hệ thống.

**Năm E × 3 slide (15):** mỗi E gồm
- A · Kiến trúc & hướng nghiên cứu (mô hình nền, bảng cấu hình / sơ đồ, giả thuyết).
- B · Tiến trình nghiên cứu (các mốc tuần tự + biểu đồ tiến độ).
- C · Kết quả & phân tích (số nội bộ + vòng chung + kiểu lỗi + giới hạn).

**Tổng hợp (3):**
15. Bảng định lượng 25 đề (E4 9.20, E5 8.44, E1 3.30, E2 3.18, E3 2.81; E4−E5 +0.76).
16. Ba lớp bằng chứng không thay thế nhau.
17. Kết luận + hạn chế + bước tiếp.

## Nguồn nội dung & biểu đồ

| E | Báo cáo | Biểu đồ (figures/tracks + báo cáo cá nhân) |
|---|---|---|
| E1 | `trieulh/report/report.md` | `e1_training_curves`, `04_scaling_law`, `11_score_progression`, `19_headtohead_progression`, `llama-vs-gpt2-architecture.svg` |
| E2 | `report.md` (gốc, tungd) | `e2_training_curves`, `e2_tokenizer_metrics`, `e2_v3_failure_chain`, `e2_v10_causal_replay`, `e2_v15_representation_transfer`, `e2_v14_causal_epochs` |
| E3 | `thanhnc/report/report.vi.md` | `e3_train_dynamics`, `e3_lora_ablation`, `e3_human_eval` |
| E4 | `lienntp/results/FINAL_EXPERIMENT_REPORT.md` | `e4_lora_ablation`, `e4_human_eval`, `automatic_reliability.svg` |
| E5 | `hoangndl/report.md` | `e5_training_curves` |

## Ràng buộc kỹ thuật

- Giữ template/master hiện tại (màu, font, footer "TINYSTORY-VN · NHÓM 16 · IT5410").
- Nhân bản slide mẫu bằng `scripts/add_slide.py`; sửa nội dung; giữ chart native/hình.
- SVG → chuyển PNG trước khi nhúng (python-pptx không đọc SVG).
- `scripts/office/validate.py` phải PASS; render thumbnail để QA.
- Ghi đè `output/presentation/tinystory-vn-summary.pptx`; sao lưu bản cũ trước;
  xuất PDF `output/pdf/` nếu cần.

## Tiêu chí hoàn thành

- 23 slide đúng bố cục, mở được trong PowerPoint (validate PASS).
- Mỗi E có kiến trúc + tiến trình (biểu đồ) + kết quả; số liệu khớp báo cáo gốc.
- Văn phong đồng nhất, chuyên nghiệp; không xóa/sai lệch nội dung của thành viên khác.

# Design: Step 5B v2 — ngưỡng diễn giải trên biểu đồ + bảng verdict tự động + comment tham số

Ngày: 2026-07-08
Trạng thái: Approved
Branch: `feat/slm-pretrain-tf1`
Tiền thân: `2026-07-08-step5b-analysis-dashboard-design.md`. Bản này nâng cấp dashboard để "đọc 5 giây biết train hiệu quả chưa" và giải thích tham số đến mức ngưỡng/hệ quả.

## 1. Quyết định đã chốt (grill 2026-07-08)

| Chủ đề | Quyết định |
|---|---|
| Diễn giải ngưỡng | **Cả hai**: annotate trực tiếp lên biểu đồ (vùng màu, mốc neo, % gap, PASS trong tiêu đề) + bảng verdict tự động cuối dashboard |
| Loss-gain | **Thay bằng log-log loss vs step** + fit tuyến tính (kiểm chứng power-law, Kaplan 2020) — gain chỉ là ảnh lật của loss, không thêm thông tin |

## 2. Thay đổi từng figure

### Fig 1 — Training dynamics
- Subplot loss: 3 vùng nền — `>2.0` undertrained (đỏ nhạt), `1.5-2.0` partial (vàng nhạt), `<1.5` target (xanh nhạt). Ngưỡng là heuristic cho setup này (30M, vocab 12k, TF1), ghi rõ ở markdown.
- Subplot 2 (thay loss-gain): log-log loss vs step, bỏ ~10% điểm đầu (warmup), fit `np.polyfit` → annotate hệ số mũ + R². Gần tuyến tính (R² > 0.95) = đúng power-law regime.
- LR (WSD) + run summary giữ nguyên.

### Fig 2 — Intrinsic quality
- Distinct-1/2 gộp một subplot (4 bar), annotate **gap tương đối** so với real; `<= 15%` = PASS.
- Self-BLEU: annotate **gap tuyệt đối** (giá trị real quá nhỏ nên gap tương đối nổ vì nhiễu); `<= 0.05` = PASS.
- Flesch: subplot riêng, vùng mục tiêu **80-100** (chuẩn truyện thiếu nhi).
- Histogram độ dài: annotate **hệ số chồng lấp phân bố** (sum of min of normalized hists); `>= 50%` = PASS.

### Fig 3 — LM behavior
- Perplexity: trục log, 3 bar — **ceiling** = vocab size 12000 (đoán mò uniform), **floor** = e^(final train loss) (kỳ vọng lý thuyết), và ppl held-out của SLM. Tốt = sát floor, thấp hơn ceiling nhiều bậc.
- Per-position loss: đổi từ 64 token đầu sang **16 bin vị trí tương đối 0-100%** của truyện.
- Zipf giữ nguyên.

## 3. Bảng verdict tự động (cell mới, cuối dashboard)

In bảng text (không emoji): metric / value / target / verdict (PASS-WARN-FAIL), và merge vào `analysis_30M.json`:

| Metric | PASS | WARN |
|---|---|---|
| final train loss | < 1.5 | 1.5-2.0 |
| scaling-law fit R² | > 0.95 | > 0.90 |
| held-out ppl / floor | <= 1.5x | <= 3x |
| Distinct-1/2 gap vs real | <= 15% | <= 30% |
| Self-BLEU abs gap | <= 0.05 | <= 0.15 |
| Flesch | 80-100 | 60-80 |
| length overlap | >= 50% | >= 30% |

Verdict cell **tự tính lại** mọi số dẫn xuất (fit, overlap) từ dict `analysis` để không phụ thuộc thứ tự chạy các cell figure.

## 4. Comment tham số

- **HYPERPARAMS**: mỗi tham số ghi ý nghĩa + khoảng an toàn cho setup này + hệ quả khi lệch (vd PEAK_LR >5e-3 dễ spike/phân kỳ, <1e-3 hội tụ chậm; WARMUP_FRAC <1% spike đầu run; DECAY_FRAC <10% bỏ lỡ cú giảm loss cuối...).
- **Harness**: comment N_GEN (30 đủ xu hướng, 100+ cho error bar báo cáo), N_PPL (200 ổn định ±2%), GEN_TEMP (cao → đa dạng nhưng lỗi ngữ pháp; thấp → lặp), GEN_MAXNEW, POS_BINS.
- **Markdown Step 5b**: mục "How to read" cho 3 figure + verdict, nêu rõ ngưỡng là heuristic.

## 5. Ràng buộc
- Không đổi data/train recipe. Không emoji, không em dash. Nhãn tiếng Anh.
- Lưu bản `.ipynb` local vào `notebooks/` (đồng bộ với bản Drive) để commit git.

## 6. Tiêu chí thành công
- Mỗi biểu đồ tự nói được "đạt hay chưa" không cần người đọc biết trước ngưỡng.
- Bảng verdict in cuối + nằm trong `analysis_30M.json`.
- Đọc cell HYPERPARAMS hiểu được vì sao chọn từng giá trị và lệch thì sao.

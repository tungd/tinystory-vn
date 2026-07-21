# Nghiên cứu hướng cải thiện SLM 30M — bám Materials môn học (Week 8-10)

Ngày: 2026-07-20 | Nguồn: đọc trực tiếp Week8-Diffusion, Week9-RL, Week10-Deep-Q + phát hiện thực nghiệm của đồ án.

## Reframe: nút thắt thật (từ chuỗi thí nghiệm)

Model 30M **đủ năng lực sinh bản chất lượng cao** (best-of-N: mean 7.7 → best-of-3 8.5, sát Qwen 9.75) — **không phải trần capacity**. Nút thắt là **(a) tính nhất quán của phân bố mặc định** và **(b) tín hiệu huấn luyện yếu**: DPO (~200 pairs, judge Qwen-4B) NULL, SFT-on-best NULL. Cần cơ chế có **exploration + reward tuyệt đối**, thứ mà DPO/SFT thiếu.

## Material nào cho hướng nào

- **Week10 (Deep-Q) — mỏ vàng**: ngoài DQN, dạy hẳn **Policy Gradient (tr.11-13), REINFORCE (14-17), Variance Reduction + Baseline (18-19), Actor-Critic (20-21)**. Đây là nền tảng "đúng bài" cho hướng reward-guided.
- **Week9 (RL)**: nền tảng RL cổ điển (reward hypothesis tr.7, exploration/exploitation tr.28-30, policy tr.19). Lý giải: DPO/SFT là **pure exploitation** trên data cố định → thiếu exploration → null. Best-of-N thắng vì nó LÀ exploration.
- **Week8 (Diffusion)**: chủ yếu cho dữ liệu liên tục (ảnh), **không áp dụng trực tiếp** cho text rời rạc. Rút được 1 analog inference-time: **Classifier-Free Guidance → contrastive decoding** (steer logits theo điều kiện). DQN trực tiếp trên token: **không khả thi** (action space = vocab quá lớn, xác nhận Week10 tr.5).

## Các hướng khả thi (xếp hạng theo ROI × bằng chứng × khả thi T4/M3)

| # | Hướng | Material | Vì sao có thể thắng nơi DPO/SFT null | Khả thi |
|---|---|---|---|---|
| 1 | **GRPO-lite = REINFORCE + baseline** | Week10 tr.18-19 | Reward TUYỆT ĐỐI (judge score) + exploration (rollout mới mỗi step) + baseline `(r_i − mean(r))` giải quyết "all-positive reward". ~50 dòng PyTorch, không cần lib RL. Chính là cơ chế DeepSeek-R1. | Có (T4/M3); chi phí = gọi judge mỗi rollout |
| 2 | **Reward-weighted SFT (RAFT) có NGƯỠNG + QUY MÔ** | Week9 tr.7 (reward hypothesis) | Kiểm tra lại data: corpus SFT cũ (42 truyện) thực ra 93% đã ≥8.5 — vậy biến số thiếu KHÔNG chỉ là ngưỡng mà là QUY MÔ (42 truyện ≈ 12k token, quá nhỏ so 600M token pretrain). RAFT vòng 1: 200 truyện ≥9.0 (105 harvest từ thí nghiệm cũ + sinh bổ sung), lr 2e-5. | Có (dễ nhất) |
| 3 | **Best-of-N (đã deploy)** | Week9 exploration tr.28-30 | Đã đo: mean 7.7 → 8.5. Không train. | Đã xong |
| 4 | **Reward-model nhỏ làm Critic/reranker** | Week10 Actor-Critic tr.20-21 | Train RM 30M trên (story, judge-score) → rerank best-of-N NHANH (khỏi gọi Qwen mỗi lần) → cho phép tăng N. Cũng dùng làm baseline cho GRPO. | Có |
| 5 | **Contrastive decoding (CFG-analog)** | Week8 tr.44-49 | `logits = logits_uncond + γ(logits_cond − logits_uncond)` → ép bám 5 slot mạnh hơn. Inference-time, không train lại (model đã có slot-dropout → có nhánh uncond). Nhắm thẳng adherence. | Có (~2x latency/token) |
| — | DQN trực tiếp / Diffusion-LM from scratch | Week10 tr.5 / Week8 | Action space quá lớn / sai objective + quá tốn compute | KHÔNG |

## Khuyến nghị lộ trình

1. **Ngay (inference, rẻ)**: giữ best-of-N (#3) + thử **contrastive decoding (#5)** cho adherence + **RM critic (#4)** để best-of-N nhanh.
2. **Nâng cấp "đúng bài" (huấn luyện)**: **RAFT có ngưỡng (#2)** trước (đơn giản, ổn định) → nếu muốn mạnh hơn thì **GRPO-lite (#1)** — đây là hướng bám Week10 nhất và là câu chuyện khoa học đẹp (khép vòng: material dạy REINFORCE+baseline → áp vào chính bài toán).

## Rủi ro cần lường
- GRPO/REINFORCE ở 30M có thể bất ổn (variance cao) + tốn chi phí judge mỗi rollout → cần baseline + có thể cần RM nội bộ (#4) làm reward rẻ.
- Judge Qwen-4B vẫn là điểm yếu tín hiệu (đã thấy ở DPO). RM huấn luyện tốt hoặc judge mạnh hơn sẽ giúp mọi hướng reward-based.

# Design: Alignment chương cuối — ORPO trên preference pairs (RLAIF) cho SLM 30M

Ngày: 2026-07-11
Trạng thái: Approved (grill cùng ngày)
Branch: `feat/slm-pretrain-tf1`
Điều kiện tiên quyết: pha 2 hoàn tất (notebook `pretrain_slm_30m_dashboard.ipynb`, user chạy tối 2026-07-11) -> có `30M-p2` + `slm-30m-p2.gguf`.

## 1. Mục tiêu & vị trí trong câu chuyện khoa học

Chuỗi báo cáo: **pretrain (pha 1) -> chẩn đoán định tính -> data fix + annealing (pha 2) -> alignment (ORPO) -> đánh giá cuối**. Mỗi bước có delta đo được. Chương ORPO chứng minh: model nhỏ sau pretrain có thể tự cải thiện adherence/enactment bằng preference optimization với **AI feedback** (RLAIF) — không cần người gán nhãn, không cần API trả phí.

Cơ sở: ORPO (Hong et al. 2024) — odds-ratio preference optimization, KHÔNG cần reference model (nhẹ hơn DPO, hợp T4); DPO (Rafailov et al. 2023) làm đối chiếu khái niệm; RLAIF (Bai et al. 2022, Constitutional AI). Rubric judge 4 trục theo TF1-EN-3M (arXiv:2504.20605, `app/judge.py`).

## 2. Quyết định đã chốt (grill 2026-07-11)

| Chủ đề | Quyết định |
|---|---|
| Kỹ thuật | **ORPO** (TRL), không cần reference model |
| Prompt nguồn | **1500 prompt** từ TF1 split `test`, LẤY SAU 500 record đầu (offset 500+) để không đụng bộ eval hiện tại |
| Sinh pairs | 2 truyện/prompt từ `slm-30m-p2` qua Ollama local (temp 0.8, seed khác nhau) — SLM 915 tok/s nên nhanh |
| Judge preference | **1 judge `qwen3:4b` local** chấm rubric 4 trục cho từng bản; chỉ giữ pair khi chênh overall >= 1.0 (lọc độ tự tin). Kỳ vọng giữ ~60-70% -> ~1000 pairs sạch |
| Train ORPO | Colab T4 (TRL `ORPOTrainer`), khởi từ `30M-p2`, 1-2 epoch trên ~1000 pairs, LR nhỏ (~1e-5), theo dõi perplexity held-out để tránh catastrophic forgetting. Artifact hậu tố `-orpo` |
| Đánh giá cuối | Batch eval panel 3 judge (`scripts/eval_slm.py`) với **--limit 15** (giảm từ 30 theo yêu cầu — kết luận theo rank nên 15 đủ; ~30 phút local) cho {`slm-30m-orpo`, `base-qwen3-4b`} -> `results/eval_summary.json` -> Results tab |
| Kiểm soát hồi quy | So perplexity held-out trước/sau ORPO (không được tăng quá ~10%); nếu vỡ văn phong -> giảm epoch/LR |

## 3. Pipeline & phân công hạ tầng

```
[Local, qua đêm]  gen 1500x2 truyện (ollama slm-30m-p2) -> judge qwen3:4b từng bản
                  -> pairs.jsonl (chosen/rejected + score, lọc chênh >= 1.0)
[Colab T4]        ORPO train (~30-40 phút) -> 30M-orpo -> GGUF slm-30m-orpo + Modelfile
[Local]           ollama create slm-30m-orpo -> đánh giá định tính 10 prompt
                  -> batch eval panel 3 judge --limit 15 -> eval_summary.json -> app Results
```

Script mới:
- `scripts/gen_preference_pairs.py` — sinh + judge + lọc, ghi `data/orpo/pairs.jsonl` (chosen, rejected, prompt, score_chosen, score_rejected). Resume-safe (append, skip prompt đã có).
- `scripts/orpo_train.py` hoặc cell notebook — TRL ORPOTrainer, load 30M-p2, format pair theo template train (`cond \n <|story|> story <|end|>`).

## 4. Tiêu chí thành công

- >= 800 pairs sau lọc.
- ORPO xong không hồi quy: perplexity held-out tăng <= 10% so với 30M-p2.
- Đánh giá định tính 10 prompt: adherence/moral-enactment tăng so với p2 (ước >= +1 điểm trục adherence).
- Batch eval 15 prompt: `slm-30m-orpo` cải thiện overall so với điểm p2; khoảng cách với Qwen thu hẹp so với pha 1 (2.5 -> ?).
- Nhật ký thử nghiệm cập nhật delta từng nấc: pha 1 -> pha 2 -> ORPO.

## 5. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Catastrophic forgetting ở 30M | LR nhỏ, 1-2 epoch, guard perplexity, giữ checkpoint p2 để rollback |
| Judge đơn thiên vị/nhiễu | Lọc chênh >= 1.0; panel 3 judge chỉ dùng ở đánh giá cuối (độc lập với judge tạo data) |
| Pairs đồng nhất (2 bản giống nhau) | temp 0.8 + seed khác; nếu tỷ lệ loại cao, tăng temp bản thứ hai lên 1.0 |
| Quota GPU | ORPO chỉ ~30-40 phút; pair-gen + judge chạy local hoàn toàn |

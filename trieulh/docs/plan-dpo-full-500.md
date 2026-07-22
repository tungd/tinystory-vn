# Plan: Full DPO alignment với 500 preference pairs

Ngày: 2026-07-20 | Branch: `feat/trieulh-improve`
Mục tiêu: nâng prompt-adherence (hạn chế chính, hiện ~76% sau trial 115 pairs) bằng DPO đầy đủ trên 500 pairs. Bằng chứng khả thi: trial 115 pairs đã +5đ adherence, reward-acc 1.0, ppl 0% drift.

## Các bước

1. **Resume pair-gen tới 500** (local, judge `qwen3-4b-instruct`, `--min-margin 1.0 --max-pairs 500`, resume-safe, tự dừng). Từ 115 pairs hiện có -> ~895 prompt nữa (~4h, rải phiên được). Output `data/orpo/pairs.jsonl`.
2. **DPO full local** (`trieulh/scripts/dpo_train_local.py`, MPS, 2 epoch, lr 5e-6) trên 500 pairs -> `out/30M-dpo` (ghi đè trial). Guard perplexity held-out (<= +10%).
3. **Export + nạp app**: GGUF q8 (llama.cpp + chkhsh patch) -> `ollama create slm-30m-dpo` -> giữ registry id `slm-30m-dpo`. Smoke test.
4. **Đánh giá**: adherence probe (slot-recall) p2 vs dpo-500 trên >=12 prompt; completeness; so với trial 115 pairs (+5đ). Kỳ vọng delta mạnh hơn.
5. **Cập nhật báo cáo + log**: ghi kết quả 500-pair vào `trieulh/docs/experiments/`, cập nhật figure adherence + report; đóng gói model -> Drive (zip + lấy link qua MCP).

## Tiêu chí thành công
- >= 400 pairs sau lọc (mục tiêu 500).
- Adherence tăng rõ hơn trial (>= +5đ so p2, tốt hơn 76%), completeness giữ, ppl drift <= +10%.
- Model + báo cáo cập nhật, PR mới vào main.

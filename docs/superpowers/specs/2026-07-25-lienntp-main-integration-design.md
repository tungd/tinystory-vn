# Spec: Tích hợp feat/lienntp vào main (integration merge)

Ngày: 2026-07-25. Người duyệt: trieulh. Phương án: **B — integration branch**.

## Bối cảnh

`feat/lienntp` (2 commit, +34.7k dòng) tách từ main cũ (0ad3745), 5 commit trước HEAD
hiện tại của main. Hai bên đã sửa chồng lấn 13 file dùng chung. Đề tài hai người độc lập
(lienntp: fine-tune Llama 3.2 3B, SFT/LoRA, generation-mode Raw/Post/Repair; trieulh:
SLM from scratch, best-of-N), nhưng app dùng chung bị lienntp đổi defaults + thay sạch
registry.

## Nguyên tắc đã chốt với trieulh

1. **Registry union**: `config/models.json` gộp cả 11 model (5 của main + 6 của lienntp).
2. **Defaults theo main**: `JUDGE_MODEL_ID=base-qwen3-4b`, GenReq `model_id=base-qwen3-4b`,
   giữ GEN_* của main. Thêm `REPAIR_MODEL_ID` (lienntp). lienntp đổi default trên máy
   mình bằng env var (`FABLE_JUDGE_MODEL_ID`, `FABLE_BASE_MODEL`, ...).
3. **Hai feature app cùng tồn tại**: `best_of_n` (main) và `generation_mode`
   raw/postprocess/repair (lienntp) đều có trong GenReq + UI.
4. **Chấp nhận deletions của lienntp** (frontend/ vanilla, 2 notebook finetune cũ,
   scripts/prepare_data.py + test): đã kiểm chứng tungd tự xóa 4/5 trên feat/td,
   thanhnc không dùng, trieulh không tham chiếu.
5. **Không đụng branch gốc của lienntp** (không rebase/force-push).

## Quy trình

1. Branch `feat/lienntp-integration` từ `origin/feat/lienntp`.
2. `git merge origin/main`, resolve theo ma trận dưới.
3. Kiểm chứng: `pytest` toàn suite; smoke app 3 đường (generate slm-60m, best-of-N=3,
   generation_mode=repair với fallback khi thiếu model repair).
4. PR `feat/lienntp-integration` -> main, squash-merge.

## Ma trận resolve

| File | Resolve |
|---|---|
| `config/models.json` | Union 11 entry, thứ tự: main trước (qwen3-4b, slm-*), lienntp sau |
| `app/config.py` | Base = main; + `REPAIR_MODEL_ID` từ lienntp; `REQUEST_TIMEOUT_SECONDS=300`; LENGTH_HINTS: giữ bản main |
| `app/main.py` | Base = main (best-of-N, objective eval); ghép từ lienntp: import `enhance_story` + `REPAIR_MODEL_ID`, field `generation_mode`, `_req_prompt_row`, `_resolve_repair_model`, nhánh xử lý mode trong stream + meta; default model_id giữ base-qwen3-4b |
| `app/prompt_en.py` | Base = main (LENGTH_NUM_PREDICT); ghép thay đổi cộng thêm của lienntp nếu không đổi hành vi SLM |
| `app/ollama_client.py` | Union: main (gmeta/best-of-N) + lienntp (repair options) |
| `web/src/components/InputPanel.tsx` | Base = main (Best-of-N); thêm block Mode selector + `generation_mode` vào FablePayload |
| `web/src/api.ts`, `ObservabilityPanel.tsx`, `App.tsx` | Union type/meta hai bên |
| `README.md` | Khung main; thêm dòng lienntp vào bảng "Báo cáo theo người"; thêm mô tả generation mode ở phần tính năng |
| `tests/*` | Union hai bên; bỏ test_prepare_data.py (theo deletion) |
| `pyproject.toml`, `.gitignore` | Union |
| deletions | Giữ deletion của lienntp |

## Tiêu chí hoàn thành

- pytest pass toàn bộ (trừ 6 test của test_prepare_data.py đã xóa có chủ đích).
- App khởi động, `/models` trả 11 entry, generate + best-of-N + mode đều hoạt động
  (repair fallback mềm khi máy thiếu model của lienntp).
- Diff của PR không xóa bất kỳ file nào thuộc `trieulh/` hoặc số liệu báo cáo.

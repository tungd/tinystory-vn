# Design: Can thiệp data + resume 2 pha cho SLM 30M (phase 2)

Ngày: 2026-07-11
Trạng thái: Approved
Branch: `feat/slm-pretrain-tf1`
Căn cứ: đánh giá định tính 2026-07-11 (training log) — 4 vấn đề còn lại sau sampling fix; khảo sát data: "wise old owl" chiếm 28% fable thật nhưng model sinh 90% (mode amplification); slot dropout p=0.3 khiến Teaching/Outcome chỉ xuất hiện ~67% trong conditioning -> moral substitution + outcome bị bỏ.

## 1. Quyết định đã chốt

| Chủ đề | Quyết định |
|---|---|
| Data fix 1 | **Cap template**: giới hạn tỷ lệ example có "wise old owl" trong fable xuống **10%** corpus (cơ chế cap-phrase generic trong `prepare_tf1_pretrain.py`) |
| Data fix 2 | **Rebalance slot dropout**: `teaching`/`outcome` p=**0.15**, các slot khác giữ 0.3, `p_all` giữ 0.05 (mở rộng `apply_dropout` nhận override per-slot, TDD) |
| Tokenizer | **Giữ nguyên** artifact BPE 12k (đổi sẽ vô hiệu checkpoint + GGUF hash) |
| Training | **Resume 2 pha**: nạp checkpoint-1500 từ `ckpt_30M` Drive, STEPS 1800 -> **3600**, pha 2 chạy trên corpus mới (annealing trên data sạch, kiểu MiniCPM/Llama-3 mid-training). `ignore_data_skip=True` để không bỏ phí 192k example đầu của corpus mới |
| Hạ tầng | **Colab CLI** (session `slm`, T4): drivemount, upload runner script, exec, monitor. Không dùng colab-mcp |
| Vị trí Drive | Folder user chỉ định (id `1N852R4wZ_QUq8PruO0uULLIT7VqL4sqv`) — định vị lúc runtime bằng cách tìm thư mục chứa `ckpt_30M`; mọi artifact mới ghi vào đó |
| Versioning artifact | KHÔNG ghi đè artifact Run 3: corpus mới cache ở `data_tf1_v2/`, model `30M-p2/`, `loss_log_30M_p2.json`, `analysis_30M_p2.json`, fig `*_30M_p2.png`, GGUF `slm-30m-p2.gguf` + `Modelfile-30M-p2`. App chỉ trỏ sang bản p2 sau khi re-eval xác nhận tốt hơn |

## 2. Thay đổi code (TDD)

- `scripts/tf1_pretrain/format.py` — `apply_dropout(slots, rng, p=0.3, p_all=0.05, p_overrides=None)`: override xác suất dropout theo slot; back-compat (mặc định không đổi hành vi).
- `scripts/prepare_tf1_pretrain.py` — cap-phrase: CLI `--cap-phrase "wise old owl" --cap-frac 0.10`; trong `_write_split`, example có phrase (lowercase, so trong fable) bị skip nếu số example-có-phrase đã đạt `cap_frac` của tổng đã ghi. CLI `--slot-dropout teaching=0.15 outcome=0.15` (k=v lặp lại) truyền vào `p_overrides`.
- Test: `tests/test_prepare_tf1.py` (hoặc file test sẵn có của module) — dropout override (p=1/p=0 xác định), cap-phrase trên fixture nhỏ.

## 3. Runner phase 2 (chạy trên VM qua CLI)

`scripts/colab_phase2.py` — tự chứa, chạy tuần tự: clone repo (branch này, đã push) -> cài deps -> định vị `DRIVE` (tìm `ckpt_30M`) -> khôi phục tokenizer từ cache cũ -> build corpus mới 400k với 2 fix -> cache `data_tf1_v2` -> encode -> train STEPS=3600 resume từ `ckpt_30M` (checkpoint pha 2 ghi `ckpt_30M_p2`) -> lưu `30M-p2` (local + Drive) -> harness phân tích (như Step 5b, hậu tố p2) -> verdict -> export GGUF p2. Mọi print flush để CLI stream được log.

## 4. Tiêu chí thành công

- Corpus mới: tỷ lệ "wise old owl" ~10%; thống kê slot presence Teaching/Outcome ~85%.
- Loss cuối pha 2 < 1.40 (từ 1.447).
- Re-eval 10 prompt: tỷ lệ truyện có owl giảm rõ (<50%), outcome/moral yêu cầu được bám tốt hơn; verdict >= 7 PASS.
- Artifact p2 đầy đủ trên folder Drive user chỉ định; Run 3 artifacts còn nguyên.

## 5. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Resume + data mới làm loss nhảy lúc đầu | Bình thường với distribution shift nhẹ; theo dõi 100 step đầu, spike lớn thì hạ LR |
| Quota GPU cắt giữa pha 2 | checkpoint `ckpt_30M_p2` mỗi 500 step + resume như cũ |
| Cap owl làm corpus thiếu | Pool 2.8M dư; prepare stream tiếp tới khi đủ 400k |
| VM path Drive khác dự kiến | Runner tự dò `find ... -name ckpt_30M`, fail-fast với thông báo rõ |

# Design: Đảm bảo truyện sinh ra hoàn thiện (single-shot) cho SLM 30M

Ngày: 2026-07-16
Trạng thái: Approved (chờ review spec)
Branch: `feat/slm-pretrain-tf1`
Bối cảnh: model 30M-p2 đôi lúc sinh truyện cụt giữa câu (vd kết ở "...knowing that she had"). Nguyên nhân đã chẩn đoán: truyện đụng trần `num_predict` (Short=300) trước khi model kịp phát `<|end|>`, cộng cốt truyện lan man không hội tụ.

## 1. Mục tiêu & ràng buộc

Ưu tiên: **nội dung hoàn thiện** > hội tụ (không lan man) > độ dài (thấp, không bắt buộc) > đa dạng (được phép giảm). **Sinh một lần mỗi lần bấm Generate** (không best-of-N, không auto-regenerate).

Ràng buộc kiến trúc cứng: model train `seq_len=512`, Modelfile `num_ctx=512`. Prompt đo thực tế ~52 token (2 slot) tới ~110 token (5 slot), nên **story headroom ~400-460 token** (~250-320 từ). Vượt 512 = ra ngoài context đã train (RoPE chưa học vị trí >512) -> sinh rác. Do đó KHÔNG giải bằng tăng token/continuation; phải để model tự kết trong budget.

Tín hiệu sẵn có (đang bị bỏ): Ollama trả `done_reason`. `"stop"` = model phát `<|end|>` (Ollama nuốt token này khi stop) = **kết thật, hoàn thiện**. `"length"` = đụng trần = cụt.

## 2. Bốn thay đổi (single-shot, giữ streaming)

### 2.1 Right-size `num_predict` về đúng headroom (`app/config.py`)
`LENGTH_NUM_PREDICT`: `short 300->400`, `medium 600->440`, `long 1100->460`. Bỏ các giá trị ảo (600/1100 vốn vượt context). 460 sát headroom khi prompt ngắn; prompt dài hơn thì Ollama tự cắt ở context 512 và lưới trim (2.4) xử lý.

### 2.2 Sampling hội tụ hơn (`app/config.py`)
`GEN_TEMPERATURE 0.8->0.7`, `GEN_TOP_P 0.9->0.85`. Bám nhánh xác suất cao -> cốt truyện đi thẳng tới đoạn kết đã thấy trong training, ít mở nhánh. Giữ `GEN_REPEAT_PENALTY=1.1` (cao hơn gây entity drift - đã biết). Đo thực nghiệm: temp 0.7 + np 400 cho truyện hoàn thiện (`done_reason=stop`, 320 token, kết bằng moral).

### 2.3 Kéo mốc độ dài về mức model kết được (`app/prompt_en.py` `LENGTH_HINT_EN`)
Hint hiện tại "long 450-600 words" vượt khả năng (model chỉ nhét ~300-350 từ trong context) -> chính long dễ cụt nhất. Sửa:
- short: "Keep it short (about 120-160 words)."
- medium: "Write a medium-length fable (about 200-260 words)."
- long: "Write a fuller fable (about 280-340 words)."
Cả ba đều hoàn thiện được trong headroom; giữ gradient tương đối.

### 2.4 Chuyển `done_reason` ra + lưới trim (`app/ollama_client.py`, `app/main.py`, util mới)
- `ollama_client.generate_stream`: hiện chỉ yield string. Bổ sung một cách chuyển `done_reason` của chunk cuối ra caller (vd nhận một callback `on_done(reason)` hoặc yield chunk cuối là sentinel `{"done_reason": ...}`; chọn callback để không phá kiểu Iterator[str] mà test khác dựa vào). `generate_meta`: thêm `done_reason` vào dict trả về (đã có `data`).
- Util mới `trim_to_last_sentence(text) -> str` (pure, TDD): nếu text kết bằng dấu câu hoàn chỉnh (`. ! ? "` hoặc `."`/`!"`/`?"`) thì giữ nguyên; nếu không, cắt về vị trí dấu kết câu cuối cùng. Nếu không tìm thấy dấu nào (hiếm) thì giữ nguyên text.
- `app/main.py` cả 2 nhánh:
  - Guardrail OFF (stream): sau vòng stream, nếu `done_reason=="length"` -> `story = trim_to_last_sentence(story)`; sự kiện `done` gửi `story` đã làm sạch (frontend dùng lại field `story` sẵn có để thay text đã stream); thêm 1 step log "output trimmed to last complete sentence (hit context limit)".
  - Guardrail ON (meta): tương tự trên `meta["text"]` trước khi qua output-check.
  - KHÔNG gắn câu kết tự chế (giữ trung thực với output model - quyết định grill 2026-07-16).

## 3. Định vị trung thực
Mục 2.1-2.3 làm phần lớn truyện có **kết thật** (resolution + moral); 2.4 là lưới cho số ít còn cụt -> worst case là **kết ở ranh giới câu, không bao giờ cụt giữa từ**. Single-shot không đảm bảo tuyệt đối kết thật nhưng đảm bảo không cụt giữa từ + tối đa hóa tỷ lệ kết thật.

## 4. Non-goals
- KHÔNG best-of-N / regenerate (đã chốt single-shot).
- KHÔNG đổi model, tokenizer, num_ctx (giữ 512 = seq_len train).
- KHÔNG gắn câu kết tổng hợp.
- KHÔNG đụng frontend ngoài việc dựa vào field `story` trong sự kiện `done` đã có.

## 5. Testing
- `trim_to_last_sentence`: unit test (kết sạch giữ nguyên; cụt giữa câu -> cắt về dấu cuối; cụt trong dấu ngoặc kép; không có dấu -> giữ nguyên; chuỗi rỗng).
- API: `done_reason` được chuyển ra; nhánh stream/meta trim khi length và không trim khi stop (fake Ollama trả done_reason tương ứng).
- Regression: các test API hiện có vẫn pass (14 test `test_api_en.py`).

## 6. Tiêu chí thành công
- Tái tạo prompt cũ (Short, skunk/flower field): output kết bằng câu hoàn chỉnh, `done_reason=stop` (hoặc nếu length thì đã trim sạch), không còn "...knowing that she had".
- Sinh thử 10 prompt held-out: >= 8/10 có `done_reason=stop` (kết thật); 10/10 không cụt giữa từ.
- Toàn bộ test suite pass.

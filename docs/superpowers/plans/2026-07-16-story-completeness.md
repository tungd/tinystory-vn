# Story Completeness Implementation Plan

> Thực thi inline. Spec: `docs/superpowers/specs/2026-07-16-story-completeness-design.md`.

**Goal:** Single-shot sinh truyện hoàn thiện cho SLM 30M: right-size num_predict + hội tụ + done_reason + trim câu dở.

## Global Constraints
- Hằng số độ dài THẬT ở `app/prompt_en.py` (không phải config.py - bản đó unused).
- Không best-of-N/regenerate; giữ streaming; không gắn câu kết tự chế.
- `done_reason=="stop"` = hoàn thiện; `"length"` = cụt -> trim.

### T1: util `trim_to_last_sentence` (TDD)
- `tests/test_textproc.py`: kết sạch giữ nguyên; cụt giữa câu -> cắt về dấu `.!?` cuối (kể cả `."`); không có dấu -> nguyên; rỗng -> rỗng.
- `app/textproc.py`: implement pure.

### T2: sampling hội tụ (`app/config.py`)
- `GEN_TEMPERATURE` 0.8 -> 0.7; `GEN_TOP_P` 0.9 -> 0.85. Giữ repeat_penalty 1.1.

### T3: right-size length (`app/prompt_en.py`)
- `LENGTH_NUM_PREDICT`: short 400, medium 440, long 460.
- `LENGTH_HINT_EN`: short "about 120-160 words", medium "about 200-260 words", long "about 280-340 words".

### T4: surface done_reason (`app/ollama_client.py`)
- `generate_stream(..., on_done=None)`: khi chunk done -> `on_done(chunk.get("done_reason"))` nếu có, rồi break. Giữ nguyên Iterator[str].
- `generate_meta`: thêm `"done_reason": data.get("done_reason")` vào dict trả về.

### T5: wire trim ở 2 nhánh (`app/main.py`)
- Stream (guardrail OFF): holder `{"reason": None}`; truyền `on_done`; sau stream nếu `reason=="length"` -> trim + step log "trimmed to last complete sentence (hit context limit)"; recompute output_tokens; done gửi story đã trim.
- Meta (guardrail ON): nếu `result.get("done_reason")=="length"` -> trim `story` + step log; trước output-check.

### T6: test API + regression
- `tests/test_api_en.py`: fake stream có on_done gọi reason="length" + text cụt -> done.story đã trim + có step trimmed; reason="stop" -> không trim. Meta tương tự. 14 test cũ vẫn pass.

### T7: smoke + đánh giá (Claude)
- Restart uvicorn. Chạy đúng prompt Short skunk/flower field + 10 prompt held-out qua API. Claude đọc, đánh giá done_reason rate + độ hoàn thiện, đề xuất tinh chỉnh temp/num_predict nếu cần.

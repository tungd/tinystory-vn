# 3. Train a 200M fable transformer from scratch (no fine-tune)

Ngày: 2026-07-08
Trạng thái: Accepted

## Bối cảnh
Đồ án **không được phép fine-tune** base model (Qwen3-4B). Yêu cầu: "train a model to
generate stories based on keyword guidance", chạy trên Colab qua `google-colab-cli`,
output là weight checkpoint. Keyword guidance được xác nhận là **chỉ hai seed**: nhân
vật chính (`character`) + bài học (`moral`), **không RAG**, sinh pure generation.

## Quyết định
Thay vì fine-tune, **train từ đầu** một transformer ~200M (GPT2-style, `transformers`
+ `tokenizers`) trên tập con TF1-EN-3M, conditioned qua control-prefix:

```
<char> {character} </char>
<moral> {moral} </moral>
<story>
{story}
</story>
```

- **Hai notebook Colab**: A = train → checkpoint (Drive); B = eval + gen story.
- Dùng `google-colab-cli` để provision T4, upload scripts, `exec` notebook, download
  checkpoint, tear down. Cài đặt deps bằng `uv` (`uv tool install google-colab-cli`,
  `uv pip install transformers datasets tokenizers accelerate`).
- Checkpoint export sang GGUF q8 → `ollama create fable-200m` → app dùng như model
  `kind: "finetuned"` (Compare / Results không đổi).
- Đánh giá giữ nguyên methodology ADR-0002 (4 trục LLM-as-judge + metric khách quan
  Distinct/Self-BLEU/Flesch); so sánh theo thứ hạng.

## Hệ quả
- (+) Hợp quy tắc "không fine-tune" — đây là training từ đầu, weight checkpoint hợp lệ.
- (+) Tránh nhập nhằng fine-tune/base; narrative "trained a 200M model on 3M fables"
  mạnh cho đồ án generative-AI.
- (−) Chất lượng trần thấp hơn fine-tune base 4B; với fable trẻ em 4–7 tuổi là chấp nhận.
- (−) Colab free T4 huấn luyện 200M chậm hơn; dùng subset 100k–500k fables.
- App: `app/prompt_en.build_seed_prompt` sinh prefix khớp format train; `config/models.json`
  có entry `fable-200m`.

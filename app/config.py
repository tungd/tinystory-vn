import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("FABLE_MODEL", "fable-tuned")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("FABLE_TIMEOUT", "300"))
BASE_MODEL = os.getenv("FABLE_BASE_MODEL", "qwen3:4b")
TUNED_MODEL = os.getenv("FABLE_TUNED_MODEL", "fable-tuned")
ENABLE_THINKING = os.getenv("FABLE_THINK", "false").lower() == "true"
MODELS_PATH = os.getenv("FABLE_MODELS_PATH", "config/models.json")
JUDGE_MODEL_ID = os.getenv("FABLE_JUDGE_MODEL_ID", "base-qwen3-4b")
REPAIR_MODEL_ID = os.getenv("FABLE_REPAIR_MODEL_ID", "sft-llama32-3b-clean3k")
RESULTS_PATH = os.getenv("FABLE_RESULTS_PATH", "results/eval_summary.json")

# 0.7/0.85 (hạ từ 0.8/0.9): sampling hội tụ hơn -> cốt truyện bám nhánh xác suất
# cao, đi thẳng tới đoạn kết đã thấy trong training thay vì mở nhánh lan man rồi
# đụng trần context mà chưa kịp phát <|end|>. Đổi lấy giảm đa dạng (chấp nhận được).
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.7"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.85"))
# 1.1 (not 1.3): a high repeat penalty punishes re-mentioning character-name tokens,
# which makes tiny SLMs swap the main character mid-story (entity drift).
GEN_REPEAT_PENALTY = float(os.getenv("GEN_REPEAT_PENALTY", "1.1"))

LENGTH_NUM_PREDICT = {"short": 300, "medium": 600, "long": 1100}
LENGTH_HINT = {
    "short": "Write a concise fable, about 120-180 words.",
    "medium": "Write a medium-length fable, about 250-350 words.",
    "long": "Write a longer fable, about 450-600 words.",
}

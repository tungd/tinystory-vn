import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("FABLE_MODEL", "fable-tuned")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("FABLE_TIMEOUT", "120"))
BASE_MODEL = os.getenv("FABLE_BASE_MODEL", "qwen3:4b")
TUNED_MODEL = os.getenv("FABLE_TUNED_MODEL", "fable-tuned")
ENABLE_THINKING = os.getenv("FABLE_THINK", "false").lower() == "true"
MODELS_PATH = os.getenv("FABLE_MODELS_PATH", "config/models.json")
JUDGE_MODEL_ID = os.getenv("FABLE_JUDGE_MODEL_ID", "base-qwen3-4b")

GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.8"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.9"))
GEN_REPEAT_PENALTY = float(os.getenv("GEN_REPEAT_PENALTY", "1.3"))

LENGTH_NUM_PREDICT = {"short": 300, "medium": 600, "long": 1100}
LENGTH_HINT = {
    "short": "Hãy viết truyện NGẮN GỌN, khoảng 150-250 từ.",
    "medium": "Hãy viết truyện VỪA PHẢI, khoảng 350-450 từ.",
    "long": "Hãy viết truyện DÀI, khoảng 600-800 từ.",
}

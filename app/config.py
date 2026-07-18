import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("FABLE_MODEL", "llama3.2:3b-instruct-fp16")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("FABLE_TIMEOUT", "300"))
BASE_MODEL = os.getenv("FABLE_BASE_MODEL", "llama3.2:3b-instruct-fp16")
TUNED_MODEL = os.getenv("FABLE_TUNED_MODEL", "fable-tuned")
ENABLE_THINKING = os.getenv("FABLE_THINK", "false").lower() == "true"
MODELS_PATH = os.getenv("FABLE_MODELS_PATH", "config/models.json")
JUDGE_MODEL_ID = os.getenv("FABLE_JUDGE_MODEL_ID", "base-llama32-3b-instruct")
REPAIR_MODEL_ID = os.getenv("FABLE_REPAIR_MODEL_ID", "sft-llama32-3b-clean3k")
RESULTS_PATH = os.getenv("FABLE_RESULTS_PATH", "results/eval_summary.json")

GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.8"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.9"))
GEN_REPEAT_PENALTY = float(os.getenv("GEN_REPEAT_PENALTY", "1.3"))

LENGTH_NUM_PREDICT = {"short": 300, "medium": 600, "long": 1100}
LENGTH_HINT = {
    "short": "Write a concise fable, about 120-180 words.",
    "medium": "Write a medium-length fable, about 250-350 words.",
    "long": "Write a longer fable, about 450-600 words.",
}

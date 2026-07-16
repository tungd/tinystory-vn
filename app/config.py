import os

# Backend: "ollama" (default) or "openai" (MLX server / llama.cpp / any OpenAI-compatible API)
BACKEND = os.getenv("FABLE_BACKEND", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("FABLE_MODEL", "fable-tuned")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("FABLE_TIMEOUT", "120"))
BASE_MODEL = os.getenv("FABLE_BASE_MODEL", "qwen3:4b")
TUNED_MODEL = os.getenv("FABLE_TUNED_MODEL", "fable-tuned")
ENABLE_THINKING = os.getenv("FABLE_THINK", "false").lower() == "true"
MODELS_PATH = os.getenv("FABLE_MODELS_PATH", "config/models.json")
JUDGE_MODEL_ID = os.getenv("FABLE_JUDGE_MODEL_ID", "base-qwen3-4b")
RESULTS_PATH = os.getenv("FABLE_RESULTS_PATH", "results/eval_summary.json")

# Judge backend: if set, the /evaluate endpoint uses a separate backend from
# the generation model. This allows e.g. MLX for generation + Gemini for judging.
# Falls back to the main BACKEND / OLLAMA_BASE_URL if not set.
JUDGE_BACKEND = os.getenv("FABLE_JUDGE_BACKEND", "")
JUDGE_BASE_URL = os.getenv("FABLE_JUDGE_BASE_URL", "")
JUDGE_API_KEY = os.getenv("FABLE_JUDGE_API_KEY", "")
JUDGE_THINKING_LEVEL = os.getenv("FABLE_JUDGE_THINKING_LEVEL", "minimal").lower()
# Judges are instruction-tuned → use chat completions by default.
# Set FABLE_JUDGE_USE_COMPLETION=true only for base-LM judges.
JUDGE_USE_COMPLETION = os.getenv("FABLE_JUDGE_USE_COMPLETION", "false").lower() == "true"

GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.8"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.9"))
GEN_REPEAT_PENALTY = float(os.getenv("GEN_REPEAT_PENALTY", "1.3"))

LENGTH_NUM_PREDICT = {"short": 150, "medium": 350, "long": 600}
# Length hints live in prompt_en.py (imported by main.py). They are
# natural-language instructions for instruction-tuned models only;
# the from-scratch base LM skips them (length = num_predict).

import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("FABLE_MODEL", "fable-tuned")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("FABLE_TIMEOUT", "120"))
BASE_MODEL = os.getenv("FABLE_BASE_MODEL", "fable-base")
TUNED_MODEL = os.getenv("FABLE_TUNED_MODEL", "fable-tuned")

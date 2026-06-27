from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import ollama_client
from app.config import BASE_MODEL, TUNED_MODEL
from app.guardrail.input_filter import check_input
from app.guardrail.output_filter import check_output
from app.prompt import SYSTEM_PROMPT_GUARDED, SYSTEM_PROMPT_MINIMAL, build_instruction

app = FastAPI(title="Vietnamese Fable Generator")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_REGEN = 1


def resolve_model(choice: str) -> str:
    return BASE_MODEL if choice == "base" else TUNED_MODEL


class GenerateRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    moral: str = Field(min_length=1, max_length=200)
    age_range: str = Field(min_length=1, max_length=50)
    guardrail_enabled: bool = True
    model_choice: Literal["base", "tuned"] = "tuned"


class GenerateResponse(BaseModel):
    status: str
    story: str | None = None
    reason: str | None = None


def generate_fn():
    """Dependency: trả về hàm sinh. Override được trong test."""
    return ollama_client.generate


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, gen=Depends(generate_fn)) -> GenerateResponse:
    instruction = build_instruction(req.topic, req.moral, req.age_range)
    model = resolve_model(req.model_choice)

    if not req.guardrail_enabled:
        try:
            story = gen(prompt=instruction, system=SYSTEM_PROMPT_MINIMAL, model=model)
        except ollama_client.OllamaError as exc:
            return GenerateResponse(status="error", reason=str(exc))
        return GenerateResponse(status="success", story=story)

    # Lớp 1: lọc đầu vào
    decision = check_input(req.topic, req.moral, req.age_range)
    if not decision.allowed:
        return GenerateResponse(status="refused", reason=decision.reason)

    # Lớp 2 + 3: system prompt ràng buộc + model đã học từ chối
    for _ in range(MAX_REGEN + 1):
        try:
            story = gen(prompt=instruction, system=SYSTEM_PROMPT_GUARDED, model=model)
        except ollama_client.OllamaError as exc:
            return GenerateResponse(status="error", reason=str(exc))
        # Lớp 4: lọc đầu ra
        out = check_output(story)
        if out.ok:
            return GenerateResponse(status="success", story=story)

    return GenerateResponse(status="refused", reason=out.reason)


# Phục vụ frontend tĩnh (mount sau /generate để không che API)
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

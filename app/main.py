import json
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import ollama_client
from app.config import BASE_MODEL, LENGTH_HINT, LENGTH_NUM_PREDICT, TUNED_MODEL
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
    length: Literal["short", "medium", "long"] = "medium"


class GenerateResponse(BaseModel):
    status: str
    story: str | None = None
    reason: str | None = None


def generate_fn():
    """Dependency: trả về hàm sinh. Override được trong test."""
    return ollama_client.generate


def stream_fn():
    """Dependency: trả về hàm sinh stream. Override được trong test."""
    return ollama_client.generate_stream


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, gen=Depends(generate_fn)) -> GenerateResponse:
    hint = LENGTH_HINT[req.length]
    num_predict = LENGTH_NUM_PREDICT[req.length]
    instruction = build_instruction(req.topic, req.moral, req.age_range, hint)
    model = resolve_model(req.model_choice)

    if not req.guardrail_enabled:
        try:
            story = gen(prompt=instruction, system=SYSTEM_PROMPT_MINIMAL, model=model,
                        num_predict=num_predict)
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
            story = gen(prompt=instruction, system=SYSTEM_PROMPT_GUARDED, model=model,
                        num_predict=num_predict)
        except ollama_client.OllamaError as exc:
            return GenerateResponse(status="error", reason=str(exc))
        # Lớp 4: lọc đầu ra
        out = check_output(story)
        if out.ok:
            return GenerateResponse(status="success", story=story)

    return GenerateResponse(status="refused", reason=out.reason)


@app.post("/generate/stream")
def generate_stream_endpoint(req: GenerateRequest,
                             gen=Depends(generate_fn),
                             gen_stream=Depends(stream_fn)) -> StreamingResponse:
    hint = LENGTH_HINT[req.length]
    num_predict = LENGTH_NUM_PREDICT[req.length]
    instruction = build_instruction(req.topic, req.moral, req.age_range, hint)
    model = resolve_model(req.model_choice)

    def events():
        # GUARDRAIL TẮT: stream token trực tiếp
        if not req.guardrail_enabled:
            yield _sse({"type": "step", "stage": "generating", "status": "running",
                        "detail": f"Sinh truyện bằng {model} (guardrail TẮT)"})
            buf = []
            try:
                for piece in gen_stream(prompt=instruction, system=SYSTEM_PROMPT_MINIMAL,
                                        model=model, num_predict=num_predict):
                    buf.append(piece)
                    yield _sse({"type": "token", "text": piece})
            except ollama_client.OllamaError as exc:
                yield _sse({"type": "error", "reason": str(exc)})
                return
            yield _sse({"type": "done", "status": "success", "story": "".join(buf)})
            return

        # GUARDRAIL BẬT: log từng lớp, KHÔNG stream token (Lớp 4 cần toàn bộ text)
        yield _sse({"type": "step", "stage": "input_check", "status": "running",
                    "detail": "Lớp 1: kiểm tra yêu cầu đầu vào"})
        decision = check_input(req.topic, req.moral, req.age_range)
        if not decision.allowed:
            yield _sse({"type": "step", "stage": "input_check", "status": "blocked",
                        "detail": decision.reason})
            yield _sse({"type": "done", "status": "refused", "reason": decision.reason})
            return
        yield _sse({"type": "step", "stage": "input_check", "status": "ok",
                    "detail": "Yêu cầu hợp lệ"})

        out_reason = "Truyện sinh ra chứa nội dung không phù hợp."
        for attempt in range(MAX_REGEN + 1):
            yield _sse({"type": "step", "stage": "generating", "status": "running",
                        "detail": f"Lớp 2-3: sinh truyện bằng {model} (lần {attempt + 1})"})
            try:
                story = gen(prompt=instruction, system=SYSTEM_PROMPT_GUARDED,
                            model=model, num_predict=num_predict)
            except ollama_client.OllamaError as exc:
                yield _sse({"type": "error", "reason": str(exc)})
                return
            yield _sse({"type": "step", "stage": "output_check", "status": "running",
                        "detail": "Lớp 4: kiểm tra nội dung sinh ra"})
            out = check_output(story)
            if out.ok:
                yield _sse({"type": "step", "stage": "output_check", "status": "ok",
                            "detail": "Nội dung an toàn"})
                yield _sse({"type": "done", "status": "success", "story": story})
                return
            out_reason = out.reason
            yield _sse({"type": "step", "stage": "output_check", "status": "blocked",
                        "detail": out.reason})

        yield _sse({"type": "done", "status": "refused", "reason": out_reason})

    return StreamingResponse(events(), media_type="text/event-stream")


# Phục vụ frontend tĩnh (mount sau /generate để không che API)
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

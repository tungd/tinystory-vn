import json
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import ollama_client, judge
from app.models_registry import load_models, resolve_ollama
from app.config import JUDGE_MODEL_ID
from app.guardrail.input_filter import check_input_en
from app.guardrail.output_filter import check_output_en
from app.prompt_en import (
    SYSTEM_PROMPT_EN,
    SYSTEM_PROMPT_MINIMAL_EN,
    LENGTH_NUM_PREDICT,
    LENGTH_HINT_EN,
    build_fable_prompt,
)

app = FastAPI(title="English Fable Generator")
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
MAX_REGEN = 1


class GenReq(BaseModel):
    character: str = Field("", max_length=200)
    setting: str = Field("", max_length=200)
    challenge: str = Field("", max_length=300)
    outcome: str = Field("", max_length=300)
    teaching: str = Field("", max_length=200)
    length: Literal["short", "medium", "long"] = "medium"
    model_id: str = "base-qwen3-4b"
    guardrail_enabled: bool = True


class EvalReq(BaseModel):
    story: str
    prompt: str
    judge_model_id: str | None = None


def generate_fn():
    return ollama_client.generate


def stream_fn():
    return ollama_client.generate_stream


def judge_fn():
    return ollama_client.generate


def _sse(d: dict) -> str:
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"


@app.get("/models")
def models():
    return load_models()


@app.post("/generate/stream")
def generate_stream(req: GenReq, gen=Depends(generate_fn), gstream=Depends(stream_fn)):
    hint = LENGTH_HINT_EN[req.length]
    num_predict = LENGTH_NUM_PREDICT[req.length]
    prompt = build_fable_prompt(
        req.character, req.setting, req.challenge, req.outcome, req.teaching, hint
    )
    try:
        model = resolve_ollama(req.model_id)
    except KeyError:
        return StreamingResponse(
            iter([_sse({"type": "error", "reason": f"Unknown model_id: {req.model_id}"})]),
            media_type="text/event-stream",
        )

    def events():
        if not req.guardrail_enabled:
            yield _sse(
                {
                    "type": "step",
                    "stage": "generating",
                    "status": "running",
                    "detail": f"Generating with {model} (guardrail OFF)",
                }
            )
            buf = []
            try:
                for piece in gstream(
                    prompt=prompt,
                    system=SYSTEM_PROMPT_MINIMAL_EN,
                    model=model,
                    num_predict=num_predict,
                ):
                    buf.append(piece)
                    yield _sse({"type": "token", "text": piece})
            except ollama_client.OllamaError as e:
                yield _sse({"type": "error", "reason": str(e)})
                return
            yield _sse({"type": "done", "status": "success", "story": "".join(buf)})
            return

        # guardrail ON
        yield _sse(
            {
                "type": "step",
                "stage": "input_check",
                "status": "running",
                "detail": "Layer 1: checking request",
            }
        )
        d = check_input_en(
            req.character, req.setting, req.challenge, req.outcome, req.teaching
        )
        if not d.allowed:
            yield _sse(
                {"type": "step", "stage": "input_check", "status": "blocked", "detail": d.reason}
            )
            yield _sse({"type": "done", "status": "refused", "reason": d.reason})
            return
        yield _sse(
            {"type": "step", "stage": "input_check", "status": "ok", "detail": "Request OK"}
        )
        reason = "The generated story was not appropriate."
        for attempt in range(MAX_REGEN + 1):
            yield _sse(
                {
                    "type": "step",
                    "stage": "generating",
                    "status": "running",
                    "detail": f"Layer 2-3: generating with {model} (try {attempt + 1})",
                }
            )
            try:
                story = gen(
                    prompt=prompt,
                    system=SYSTEM_PROMPT_EN,
                    model=model,
                    num_predict=num_predict,
                )
            except ollama_client.OllamaError as e:
                yield _sse({"type": "error", "reason": str(e)})
                return
            yield _sse(
                {
                    "type": "step",
                    "stage": "output_check",
                    "status": "running",
                    "detail": "Layer 4: checking output",
                }
            )
            out = check_output_en(story)
            if out.ok:
                yield _sse(
                    {
                        "type": "step",
                        "stage": "output_check",
                        "status": "ok",
                        "detail": "Content safe",
                    }
                )
                yield _sse({"type": "done", "status": "success", "story": story})
                return
            reason = out.reason
            yield _sse(
                {
                    "type": "step",
                    "stage": "output_check",
                    "status": "blocked",
                    "detail": out.reason,
                }
            )
        yield _sse({"type": "done", "status": "refused", "reason": reason})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/evaluate")
def evaluate(req: EvalReq, jf=Depends(judge_fn)):
    jid = req.judge_model_id or JUDGE_MODEL_ID
    try:
        model = resolve_ollama(jid)
    except KeyError:
        return JSONResponse({"error": f"Unknown judge model: {jid}"}, status_code=400)
    return judge.evaluate(req.story, req.prompt, model=model, gen=jf)


# Serve web build if it exists (Phase B)
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

import json
import time
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import ollama_client, judge
from app.models_registry import load_models, resolve_ollama
from app.config import (
    JUDGE_MODEL_ID,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    GEN_REPEAT_PENALTY,
    RESULTS_PATH,
)
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
    seed: int | None = None


class EvalReq(BaseModel):
    story: str
    prompt: str
    judge_model_id: str | None = None


def generate_fn():
    return ollama_client.generate


def meta_fn():
    return ollama_client.generate_meta


def stream_fn():
    return ollama_client.generate_stream


def judge_fn():
    return ollama_client.generate


def _sse(d: dict) -> str:
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"


def _resolve_model_info(model_id: str) -> tuple[str, str]:
    """Return (model_name, kind) from registry for a given model_id."""
    for m in load_models():
        if m["id"] == model_id:
            return m.get("name", model_id), m.get("kind", "base")
    return model_id, "base"


@app.get("/models")
def models():
    return load_models()


@app.post("/generate/stream")
def generate_stream(
    req: GenReq,
    gen=Depends(generate_fn),
    gstream=Depends(stream_fn),
    gmeta=Depends(meta_fn),
):
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

    model_name, kind = _resolve_model_info(req.model_id)

    seed_str = str(req.seed) if req.seed is not None else "random"
    params_detail = (
        f"Model: {model_name} ({kind}) via Ollama '{model}' | "
        f"temp {GEN_TEMPERATURE}, top_p {GEN_TOP_P}, "
        f"repeat_penalty {GEN_REPEAT_PENALTY}, seed {seed_str}"
    )

    def events():
        # Preparation steps (both paths): prompt + model config, for observability.
        yield _sse(
            {
                "type": "step",
                "stage": "prepare",
                "status": "ok",
                "detail": (
                    f"Prompt built: {req.length} fable "
                    f"({len(prompt)} chars), max {num_predict} tokens"
                ),
            }
        )
        yield _sse(
            {"type": "step", "stage": "model", "status": "ok", "detail": params_detail}
        )

        if not req.guardrail_enabled:
            yield _sse(
                {
                    "type": "step",
                    "stage": "generating",
                    "status": "running",
                    "detail": f"Streaming from {model_name} (guardrail OFF)",
                }
            )
            buf = []
            t0 = time.perf_counter()
            try:
                for piece in gstream(
                    prompt=prompt,
                    system=SYSTEM_PROMPT_MINIMAL_EN,
                    model=model,
                    num_predict=num_predict,
                    seed=req.seed,
                    temperature=GEN_TEMPERATURE,
                    top_p=GEN_TOP_P,
                    repeat_penalty=GEN_REPEAT_PENALTY,
                ):
                    buf.append(piece)
                    yield _sse({"type": "token", "text": piece})
            except ollama_client.OllamaError as e:
                yield _sse({"type": "error", "reason": str(e)})
                return
            latency_ms = int((time.perf_counter() - t0) * 1000)
            story_text = "".join(buf)
            output_tokens = len(story_text.split())
            tokens_per_sec = (
                output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0
            )
            yield _sse(
                {
                    "type": "step",
                    "stage": "generating",
                    "status": "ok",
                    "detail": (
                        f"Generated {output_tokens} tokens in {latency_ms} ms "
                        f"({round(tokens_per_sec, 1)} tok/s)"
                    ),
                }
            )
            meta = {
                "model_id": req.model_id,
                "model_name": model_name,
                "kind": kind,
                "temperature": GEN_TEMPERATURE,
                "top_p": GEN_TOP_P,
                "repetition_penalty": GEN_REPEAT_PENALTY,
                "num_predict": num_predict,
                "seed": req.seed,
                "prompt_sent": prompt,
                "input_tokens": 0,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "tokens_per_sec": round(tokens_per_sec, 2),
            }
            yield _sse({"type": "done", "status": "success", "story": story_text, "meta": meta})
            return

        # guardrail ON
        yield _sse(
            {
                "type": "step",
                "stage": "input_check",
                "status": "running",
                "detail": "Layer 1: scanning request for unsafe or out-of-scope content",
            }
        )
        d = check_input_en(
            req.character, req.setting, req.challenge, req.outcome, req.teaching
        )
        if not d.allowed:
            yield _sse(
                {
                    "type": "step",
                    "stage": "input_check",
                    "status": "blocked",
                    "detail": f"Layer 1 BLOCKED [{d.category}]: {d.reason}",
                }
            )
            yield _sse({"type": "done", "status": "refused", "reason": d.reason})
            return
        yield _sse(
            {
                "type": "step",
                "stage": "input_check",
                "status": "ok",
                "detail": "Layer 1 passed: request is in scope",
            }
        )
        reason = "The generated story was not appropriate."
        for attempt in range(MAX_REGEN + 1):
            yield _sse(
                {
                    "type": "step",
                    "stage": "generating",
                    "status": "running",
                    "detail": f"Layer 2-3: generating with {model_name} (attempt {attempt + 1})",
                }
            )
            try:
                result = gmeta(
                    prompt=prompt,
                    system=SYSTEM_PROMPT_EN,
                    model=model,
                    num_predict=num_predict,
                    seed=req.seed,
                    temperature=GEN_TEMPERATURE,
                    top_p=GEN_TOP_P,
                    repeat_penalty=GEN_REPEAT_PENALTY,
                )
            except ollama_client.OllamaError as e:
                yield _sse({"type": "error", "reason": str(e)})
                return
            story = result["text"]
            input_tokens = result.get("input_tokens", 0)
            output_tokens = result.get("output_tokens", 0)
            latency_ms = result.get("latency_ms", 0)
            tokens_per_sec = (
                output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0
            )
            yield _sse(
                {
                    "type": "step",
                    "stage": "generating",
                    "status": "ok",
                    "detail": (
                        f"Generated {output_tokens} tokens in {latency_ms} ms "
                        f"({round(tokens_per_sec, 1)} tok/s, {input_tokens} prompt tokens)"
                    ),
                }
            )
            yield _sse(
                {
                    "type": "step",
                    "stage": "output_check",
                    "status": "running",
                    "detail": "Layer 4: scanning generated story for unsafe words",
                }
            )
            out = check_output_en(story)
            if out.ok:
                yield _sse(
                    {
                        "type": "step",
                        "stage": "output_check",
                        "status": "ok",
                        "detail": "Layer 4 passed: content is safe",
                    }
                )
                meta = {
                    "model_id": req.model_id,
                    "model_name": model_name,
                    "kind": kind,
                    "temperature": GEN_TEMPERATURE,
                    "top_p": GEN_TOP_P,
                    "repetition_penalty": GEN_REPEAT_PENALTY,
                    "num_predict": num_predict,
                    "seed": req.seed,
                    "prompt_sent": prompt,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": latency_ms,
                    "tokens_per_sec": round(tokens_per_sec, 2),
                }
                yield _sse({"type": "done", "status": "success", "story": story, "meta": meta})
                return
            reason = out.reason
            attempts_left = MAX_REGEN - attempt
            regen_note = (
                f" Regenerating (attempts left: {attempts_left})..."
                if attempts_left > 0
                else " No attempts left; refusing."
            )
            yield _sse(
                {
                    "type": "step",
                    "stage": "output_check",
                    "status": "blocked",
                    "detail": f"Layer 4 BLOCKED: {out.reason}{regen_note}",
                }
            )
        meta = {
            "model_id": req.model_id,
            "model_name": model_name,
            "kind": kind,
            "temperature": GEN_TEMPERATURE,
            "top_p": GEN_TOP_P,
            "repetition_penalty": GEN_REPEAT_PENALTY,
            "num_predict": num_predict,
            "seed": req.seed,
            "prompt_sent": prompt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "tokens_per_sec": round(tokens_per_sec, 2),
        }
        yield _sse({"type": "done", "status": "refused", "reason": reason, "meta": meta})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/evaluate")
def evaluate(req: EvalReq, jf=Depends(judge_fn)):
    jid = req.judge_model_id or JUDGE_MODEL_ID
    try:
        model = resolve_ollama(jid)
    except KeyError:
        return JSONResponse({"error": f"Unknown judge model: {jid}"}, status_code=400)
    return judge.evaluate(req.story, req.prompt, model=model, gen=jf)


@app.get("/results")
def results():
    """Read batch eval summary from RESULTS_PATH.

    Returns:
      - {"available": true, "data": <json>} if file exists and is valid JSON
      - {"available": false, "data": null} if file absent or invalid JSON (HTTP 200)
    """
    import os
    results_path = os.getenv("FABLE_RESULTS_PATH", "results/eval_summary.json")
    if not Path(results_path).exists():
        return JSONResponse({"available": False, "data": None}, status_code=200)
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
        return JSONResponse({"available": True, "data": data}, status_code=200)
    except (json.JSONDecodeError, IOError):
        return JSONResponse({"available": False, "data": None}, status_code=200)


# Serve web build if it exists (Phase B)
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

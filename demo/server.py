"""Standalone Demo Studio HTTP app.

Run from the repository root with::

    .venv/bin/python -m demo.server
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from demo.judge import judge
from demo.runtime import runtime


STATIC_DIR = Path(__file__).resolve().parent / "static"
PRESETS_PATH = Path(__file__).resolve().parents[1] / "results/global_judge/global_prompts_v1.jsonl"


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    runtime.close()


app = FastAPI(title="TinyStory Demo Studio", lifespan=lifespan)


class StoryRequest(BaseModel):
    character: str = Field("A patient fox", max_length=200)
    setting: str = Field("a quiet forest after rain", max_length=200)
    challenge: str = Field("the river rises before the animals can cross", max_length=300)
    outcome: str = Field("the fox helps the smaller animals find a safe path", max_length=300)
    teaching: str = Field("Kindness makes every journey easier", max_length=200)
    length: Literal["short", "medium", "long"] = "medium"
    seed: int | None = Field(None, ge=0, le=2_147_483_647)


class JudgeRequest(BaseModel):
    story: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)


class CompareRequest(BaseModel):
    model_ids: list[str] = Field(..., min_length=2, max_length=5)
    prompt: StoryRequest


def load_presets() -> list[dict]:
    """Return the canonical 25 global-judge cases as selectable demo briefs."""

    rows: list[dict] = []
    with PRESETS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append({**row, "length": "medium"})
    return rows


def _failure(exc: Exception, status_code: int = 500) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=status_code)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app": "demo-studio", "judge": judge.status()}


@app.get("/api/models")
def models() -> list[dict]:
    return runtime.snapshots()


@app.get("/api/presets")
def presets() -> list[dict]:
    try:
        return load_presets()
    except Exception as exc:
        return _failure(exc, 500)


@app.post("/api/models/{model_id}/load")
def load_model(model_id: str):
    try:
        return runtime.load(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return _failure(exc, 503)


@app.post("/api/models/{model_id}/unload")
def unload_model(model_id: str):
    try:
        return runtime.unload(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return _failure(exc)


@app.post("/api/models/{model_id}/run")
def run_model(model_id: str, request: StoryRequest):
    try:
        payload = request.model_dump()
        return runtime.run(model_id, payload, seed=request.seed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        return _failure(exc, 409)
    except Exception as exc:
        return _failure(exc)


@app.post("/api/compare")
def compare_models(request: CompareRequest):
    if len(set(request.model_ids)) != len(request.model_ids):
        raise HTTPException(status_code=422, detail="model_ids must be unique")
    payload = request.prompt.model_dump()
    results: list[dict] = []
    for model_id in request.model_ids:
        try:
            results.append(runtime.run(model_id, payload, seed=request.prompt.seed))
        except Exception as exc:
            results.append({"model_id": model_id, "error": str(exc)})
    return {
        "prompt": payload,
        "results": results,
        "completed": sum("error" not in result for result in results),
    }


@app.get("/api/judge/status")
def judge_status() -> dict:
    return judge.status()


@app.post("/api/judge")
def evaluate(request: JudgeRequest):
    try:
        return judge.evaluate(request.story, request.prompt)
    except Exception as exc:
        return _failure(exc, 503)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="demo-ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("demo.server:app", host="127.0.0.1", port=8010, reload=False)

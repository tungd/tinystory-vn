"""Persistent MLX or Transformers/PEFT worker used by Demo Studio."""

from __future__ import annotations

import argparse
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Demo Studio model worker")
ENGINE: Any = None
BACKEND = ""


class GenerateRequest(BaseModel):
    prompt: str
    seed: int = 5410
    max_new_tokens: int = Field(440, ge=1, le=1200)
    temperature: float = Field(0.7, gt=0, le=2)
    top_p: float = Field(0.9, gt=0, le=1)
    repeat_penalty: float = Field(1.1, ge=1, le=2)


def _load_mlx(model_path: str) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(model_path)
    return {"model": model, "tokenizer": tokenizer, "mx": mx}


def _load_peft(model_path: str, adapter_path: str) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "torch": torch, "device": device}


def _generate_mlx(engine: dict[str, Any], request: GenerateRequest) -> dict[str, Any]:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

    engine["mx"].random.seed(request.seed)
    started = time.perf_counter()
    text = generate(
        engine["model"],
        engine["tokenizer"],
        prompt=request.prompt,
        max_tokens=request.max_new_tokens,
        sampler=make_sampler(temp=request.temperature, top_p=request.top_p),
        logits_processors=[make_repetition_penalty(request.repeat_penalty)],
        verbose=False,
    )
    return {
        "text": text,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _generate_peft(engine: dict[str, Any], request: GenerateRequest) -> dict[str, Any]:
    torch = engine["torch"]
    tokenizer = engine["tokenizer"]
    encoded = tokenizer(request.prompt, return_tensors="pt").to(engine["device"])
    torch.manual_seed(request.seed)
    started = time.perf_counter()
    with torch.inference_mode():
        output = engine["model"].generate(
            **encoded,
            do_sample=True,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repeat_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return {
        "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "input_tokens": int(encoded["input_ids"].shape[1]),
        "output_tokens": int(new_tokens.shape[0]),
    }


@app.get("/health")
def health() -> dict[str, str]:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="model is still loading")
    return {"status": "ok", "backend": BACKEND}


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="model is still loading")
    if BACKEND == "mlx":
        return _generate_mlx(ENGINE, request)
    if BACKEND == "transformers_peft":
        return _generate_peft(ENGINE, request)
    raise HTTPException(status_code=500, detail=f"unsupported worker backend: {BACKEND}")


def main() -> None:
    global BACKEND, ENGINE

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["mlx", "transformers_peft"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    BACKEND = args.backend
    if BACKEND == "mlx":
        ENGINE = _load_mlx(args.model)
    else:
        if not args.adapter:
            raise SystemExit("--adapter is required for transformers_peft")
        ENGINE = _load_peft(args.model, args.adapter)

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

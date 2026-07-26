#!/usr/bin/env python3
"""Reproducible evaluation of five independent directions with blinded Gemma judging."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.enhanced_generation import enhance_story  # noqa: E402
from app.judge import AXES, build_judge_prompt, parse_scores  # noqa: E402
from app.prompt_en import LENGTH_HINT_EN, SYSTEM_PROMPT_EN, build_fable_prompt  # noqa: E402

RESULTS = ROOT / "results" / "global_judge"
PROMPTS_PATH = RESULTS / "global_prompts_v1.jsonl"
GEN_DIR = RESULTS / "generations"
LOG_DIR = RESULTS / "logs"
JUDGE_PATH = RESULTS / "judge_scores.blinded.jsonl"
BLIND_MAP_PATH = RESULTS / "blind_map.private.json"
SUMMARY_PATH = RESULTS / "summary.json"
SUMMARY_MD_PATH = RESULTS / "summary.md"
RUN_MANIFEST_PATH = RESULTS / "run_manifest.json"

MODEL_DIR = ROOT / "models" / "global-bench"
CANDIDATES = {
    "e1": {
        "name": "V16 conditioned",
        "direction": "E2",
        "member_name": "Đào Đức Tùng",
        "student_code": "20252612M",
        "backend": "mlx",
        "path": ROOT / "runs" / "v16" / "artifacts" / "conditioning-mlx",
    },
    "e2": {
        "name": "SLM 60M",
        "direction": "E1",
        "member_name": "Lê Hải Triều",
        "student_code": "20252611M",
        "backend": "llama_completion",
        "path": MODEL_DIR / "slm-60m.gguf",
    },
    "e3": {
        "name": "Base + Repair",
        "direction": "E4",
        "member_name": "Nguyễn Thị Phương Liên",
        "student_code": "20252130M",
        "backend": "llama_repair",
        "path": MODEL_DIR / "llama-3.2-3b-instruct-q4.gguf",
    },
    "e4": {
        "name": "SmolLM2 135M + LoRA C",
        "direction": "E3",
        "member_name": "Nguyễn Công Thanh",
        "student_code": "20252610M",
        "backend": "transformers_peft",
        "path": MODEL_DIR / "smollm2-135m",
        "adapter": MODEL_DIR / "smollm2-135m-c-adapter",
    },
    "e5": {
        "name": "Llama 3.2 3B fable Q4",
        "direction": "E5",
        "member_name": "Nguyễn Đình Lê Hoàng",
        "student_code": "20252737M",
        "backend": "llama_chat",
        "path": MODEL_DIR / "llama3-fable-1000-q4.gguf",
    },
}

GENERATION = {
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "max_new_tokens": 400,
    "seed_formula": "5410 + prompt_index",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def directory_manifest(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): file_sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
        and ".cache" not in file.parts
        and file.name not in {".gitattributes", "README.md"}
    }


def candidate_manifest(candidate: str) -> dict[str, Any]:
    spec = CANDIDATES[candidate]
    path = Path(spec["path"])
    manifest: dict[str, Any] = {
        "candidate_id": candidate,
        "name": spec["name"],
        "direction": spec["direction"],
        "member_name": spec["member_name"],
        "student_code": spec["student_code"],
        "backend": spec["backend"],
        "path": str(path.relative_to(ROOT)),
    }
    if path.is_file():
        manifest["sha256"] = file_sha256(path)
        manifest["bytes"] = path.stat().st_size
    else:
        manifest["files"] = directory_manifest(path)
    if "adapter" in spec:
        adapter = Path(spec["adapter"])
        manifest["adapter_path"] = str(adapter.relative_to(ROOT))
        manifest["adapter_files"] = directory_manifest(adapter)
    return manifest


def prompt_text(row: dict[str, Any]) -> str:
    return build_fable_prompt(
        character=row["character"],
        setting=row["setting"],
        challenge=row["challenge"],
        outcome=row["outcome"],
        teaching=row["teaching"],
        length_hint=LENGTH_HINT_EN["medium"],
    )


def model_prompt(candidate: str, row: dict[str, Any]) -> str:
    if candidate == "e1":
        return (
            f"<char> {row['character'].strip()} </char>\n"
            f"<moral> {row['teaching'].strip()} </moral>\n"
            "<story>\n"
        )
    return prompt_text(row)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def clean_story(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?s)<\|.*?\|>", "", text)
    text = re.sub(r"(?s)</?(?:story|s)>", "", text)
    return text.strip()


def request_json(url: str, payload: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


@contextmanager
def llama_server(model: Path, port: int, log_path: Path) -> Iterator[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    command = [
        "/opt/local/bin/llama-server",
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "2048",
        "--n-gpu-layers",
        "99",
        "--no-webui",
        "--jinja",
    ]
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 180
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {process.returncode}; see {log_path}")
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                    health = json.loads(response.read())
                if health.get("status") == "ok":
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError(f"llama-server did not become healthy; see {log_path}")
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log_handle.close()


def llama_completion(base_url: str, prompt: str, seed: int, max_tokens: int = 400) -> dict[str, Any]:
    started = time.perf_counter()
    result = request_json(
        f"{base_url}/completion",
        {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": GENERATION["temperature"],
            "top_p": GENERATION["top_p"],
            "repeat_penalty": GENERATION["repeat_penalty"],
            "seed": seed,
            "stream": False,
        },
    )
    return {
        "text": result.get("content", ""),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "input_tokens": result.get("tokens_evaluated", 0),
        "output_tokens": result.get("tokens_predicted", 0),
        "stop": result.get("stop", False),
        "stopped_eos": result.get("stopped_eos", False),
        "stopped_limit": result.get("stopped_limit", False),
    }


def llama_chat(
    base_url: str,
    prompt: str,
    seed: int,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    max_tokens: int = 400,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = request_json(
        f"{base_url}/v1/chat/completions",
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_EN},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "seed": seed,
            "stream": False,
        },
    )
    usage = result.get("usage", {})
    choice = result["choices"][0]
    return {
        "text": choice["message"]["content"],
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "finish_reason": choice.get("finish_reason"),
    }


def generate_mlx(prompts: list[dict[str, Any]], pending: set[str]) -> Iterator[dict[str, Any]]:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

    spec = CANDIDATES["e1"]
    model, tokenizer = load(str(spec["path"]))
    sampler = make_sampler(temp=0.7, top_p=0.9)
    repetition_processor = make_repetition_penalty(GENERATION["repeat_penalty"])
    for index, row in enumerate(prompts):
        if row["prompt_id"] not in pending:
            continue
        seed = 5410 + index
        mx.random.seed(seed)
        started = time.perf_counter()
        text = generate(
            model,
            tokenizer,
            prompt=model_prompt("e1", row),
            max_tokens=GENERATION["max_new_tokens"],
            sampler=sampler,
            logits_processors=[repetition_processor],
            verbose=False,
        )
        yield generation_row("e1", row, seed, text, started, {})


def generate_peft(prompts: list[dict[str, Any]], pending: set[str]) -> Iterator[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = CANDIDATES["e4"]
    tokenizer = AutoTokenizer.from_pretrained(str(spec["path"]))
    model = AutoModelForCausalLM.from_pretrained(str(spec["path"]), dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(spec["adapter"]))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    for index, row in enumerate(prompts):
        if row["prompt_id"] not in pending:
            continue
        seed = 5410 + index
        torch.manual_seed(seed)
        encoded = tokenizer(model_prompt("e4", row), return_tensors="pt").to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=True,
                max_new_tokens=GENERATION["max_new_tokens"],
                temperature=GENERATION["temperature"],
                top_p=GENERATION["top_p"],
                repetition_penalty=GENERATION["repeat_penalty"],
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        yield generation_row(
            "e4",
            row,
            seed,
            text,
            started,
            {"input_tokens": int(encoded["input_ids"].shape[1]), "output_tokens": int(new_tokens.shape[0])},
        )
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def generation_row(
    candidate: str,
    prompt: dict[str, Any],
    seed: int,
    story: str,
    started: float,
    metadata: dict[str, Any],
    *,
    raw_story: str | None = None,
    actions: list[str] | None = None,
) -> dict[str, Any]:
    cleaned = clean_story(story)
    words = word_count(cleaned)
    result = {
        "candidate_id": candidate,
        "candidate_name": CANDIDATES[candidate]["name"],
        "prompt_id": prompt["prompt_id"],
        "status": "ok" if words else "error",
        "seed": seed,
        "prompt": prompt,
        "formatted_prompt": prompt_text(prompt),
        "model_prompt": model_prompt(candidate, prompt),
        "story": cleaned,
        "word_count": words,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "generation": GENERATION,
        "actions": actions or [],
    }
    if raw_story is not None:
        result["raw_story"] = raw_story
    result.update(metadata)
    return result


def generate_llama(candidate: str, prompts: list[dict[str, Any]], pending: set[str]) -> Iterator[dict[str, Any]]:
    spec = CANDIDATES[candidate]
    port = {"e2": 18082, "e3": 18083, "e5": 18085}[candidate]
    with llama_server(Path(spec["path"]), port, LOG_DIR / f"{candidate}.log") as base_url:
        for index, row in enumerate(prompts):
            if row["prompt_id"] not in pending:
                continue
            seed = 5410 + index
            started = time.perf_counter()
            if candidate == "e2":
                generated = llama_completion(base_url, model_prompt(candidate, row), seed)
                yield generation_row(candidate, row, seed, generated["text"], started, generated)
                continue
            generated = llama_chat(base_url, model_prompt(candidate, row), seed)
            if candidate == "e5":
                yield generation_row(candidate, row, seed, generated["text"], started, generated)
                continue

            def rewrite(**kwargs: Any) -> dict[str, Any]:
                return llama_chat(
                    base_url,
                    kwargs["prompt"],
                    kwargs["seed"],
                    temperature=kwargs["temperature"],
                    top_p=kwargs["top_p"],
                    repeat_penalty=kwargs["repeat_penalty"],
                    max_tokens=kwargs["num_predict"],
                )

            enhanced = enhance_story(
                clean_story(generated["text"]),
                row,
                rewrite_model="llama-3.2-3b-instruct-q4",
                generate_meta_fn=rewrite,
                seed=seed,
            )
            metadata = {
                **generated,
                "initial_validation": enhanced["initial_validation"],
                "final_validation": enhanced["final_validation"],
                "rewrite_meta": enhanced["rewrite_meta"],
            }
            yield generation_row(
                candidate,
                row,
                seed,
                enhanced["story"],
                started,
                metadata,
                raw_story=clean_story(generated["text"]),
                actions=enhanced["actions"],
            )


def command_generate(candidate: str) -> None:
    prompts = read_jsonl(PROMPTS_PATH)
    if len(prompts) != 25:
        raise ValueError(f"Expected 25 prompts, found {len(prompts)}")
    spec = CANDIDATES[candidate]
    if not Path(spec["path"]).exists():
        raise FileNotFoundError(spec["path"])
    output = GEN_DIR / f"{candidate}.jsonl"
    completed = {row["prompt_id"] for row in read_jsonl(output)}
    pending = {row["prompt_id"] for row in prompts} - completed
    print(f"{candidate}: {len(completed)} complete, {len(pending)} pending", flush=True)
    if not pending:
        return
    if spec["backend"] == "mlx":
        iterator = generate_mlx(prompts, pending)
    elif spec["backend"] == "transformers_peft":
        iterator = generate_peft(prompts, pending)
    else:
        iterator = generate_llama(candidate, prompts, pending)
    for row in iterator:
        append_jsonl(output, row)
        print(
            f"{candidate} {row['prompt_id']}: {row['status']}, "
            f"{row['word_count']} words, {row['latency_ms']} ms",
            flush=True,
        )


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


def blind_items() -> list[dict[str, Any]]:
    items = []
    for candidate in CANDIDATES:
        rows = read_jsonl(GEN_DIR / f"{candidate}.jsonl")
        latest = {row["prompt_id"]: row for row in rows}
        if len(latest) != 25:
            raise ValueError(f"{candidate}: expected 25 generation attempts, found {len(latest)}")
        for prompt_id, row in latest.items():
            items.append(
                {
                    "candidate_id": candidate,
                    "prompt_id": prompt_id,
                    "formatted_prompt": row["formatted_prompt"],
                    "story": row["story"],
                }
            )
    random.Random(20260726).shuffle(items)
    for index, item in enumerate(items, 1):
        item["blind_id"] = f"B{index:03d}"
    return items


def command_judge(workers: int) -> None:
    from google import genai
    from google.genai import types

    load_env(ROOT / ".env")
    api_key = os.environ.get("FABLE_JUDGE_API_KEY")
    if not api_key:
        raise RuntimeError("FABLE_JUDGE_API_KEY is not configured")
    model_id = os.environ.get("FABLE_JUDGE_MODEL_ID", "gemma-4-26b-a4b-it")
    client = genai.Client(api_key=api_key)
    schema = {
        "type": "object",
        "properties": {
            axis: {"type": "integer", "minimum": 1, "maximum": 10}
            for axis in AXES
        },
        "required": AXES,
    }
    config = types.GenerateContentConfig(
        system_instruction="You are a strict, fair evaluator. Output JSON only.",
        temperature=0,
        seed=20260726,
        max_output_tokens=512,
        response_mime_type="application/json",
        response_json_schema=schema,
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.MINIMAL,
            include_thoughts=False,
        ),
    )
    items = blind_items()
    BLIND_MAP_PATH.write_text(
        json.dumps(
            [{"blind_id": i["blind_id"], "candidate_id": i["candidate_id"], "prompt_id": i["prompt_id"]} for i in items],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    completed = {row["blind_id"] for row in read_jsonl(JUDGE_PATH) if row.get("status") == "ok"}
    print(f"judge: {len(completed)} complete, {len(items) - len(completed)} pending", flush=True)

    def evaluate_item(item: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "Judge this children's fable strictly on four 1-10 integer scales. "
            "Grammar covers grammatical correctness, coherence, and child-appropriate style. "
            "Creativity covers originality and vivid storytelling. Moral clarity covers whether "
            "the requested lesson is clearly earned by the plot. Prompt adherence covers all "
            "requested character, setting, challenge, outcome, teaching, and final Moral line. "
            "Return only a JSON object with integer keys grammar, creativity, moral_clarity, "
            "prompt_adherence. Do not explain.\n\n"
            f"REQUEST:\n{item['formatted_prompt']}\n\nSTORY:\n{item['story']}\n\nJSON:"
        )
        last_error: Exception | None = None
        for attempt in range(1, 9):
            try:
                started = time.perf_counter()
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=config,
                )
                raw = response.text or ""
                scores = parse_scores(raw)
                if any(not 1 <= scores[axis] <= 10 for axis in AXES):
                    raise ValueError(f"invalid scores: {scores}")
                return {
                    "blind_id": item["blind_id"],
                    "status": "ok",
                    "judge_model": model_id,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "scores": scores,
                    "raw": raw,
                }
            except Exception as exc:
                last_error = exc
                print(f"{item['blind_id']} attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
                if attempt < 8:
                    message = str(exc)
                    retry = re.search(r"retry in ([0-9.]+)(ms|s)", message, re.I)
                    if retry:
                        delay = float(retry.group(1))
                        if retry.group(2).lower() == "ms":
                            delay /= 1000
                        time.sleep(delay + 1)
                    else:
                        time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Gemma judge failed for {item['blind_id']}: {last_error}")

    pending = [item for item in items if item["blind_id"] not in completed]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_item, item): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            row = future.result()
            append_jsonl(JUDGE_PATH, row)
            print(
                f"{item['blind_id']}: {row['scores']['overall']:.2f}, {row['latency_ms']} ms",
                flush=True,
            )


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(values: list[float], seed: int, rounds: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(rounds)
    ]
    return [round(percentile(means, 0.025), 3), round(percentile(means, 0.975), 3)]


def command_summarize() -> None:
    blind_map = {
        row["blind_id"]: row for row in json.loads(BLIND_MAP_PATH.read_text(encoding="utf-8"))
    }
    judgments = {
        row["blind_id"]: row
        for row in read_jsonl(JUDGE_PATH)
        if row.get("status") == "ok"
    }
    if len(judgments) != 125:
        raise ValueError(f"Expected 125 complete judgments, found {len(judgments)}")
    by_candidate: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in CANDIDATES}
    for blind_id, judgment in judgments.items():
        mapping = blind_map[blind_id]
        by_candidate[mapping["candidate_id"]].append({**judgment, **mapping})
    summary: dict[str, Any] = {
        "protocol": {
            "prompt_set": str(PROMPTS_PATH.relative_to(ROOT)),
            "prompt_count": 25,
            "candidate_count": 5,
            "story_count": 125,
            "judge_count": 125,
            "judge_model": next(iter(judgments.values()))["judge_model"],
            "blind_shuffle_seed": 20260726,
            "generation": GENERATION,
            "bootstrap_rounds": 10_000,
        },
        "artifacts": {candidate: candidate_manifest(candidate) for candidate in CANDIDATES},
        "candidates": {},
    }
    for c_index, (candidate, rows) in enumerate(by_candidate.items()):
        generation_rows = {
            row["prompt_id"]: row
            for row in read_jsonl(GEN_DIR / f"{candidate}.jsonl")
        }
        axis_means = {
            axis: round(statistics.fmean(row["scores"][axis] for row in rows), 3)
            for axis in AXES
        }
        overall = [row["scores"]["overall"] for row in rows]
        latencies = [generation_rows[row["prompt_id"]]["latency_ms"] for row in rows]
        words = [generation_rows[row["prompt_id"]]["word_count"] for row in rows]
        summary["candidates"][candidate] = {
            "name": CANDIDATES[candidate]["name"],
            "direction": CANDIDATES[candidate]["direction"],
            "member_name": CANDIDATES[candidate]["member_name"],
            "student_code": CANDIDATES[candidate]["student_code"],
            "n": len(rows),
            "nonempty_generations": sum(
                generation_rows[row["prompt_id"]]["word_count"] > 0 for row in rows
            ),
            **axis_means,
            "overall_mean": round(statistics.fmean(overall), 3),
            "overall_median": round(statistics.median(overall), 3),
            "overall_95pct_bootstrap_ci": bootstrap_mean_ci(overall, 9100 + c_index),
            "generation_latency_mean_ms": round(statistics.fmean(latencies)),
            "generation_latency_median_ms": round(statistics.median(latencies)),
            "word_count_mean": round(statistics.fmean(words), 1),
            "word_count_median": round(statistics.median(words), 1),
            "action_count": sum(
                bool(generation_rows[row["prompt_id"]].get("actions")) for row in rows
            ),
            "rewrite_count": sum(
                "rewrite" in generation_rows[row["prompt_id"]].get("actions", [])
                for row in rows
            ),
            "moral_postprocess_count": sum(
                "moral_postprocess"
                in generation_rows[row["prompt_id"]].get("actions", [])
                for row in rows
            ),
        }
    ranking = sorted(
        summary["candidates"],
        key=lambda candidate: summary["candidates"][candidate]["overall_mean"],
        reverse=True,
    )
    summary["ranking"] = ranking
    prompt_scores = {
        candidate: {
            row["prompt_id"]: row["scores"]["overall"]
            for row in rows
        }
        for candidate, rows in by_candidate.items()
    }
    summary["pairwise"] = {}
    pair_index = 0
    for left_index, left in enumerate(ranking):
        for right in ranking[left_index + 1 :]:
            differences = [
                prompt_scores[left][prompt_id] - prompt_scores[right][prompt_id]
                for prompt_id in sorted(prompt_scores[left])
            ]
            summary["pairwise"][f"{left}_vs_{right}"] = {
                "left": left,
                "right": right,
                "mean_difference": round(statistics.fmean(differences), 3),
                "difference_95pct_bootstrap_ci": bootstrap_mean_ci(
                    differences, 12_000 + pair_index
                ),
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
            }
            pair_index += 1
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Global Gemma judge summary",
        "",
        f"- Judge: `{summary['protocol']['judge_model']}`",
        "- Prompts: 25; candidates: 5; stories judged: 125/125",
        "",
        "| Rank | Candidate | Grammar | Creativity | Moral | Adherence | Overall | 95% CI | Mean words | Mean latency |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, candidate in enumerate(ranking, 1):
        row = summary["candidates"][candidate]
        ci = row["overall_95pct_bootstrap_ci"]
        lines.append(
            f"| {rank} | {candidate}: {row['name']} | {row['grammar']:.2f} | "
            f"{row['creativity']:.2f} | {row['moral_clarity']:.2f} | "
            f"{row['prompt_adherence']:.2f} | {row['overall_mean']:.2f} | "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] | {row['word_count_mean']:.1f} | "
            f"{row['generation_latency_mean_ms'] / 1000:.2f}s |"
        )
    lines.extend(
        [
            "",
            "| Pair | Mean difference | 95% paired bootstrap CI | W/T/L |",
            "|---|---:|---:|---:|",
        ]
    )
    for pair in summary["pairwise"].values():
        ci = pair["difference_95pct_bootstrap_ci"]
        lines.append(
            f"| {pair['left']} − {pair['right']} | {pair['mean_difference']:.2f} | "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] | "
            f"{pair['wins']}/{pair['ties']}/{pair['losses']} |"
        )
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result_files = [
        PROMPTS_PATH,
        *(GEN_DIR / f"{candidate}.jsonl" for candidate in CANDIDATES),
        JUDGE_PATH,
        BLIND_MAP_PATH,
        SUMMARY_PATH,
        SUMMARY_MD_PATH,
        Path(__file__).resolve(),
    ]
    run_manifest = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version.split()[0],
        "files": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in result_files
        },
    }
    RUN_MANIFEST_PATH.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


def command_validate() -> None:
    prompts = read_jsonl(PROMPTS_PATH)
    ids = [row["prompt_id"] for row in prompts]
    if len(prompts) != 25 or len(set(ids)) != 25:
        raise ValueError("Prompt set must contain 25 unique prompt IDs")
    required = {"prompt_id", "character", "setting", "challenge", "outcome", "teaching"}
    for row in prompts:
        if set(row) != required or any(not str(row[key]).strip() for key in required):
            raise ValueError(f"Invalid prompt row: {row}")
    for candidate in CANDIDATES:
        spec = CANDIDATES[candidate]
        if not Path(spec["path"]).exists():
            raise FileNotFoundError(spec["path"])
        if "adapter" in spec and not Path(spec["adapter"]).exists():
            raise FileNotFoundError(spec["adapter"])
    if GEN_DIR.exists():
        for candidate in CANDIDATES:
            rows = read_jsonl(GEN_DIR / f"{candidate}.jsonl")
            prompt_ids = [row["prompt_id"] for row in rows]
            if len(rows) != 25 or len(set(prompt_ids)) != 25:
                raise ValueError(f"{candidate}: expected 25 unique generation rows")
    if JUDGE_PATH.exists():
        judgments = read_jsonl(JUDGE_PATH)
        blind_ids = [row["blind_id"] for row in judgments]
        if len(judgments) != 125 or len(set(blind_ids)) != 125:
            raise ValueError("Expected 125 unique judge rows")
        for row in judgments:
            if any(not 1 <= row["scores"][axis] <= 10 for axis in AXES):
                raise ValueError(f"Invalid judge scores in {row['blind_id']}")
    print("OK: prompts, artifacts, generations, and judge rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    judge_parser = sub.add_parser("judge")
    judge_parser.add_argument("--workers", type=int, default=6)
    sub.add_parser("summarize")
    sub.add_parser("validate")
    args = parser.parse_args()
    if args.command == "generate":
        command_generate(args.candidate)
    elif args.command == "judge":
        command_judge(args.workers)
    elif args.command == "summarize":
        command_summarize()
    else:
        command_validate()


if __name__ == "__main__":
    main()

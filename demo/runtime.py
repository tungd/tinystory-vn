"""Backend-aware model lifecycle for the standalone Demo Studio.

The main application intentionally remains Ollama-oriented.  This module is a
small operator-facing runtime for the five systems used by the global benchmark:
llama.cpp, MLX, Transformers/PEFT, and the base-plus-repair composition.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.enhanced_generation import enhance_story
from app.prompt_en import (
    LENGTH_HINT_EN,
    LENGTH_NUM_PREDICT,
    SYSTEM_PROMPT_EN,
    SYSTEM_PROMPT_MINIMAL_EN,
    build_smollm2_prompt,
    build_slm_prompt,
    build_v16_prompt,
    build_evaluation_brief,
    build_fable_prompt,
)
from app.textproc import trim_to_last_sentence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAMA_SERVER = Path("/opt/local/bin/llama-server")
MORAL_LINE_RE = re.compile(r"(?im)^\s*Moral\s*:\s*(.*)$")


def _load_dotenv(path: Path) -> None:
    """Load the project .env without replacing explicitly exported values."""

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class StudioSpec:
    id: str
    direction: str
    name: str
    member_name: str
    backend: str
    artifact: Path
    port: int
    description: str
    adapter: Path | None = None
    prompt_style: str = "standard"


STUDIO_SPECS: tuple[StudioSpec, ...] = (
    StudioSpec(
        id="e1-slm-60m",
        direction="E1",
        name="SLM 60M",
        member_name="Lê Hải Triều",
        backend="llama_completion",
        artifact=ROOT / "models/global-bench/slm-60m.gguf",
        port=18101,
        description="A compact scratch-trained model served through llama.cpp completion.",
        prompt_style="slm",
    ),
    StudioSpec(
        id="e2-v16-conditioned",
        direction="E2",
        name="V16 conditioned",
        member_name="Đào Đức Tùng",
        backend="mlx",
        artifact=ROOT / "runs/v16/artifacts/conditioning-mlx",
        port=18102,
        description="The selected V16 conditioning checkpoint loaded directly with MLX-LM.",
        prompt_style="v16",
    ),
    StudioSpec(
        id="e3-smollm2-lora-c",
        direction="E3",
        name="SmolLM2 135M + LoRA C",
        member_name="Nguyễn Công Thanh",
        backend="transformers_peft",
        artifact=ROOT / "models/global-bench/smollm2-135m",
        adapter=ROOT / "models/global-bench/smollm2-135m-c-adapter",
        port=18103,
        description="SmolLM2-135M with the C LoRA adapter, served by Transformers and PEFT.",
        prompt_style="smollm2",
    ),
    StudioSpec(
        id="e4-base-repair",
        direction="E4",
        name="Base + Repair",
        member_name="Nguyễn Thị Phương Liên",
        backend="llama_repair",
        artifact=ROOT / "models/global-bench/llama-3.2-3b-instruct-q4.gguf",
        port=18104,
        description="Llama 3.2 3B Q4 with validation, moral post-processing, and self-repair.",
    ),
    StudioSpec(
        id="e5-fable-q4",
        direction="E5",
        name="Llama 3.2 3B fable Q4",
        member_name="Nguyễn Đình Lê Hoàng",
        backend="llama_chat",
        artifact=ROOT / "models/global-bench/llama3-fable-1000-q4.gguf",
        port=18105,
        description="The fable-tuned Llama 3.2 3B Q4 chat model served through llama.cpp.",
    ),
)

SPEC_BY_ID = {spec.id: spec for spec in STUDIO_SPECS}


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _clean_story(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?s)<\|.*?\|>", "", text)
    text = re.sub(r"(?s)</?(?:story|s)>", "", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def _seed(seed: int | None) -> int:
    return seed if seed is not None else int(time.time() * 1000) % 2_147_483_647


@dataclass
class _Handle:
    process: subprocess.Popen[bytes]
    base_url: str
    log_handle: Any
    loaded_at: float


@dataclass
class _State:
    status: str = "unloaded"
    detail: str = "Not loaded"
    error: str | None = None
    loaded_at: float | None = None


class StudioRuntime:
    """Owns child processes and exposes truthful lifecycle state."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.log_dir = root / "demo" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.llama_server = Path(os.getenv("FABLE_LLAMA_SERVER", str(DEFAULT_LLAMA_SERVER)))
        self._states = {spec.id: _State() for spec in STUDIO_SPECS}
        self._handles: dict[str, _Handle] = {}
        self._locks = {spec.id: threading.RLock() for spec in STUDIO_SPECS}

    def spec(self, model_id: str) -> StudioSpec:
        try:
            return SPEC_BY_ID[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown studio model: {model_id}") from exc

    def _missing(self, spec: StudioSpec) -> list[str]:
        missing: list[str] = []
        if spec.artifact.is_file():
            pass
        elif spec.artifact.is_dir() and any(spec.artifact.iterdir()):
            pass
        else:
            missing.append(f"artifact missing: {_relative(spec.artifact)}")

        if spec.adapter is not None and not spec.adapter.exists():
            missing.append(f"adapter missing: {_relative(spec.adapter)}")

        if spec.backend.startswith("llama") and not self.llama_server.exists():
            missing.append(f"llama-server missing: {self.llama_server}")
        if spec.backend == "mlx":
            for module in ("mlx", "mlx_lm"):
                if importlib.util.find_spec(module) is None:
                    missing.append(f"Python dependency missing: {module}")
        if spec.backend == "transformers_peft":
            for module in ("torch", "transformers", "peft"):
                if importlib.util.find_spec(module) is None:
                    missing.append(f"Python dependency missing: {module}")
        return missing

    def _alive(self, model_id: str) -> bool:
        handle = self._handles.get(model_id)
        if handle is None:
            return False
        if handle.process.poll() is None:
            return True
        self._states[model_id].status = "error"
        self._states[model_id].error = (
            f"Runtime exited with code {handle.process.returncode}; see "
            f"{_relative(self.log_dir / f'{model_id}.log')}"
        )
        self._states[model_id].detail = "Process exited"
        return False

    def snapshot(self, model_id: str) -> dict[str, Any]:
        spec = self.spec(model_id)
        state = self._states[model_id]
        loaded = self._alive(model_id)
        missing = self._missing(spec)
        if not loaded and state.status == "loaded":
            state.status = "error"
            state.detail = "Runtime is no longer alive"
        return {
            "id": spec.id,
            "direction": spec.direction,
            "name": spec.name,
            "member_name": spec.member_name,
            "backend": spec.backend,
            "backend_label": {
                "llama_completion": "llama.cpp · completion",
                "llama_chat": "llama.cpp · chat",
                "llama_repair": "llama.cpp · base + repair",
                "mlx": "MLX-LM",
                "transformers_peft": "Transformers · PEFT",
            }.get(spec.backend, spec.backend),
            "artifact": _relative(spec.artifact),
            "adapter": _relative(spec.adapter) if spec.adapter else None,
            "description": spec.description,
            "available": not missing,
            "availability_detail": "Available" if not missing else "; ".join(missing),
            "status": "loaded" if loaded else state.status,
            "loaded": loaded,
            "pid": self._handles[model_id].process.pid if loaded else None,
            "loaded_at": state.loaded_at,
            "detail": state.detail,
            "error": state.error,
            "port": spec.port,
        }

    def snapshots(self) -> list[dict[str, Any]]:
        return [self.snapshot(spec.id) for spec in STUDIO_SPECS]

    def _start_process(self, spec: StudioSpec, command: list[str]) -> _Handle:
        log_path = self.log_dir / f"{spec.id}.log"
        log_handle = log_path.open("ab")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(self.root), env.get("PYTHONPATH", "")) if part
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self.root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            log_handle.close()
            raise
        return _Handle(
            process=process,
            base_url=f"http://127.0.0.1:{spec.port}",
            log_handle=log_handle,
            loaded_at=time.time(),
        )

    def _wait_health(self, handle: _Handle, timeout: float = 180.0) -> None:
        deadline = time.monotonic() + timeout
        last_error = "health endpoint did not respond"
        with httpx.Client(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if handle.process.poll() is not None:
                    raise RuntimeError(
                        f"runtime exited with code {handle.process.returncode}; "
                        "inspect the model log"
                    )
                try:
                    response = client.get(f"{handle.base_url}/health")
                    if response.status_code == 200:
                        return
                    last_error = f"health returned HTTP {response.status_code}"
                except (httpx.HTTPError, OSError) as exc:
                    last_error = str(exc)
                time.sleep(0.5)
        raise TimeoutError(last_error)

    def _start(self, spec: StudioSpec) -> _Handle:
        if spec.backend.startswith("llama"):
            command = [
                str(self.llama_server),
                "--model",
                str(spec.artifact),
                "--host",
                "127.0.0.1",
                "--port",
                str(spec.port),
                "--ctx-size",
                "2048",
                "--n-gpu-layers",
                "99",
                "--no-webui",
                "--jinja",
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "demo.worker",
                "--backend",
                spec.backend,
                "--model",
                str(spec.artifact),
                "--port",
                str(spec.port),
            ]
            if spec.adapter:
                command.extend(["--adapter", str(spec.adapter)])
        handle = self._start_process(spec, command)
        try:
            self._wait_health(handle)
        except Exception:
            self._stop_handle(handle)
            raise
        return handle

    @staticmethod
    def _stop_handle(handle: _Handle) -> None:
        if handle.process.poll() is None:
            handle.process.terminate()
            try:
                handle.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                handle.process.kill()
                handle.process.wait(timeout=10)
        handle.log_handle.close()

    def load(self, model_id: str) -> dict[str, Any]:
        spec = self.spec(model_id)
        with self._locks[model_id]:
            if self._alive(model_id):
                return self.snapshot(model_id)
            missing = self._missing(spec)
            if missing:
                self._states[model_id] = _State(
                    status="unavailable",
                    detail="; ".join(missing),
                    error="; ".join(missing),
                )
                raise RuntimeError("; ".join(missing))
            self._states[model_id] = _State(status="loading", detail="Starting runtime")
            try:
                handle = self._start(spec)
            except Exception as exc:
                self._states[model_id] = _State(
                    status="error", detail="Could not start runtime", error=str(exc)
                )
                raise
            self._handles[model_id] = handle
            self._states[model_id] = _State(
                status="loaded", detail="Runtime is warm", loaded_at=handle.loaded_at
            )
            return self.snapshot(model_id)

    def unload(self, model_id: str) -> dict[str, Any]:
        self.spec(model_id)
        with self._locks[model_id]:
            handle = self._handles.pop(model_id, None)
            if handle is not None:
                self._stop_handle(handle)
            self._states[model_id] = _State(status="unloaded", detail="Not loaded")
            return self.snapshot(model_id)

    def _require_handle(self, model_id: str) -> tuple[StudioSpec, _Handle]:
        spec = self.spec(model_id)
        if not self._alive(model_id):
            raise RuntimeError(f"{spec.direction} is not loaded")
        return spec, self._handles[model_id]

    @staticmethod
    def _post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=float(os.getenv("FABLE_TIMEOUT", "300"))) as client:
            response = client.post(f"{base_url}{path}", json=payload)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _model_prompt(spec: StudioSpec, row: dict[str, str]) -> str:
        if spec.prompt_style == "slm":
            return build_slm_prompt(
                row["character"],
                row["setting"],
                row["challenge"],
                row["outcome"],
                row["teaching"],
                LENGTH_HINT_EN[row["length"]],
            )
        if spec.prompt_style == "v16":
            return build_v16_prompt(row["character"], row["teaching"])
        if spec.prompt_style == "smollm2":
            return build_smollm2_prompt(
                row["character"],
                row["setting"],
                row["challenge"],
                row["outcome"],
                row["teaching"],
            )
        return build_fable_prompt(
            row["character"],
            row["setting"],
            row["challenge"],
            row["outcome"],
            row["teaching"],
            LENGTH_HINT_EN[row["length"]],
        )

    @staticmethod
    def _judge_prompt(spec: StudioSpec, row: dict[str, str]) -> str:
        return build_evaluation_brief(
            row["character"],
            row["setting"],
            row["challenge"],
            row["outcome"],
            row["teaching"],
            LENGTH_HINT_EN[row["length"]],
        )

    @staticmethod
    def _story_format(story: str, teaching: str) -> dict[str, bool]:
        cleaned = story.strip()
        matches = list(MORAL_LINE_RE.finditer(cleaned))
        if not matches:
            return {
                "has_moral_line": False,
                "moral_is_final": False,
                "moral_is_exact": False,
            }
        last = matches[-1]

        def normalize(value: str) -> str:
            return re.sub(r"[.!?]+\s*$", "", value.strip()).casefold()

        return {
            "has_moral_line": True,
            "moral_is_final": not cleaned[last.end() :].strip(),
            "moral_is_exact": normalize(last.group(1)) == normalize(teaching),
        }

    def _llama_completion(
        self, handle: _Handle, prompt: str, seed: int, num_predict: int
    ) -> dict[str, Any]:
        result = self._post_json(
            handle.base_url,
            "/completion",
            {
                "prompt": prompt,
                "n_predict": num_predict,
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "seed": seed,
                "stream": False,
            },
        )
        return {
            "text": result.get("content", ""),
            "input_tokens": result.get("tokens_evaluated", 0),
            "output_tokens": result.get("tokens_predicted", 0),
            "finish_reason": "length" if result.get("stopped_limit") else None,
        }

    def _llama_chat(
        self,
        handle: _Handle,
        prompt: str,
        seed: int,
        num_predict: int,
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
    ) -> dict[str, Any]:
        result = self._post_json(
            handle.base_url,
            "/v1/chat/completions",
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_EN},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": num_predict,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "seed": seed,
                "stream": False,
            },
        )
        choice = result["choices"][0]
        usage = result.get("usage", {})
        return {
            "text": choice.get("message", {}).get("content", ""),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "finish_reason": choice.get("finish_reason"),
        }

    def _worker_generate(
        self, handle: _Handle, prompt: str, seed: int, num_predict: int
    ) -> dict[str, Any]:
        return self._post_json(
            handle.base_url,
            "/generate",
            {
                "prompt": prompt,
                "seed": seed,
                "max_new_tokens": num_predict,
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        )

    def run(self, model_id: str, row: dict[str, str], seed: int | None = None) -> dict[str, Any]:
        with self._locks[model_id]:
            spec, handle = self._require_handle(model_id)
            started = time.perf_counter()
            actual_seed = _seed(seed if seed is not None else row.get("seed"))
            num_predict = LENGTH_NUM_PREDICT[row["length"]]
            prompt = self._model_prompt(spec, row)
            judge_prompt = self._judge_prompt(spec, row)

            if spec.backend == "llama_completion":
                generated = self._llama_completion(handle, prompt, actual_seed, num_predict)
            elif spec.backend in {"llama_chat", "llama_repair"}:
                generated = self._llama_chat(handle, prompt, actual_seed, num_predict)
            else:
                generated = self._worker_generate(handle, prompt, actual_seed, num_predict)

            raw_story = _clean_story(str(generated.get("text", "")))
            if generated.get("finish_reason") == "length":
                raw_story = trim_to_last_sentence(raw_story)
            story = raw_story
            actions: list[str] = []
            enhancement: dict[str, Any] | None = None

            if spec.backend == "llama_repair":
                def rewrite(**kwargs: Any) -> dict[str, Any]:
                    return self._llama_chat(
                        handle,
                        kwargs["prompt"],
                        kwargs["seed"],
                        kwargs["num_predict"],
                        temperature=kwargs["temperature"],
                        top_p=kwargs["top_p"],
                        repeat_penalty=kwargs["repeat_penalty"],
                    )

                enhancement = enhance_story(
                    raw_story,
                    row,
                    rewrite_model="llama-3.2-3b-instruct-q4",
                    generate_meta_fn=rewrite,
                    seed=actual_seed,
                )
                story = enhancement["story"]
                actions = enhancement["actions"]

            return {
                "model_id": spec.id,
                "direction": spec.direction,
                "model_name": spec.name,
                "backend": spec.backend,
                "artifact": _relative(spec.artifact),
                "seed": actual_seed,
                "prompt": row,
                "prompt_sent": prompt,
                "judge_prompt": judge_prompt,
                "story": story,
                "raw_story": raw_story if enhancement else None,
                "format": self._story_format(story, row["teaching"]),
                "raw_format": self._story_format(raw_story, row["teaching"])
                if enhancement
                else None,
                "word_count": _word_count(story),
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "input_tokens": generated.get("input_tokens", 0),
                "output_tokens": generated.get("output_tokens", 0),
                "finish_reason": generated.get("finish_reason"),
                "actions": actions,
                "enhancement": enhancement,
            }

    def close(self) -> None:
        for spec in STUDIO_SPECS:
            try:
                self.unload(spec.id)
            except Exception:
                pass


runtime = StudioRuntime()

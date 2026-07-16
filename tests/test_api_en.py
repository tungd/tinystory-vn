import json
import os
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app, generate_fn, meta_fn, stream_fn, judge_fn

client = TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def _collect(payload):
    r = client.post("/generate/stream", json=payload)
    assert r.status_code == 200
    evs = []
    for b in r.text.split("\n\n"):
        b = b.strip()
        if b.startswith("data:"):
            evs.append(json.loads(b[5:].strip()))
    return evs


def test_models_endpoint():
    r = client.get("/models")
    assert r.status_code == 200
    assert any(m["id"] == "base-qwen3-4b" for m in r.json())


def test_stream_guardrail_off_streams_tokens():
    app.dependency_overrides[stream_fn] = lambda: (
        lambda prompt, system, **kw: iter(["Once ", "upon a time."])
    )
    ev = _collect(
        {
            "character": "a fox",
            "setting": "",
            "challenge": "",
            "outcome": "",
            "teaching": "",
            "length": "short",
            "model_id": "base-qwen3-4b",
            "guardrail_enabled": False,
        }
    )
    toks = "".join(e["text"] for e in ev if e["type"] == "token")
    assert toks == "Once upon a time."
    assert [e for e in ev if e["type"] == "done"][-1]["status"] == "success"


def test_stream_guardrail_on_bad_input_refused_no_tokens():
    ev = _collect(
        {
            "character": "a fucking fox",
            "setting": "",
            "challenge": "",
            "outcome": "",
            "teaching": "",
            "length": "short",
            "model_id": "base-qwen3-4b",
            "guardrail_enabled": True,
        }
    )
    assert not any(e["type"] == "token" for e in ev)
    assert [e for e in ev if e["type"] == "done"][-1]["status"] == "refused"


def test_guardrail_input_violation_logged_with_layer_and_category():
    ev = _collect(
        {
            "character": "a fucking fox",
            "setting": "",
            "challenge": "",
            "outcome": "",
            "teaching": "",
            "length": "short",
            "model_id": "base-qwen3-4b",
            "guardrail_enabled": True,
        }
    )
    blocked = [
        e
        for e in ev
        if e["type"] == "step"
        and e["stage"] == "input_check"
        and e["status"] == "blocked"
    ]
    assert blocked, "expected a blocked input_check step"
    detail = blocked[-1]["detail"]
    assert "Layer 1 BLOCKED" in detail
    assert "profanity" in detail  # the violation category is surfaced in the log


def test_stream_logs_prepare_and_model_steps():
    app.dependency_overrides[stream_fn] = lambda: (
        lambda prompt, system, **kw: iter(["Hi ", "there."])
    )
    ev = _collect(
        {
            "character": "a fox",
            "setting": "",
            "challenge": "",
            "outcome": "",
            "teaching": "",
            "length": "short",
            "model_id": "base-qwen3-4b",
            "guardrail_enabled": False,
        }
    )
    app.dependency_overrides.pop(stream_fn, None)
    stages = [e["stage"] for e in ev if e["type"] == "step"]
    assert "prepare" in stages and "model" in stages
    # generating step reaches a terminal 'ok' with token stats
    gen_ok = [
        e
        for e in ev
        if e["type"] == "step" and e["stage"] == "generating" and e["status"] == "ok"
    ]
    assert gen_ok and "tok/s" in gen_ok[-1]["detail"]


def test_stream_guardrail_on_clean_input_no_tokens():
    app.dependency_overrides[meta_fn] = lambda: (
        lambda prompt, system, **kw: {
            "text": "Once upon a time a kind fox learned to share with friends. The end.",
            "input_tokens": 10,
            "output_tokens": 20,
            "latency_ms": 100,
        }
    )
    ev = _collect(
        {
            "character": "a kind fox",
            "setting": "",
            "challenge": "",
            "outcome": "",
            "teaching": "sharing is caring",
            "length": "short",
            "model_id": "base-qwen3-4b",
            "guardrail_enabled": True,
        }
    )
    assert not any(e["type"] == "token" for e in ev)
    done_ev = [e for e in ev if e["type"] == "done"][-1]
    assert done_ev["status"] == "success"
    assert done_ev["story"] == "Once upon a time a kind fox learned to share with friends. The end."
    stages = [e["stage"] for e in ev if e["type"] == "step"]
    assert "input_check" in stages
    assert "generating" in stages
    assert "output_check" in stages


def test_evaluate_endpoint():
    app.dependency_overrides[judge_fn] = lambda: (
        lambda prompt, system, **kw: '{"grammar":9,"creativity":7,"moral_clarity":8,"prompt_adherence":10}'
    )
    r = client.post(
        "/evaluate",
        json={"story": "...", "prompt": "...", "judge_model_id": "base-qwen3-4b"},
    )
    assert r.json()["overall"] == 8.5


def test_invalid_model_id_422_or_error():
    ev = _collect(
        {
            "character": "a fox",
            "setting": "",
            "challenge": "",
            "outcome": "",
            "teaching": "",
            "length": "short",
            "model_id": "nope",
            "guardrail_enabled": False,
        }
    )
    assert [e for e in ev if e["type"] == "error"] or [
        e for e in ev if e["type"] == "done" and e["status"] == "error"
    ]


# ── New observability tests ────────────────────────────────────────────────────

META_KEYS = {
    "model_id", "model_name", "kind", "temperature", "top_p", "repetition_penalty",
    "num_predict", "seed", "prompt_sent", "input_tokens", "output_tokens",
    "latency_ms", "tokens_per_sec",
}


def test_done_event_has_meta():
    """Guardrail OFF: done event must contain a meta dict with required keys."""
    app.dependency_overrides[stream_fn] = lambda: (
        lambda prompt, system, **kw: iter(["Once ", "upon a time."])
    )
    ev = _collect(
        {
            "character": "a brave rabbit",
            "setting": "a forest",
            "challenge": "a storm",
            "outcome": "finds shelter",
            "teaching": "perseverance",
            "length": "short",
            "model_id": "base-qwen3-4b",
            "guardrail_enabled": False,
        }
    )
    done_ev = [e for e in ev if e["type"] == "done"][-1]
    assert done_ev["status"] == "success"
    assert "meta" in done_ev, "done event must have a meta dict"
    meta = done_ev["meta"]
    for key in META_KEYS:
        assert key in meta, f"meta missing key: {key}"


def test_seed_passed_through():
    """seed=123 sent in request must appear in done.meta['seed']."""
    app.dependency_overrides[stream_fn] = lambda: (
        lambda prompt, system, **kw: iter(["Hello world."])
    )
    ev = _collect(
        {
            "character": "a turtle",
            "setting": "",
            "challenge": "",
            "outcome": "",
            "teaching": "",
            "length": "short",
            "model_id": "base-qwen3-4b",
            "guardrail_enabled": False,
            "seed": 123,
        }
    )
    done_ev = [e for e in ev if e["type"] == "done"][-1]
    assert done_ev["meta"]["seed"] == 123


def test_guardrail_on_done_has_meta_with_input_tokens():
    """Guardrail ON with meta_fn fake: done.meta must contain input_tokens."""
    app.dependency_overrides[meta_fn] = lambda: (
        lambda prompt, system, **kw: {
            "text": "A kind bear shared honey with all his friends in the forest.",
            "input_tokens": 42,
            "output_tokens": 15,
            "latency_ms": 200,
        }
    )
    ev = _collect(
        {
            "character": "a kind bear",
            "setting": "forest",
            "challenge": "loneliness",
            "outcome": "makes friends",
            "teaching": "generosity",
            "length": "short",
            "model_id": "base-qwen3-4b",
            "guardrail_enabled": True,
        }
    )
    done_ev = [e for e in ev if e["type"] == "done"][-1]
    assert done_ev["status"] == "success"
    assert "meta" in done_ev
    assert done_ev["meta"]["input_tokens"] == 42




def test_models_includes_scratch_slms():
    r = client.get("/models")
    ids = {m["id"] for m in r.json()}
    assert {"slm-30m"}.issubset(ids)
    kinds = {m["id"]: m["kind"] for m in r.json()}
    assert kinds["slm-30m"] == "scratch-slm"


# ── Story completeness: done_reason + trim (2026-07-16) ────────────────────────

def test_stream_trims_when_length_cutoff():
    """Guardrail OFF + done_reason=length + mid-sentence text -> done.story trimmed."""
    def fake_stream(prompt, system, on_done=None, **kw):
        for p in ["Once upon a time. ", "But then the wise"]:
            yield p
        if on_done:
            on_done("length")
    app.dependency_overrides[stream_fn] = lambda: fake_stream
    ev = _collect({
        "character": "a fox", "setting": "", "challenge": "", "outcome": "",
        "teaching": "", "length": "short", "model_id": "slm-30m-p2",
        "guardrail_enabled": False,
    })
    done = [e for e in ev if e["type"] == "done"][-1]
    assert done["status"] == "success"
    assert done["story"] == "Once upon a time."   # trimmed at last terminator
    assert any(e["type"] == "step" and "trimmed" in e.get("detail", "").lower() for e in ev)


def test_stream_no_trim_when_stop():
    """done_reason=stop (complete) -> story kept verbatim, no trim step."""
    def fake_stream(prompt, system, on_done=None, **kw):
        yield "The fox shared its food. The end."
        if on_done:
            on_done("stop")
    app.dependency_overrides[stream_fn] = lambda: fake_stream
    ev = _collect({
        "character": "a fox", "setting": "", "challenge": "", "outcome": "",
        "teaching": "", "length": "short", "model_id": "slm-30m-p2",
        "guardrail_enabled": False,
    })
    done = [e for e in ev if e["type"] == "done"][-1]
    assert done["story"] == "The fox shared its food. The end."
    assert not any("trimmed" in e.get("detail", "").lower() for e in ev if e["type"] == "step")


def test_guardrail_on_trims_when_length():
    """Guardrail ON + meta done_reason=length -> story trimmed before output check."""
    app.dependency_overrides[meta_fn] = lambda: (
        lambda prompt, system, **kw: {
            "text": "A kind bear shared honey with friends. Then suddenly the",
            "input_tokens": 10, "output_tokens": 20, "latency_ms": 100,
            "done_reason": "length",
        }
    )
    ev = _collect({
        "character": "a kind bear", "setting": "", "challenge": "", "outcome": "",
        "teaching": "sharing", "length": "short", "model_id": "slm-30m-p2",
        "guardrail_enabled": True,
    })
    done = [e for e in ev if e["type"] == "done"][-1]
    assert done["status"] == "success"
    assert done["story"] == "A kind bear shared honey with friends."


def test_evaluate_includes_objective_and_method():
    """/evaluate now returns objective metrics + methodology metadata (scientific)."""
    app.dependency_overrides[judge_fn] = lambda: (
        lambda prompt, system, **kw: '{"grammar":{"score":8,"reason":"ok"},"creativity":{"score":7,"reason":"ok"},"moral_clarity":{"score":8,"reason":"ok"},"prompt_adherence":{"score":9,"reason":"ok"}}'
    )
    r = client.post("/evaluate", json={
        "story": "The fox shared its food. And everyone was happy in the end.",
        "prompt": "a fox", "judge_model_id": "base-qwen3-4b"})
    d = r.json()
    assert "objective" in d and set(d["objective"]) == {"distinct_1", "distinct_2", "flesch_reading_ease"}
    assert "method" in d and "overall_formula" in d["method"] and "citation" in d["method"]
    assert d["method"]["axes"].keys() >= {"grammar", "creativity", "moral_clarity", "prompt_adherence"}

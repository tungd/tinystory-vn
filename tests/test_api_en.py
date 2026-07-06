import json
from fastapi.testclient import TestClient
import app.main as main_mod
from app.main import app, generate_fn, stream_fn, judge_fn

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

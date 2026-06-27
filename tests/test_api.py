import json

from fastapi.testclient import TestClient

import app.main as main_mod
from app import config
from app.main import app, generate_fn, stream_fn

client = TestClient(app)


def _override_generate(text: str):
    app.dependency_overrides[generate_fn] = lambda: (lambda prompt, system, **kwargs: text)


def _override_capture(captured: dict, text: str = "Ngày xưa có một chú thỏ tốt bụng."):
    def fake_gen(prompt, system, **kwargs):
        captured.update(kwargs)
        return text
    app.dependency_overrides[generate_fn] = lambda: fake_gen


def _override_stream(pieces):
    def fake_stream(prompt, system, **kwargs):
        for p in pieces:
            yield p
    app.dependency_overrides[stream_fn] = lambda: fake_stream


def _collect_stream(payload):
    r = client.post("/generate/stream", json=payload)
    assert r.status_code == 200
    events = []
    for block in r.text.split("\n\n"):
        block = block.strip()
        if block.startswith("data:"):
            events.append(json.loads(block[len("data:"):].strip()))
    return events


def teardown_function():
    app.dependency_overrides.clear()


def test_guardrail_on_blocks_bad_input_without_calling_model():
    _override_generate("KHÔNG ĐƯỢC GỌI")
    r = client.post("/generate", json={
        "topic": "đụ má", "moral": "x", "age_range": "6-8 tuổi", "guardrail_enabled": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "refused"
    assert body["story"] is None


def test_guardrail_on_clean_request_returns_story():
    _override_generate("Ngày xưa có một chú thỏ tốt bụng. Bài học: hãy tử tế.")
    r = client.post("/generate", json={
        "topic": "tình bạn", "moral": "biết sẻ chia", "age_range": "6-8 tuổi", "guardrail_enabled": True,
    })
    body = r.json()
    assert body["status"] == "success"
    assert "chú thỏ" in body["story"]


def test_guardrail_on_filters_bad_output():
    _override_generate("Con thỏ chửi đụ má con rùa.")
    r = client.post("/generate", json={
        "topic": "tình bạn", "moral": "biết sẻ chia", "age_range": "6-8 tuổi", "guardrail_enabled": True,
    })
    body = r.json()
    assert body["status"] == "refused"  # output filter chặn sau khi sinh lại vẫn vi phạm


def test_guardrail_off_bypasses_filters():
    _override_generate("Con thỏ chửi đụ má con rùa.")  # bẩn nhưng guardrail tắt
    r = client.post("/generate", json={
        "topic": "đụ má", "moral": "x", "age_range": "6-8 tuổi", "guardrail_enabled": False,
    })
    body = r.json()
    assert body["status"] == "success"  # không lọc input lẫn output
    assert "đụ má" in body["story"]


def test_model_choice_base_uses_base_model():
    captured = {}
    _override_capture(captured)
    r = client.post("/generate", json={
        "topic": "tình bạn", "moral": "biết sẻ chia", "age_range": "6-8 tuổi",
        "guardrail_enabled": False, "model_choice": "base",
    })
    assert r.json()["status"] == "success"
    assert captured["model"] == config.BASE_MODEL


def test_model_choice_tuned_uses_tuned_model():
    captured = {}
    _override_capture(captured)
    r = client.post("/generate", json={
        "topic": "tình bạn", "moral": "biết sẻ chia", "age_range": "6-8 tuổi",
        "guardrail_enabled": True, "model_choice": "tuned",
    })
    assert r.json()["status"] == "success"
    assert captured["model"] == config.TUNED_MODEL


def test_model_choice_defaults_to_tuned():
    captured = {}
    _override_capture(captured)
    r = client.post("/generate", json={
        "topic": "tình bạn", "moral": "biết sẻ chia", "age_range": "6-8 tuổi",
        "guardrail_enabled": True,
    })
    assert captured["model"] == config.TUNED_MODEL


def test_invalid_model_choice_rejected():
    r = client.post("/generate", json={
        "topic": "tình bạn", "moral": "biết sẻ chia", "age_range": "6-8 tuổi",
        "model_choice": "khong-hop-le",
    })
    assert r.status_code == 422  # pydantic Literal validation


def test_stream_guardrail_off_emits_tokens():
    _override_stream(["Ngày xưa ", "có một chú thỏ."])
    ev = _collect_stream({"topic": "tình bạn", "moral": "sẻ chia", "age_range": "6-8 tuổi",
                          "guardrail_enabled": False, "model_choice": "base"})
    tokens = [e["text"] for e in ev if e["type"] == "token"]
    assert "".join(tokens) == "Ngày xưa có một chú thỏ."
    done = [e for e in ev if e["type"] == "done"][-1]
    assert done["status"] == "success" and done["story"] == "Ngày xưa có một chú thỏ."


def test_stream_guardrail_on_blocks_bad_input_no_tokens():
    _override_generate("KHÔNG ĐƯỢC GỌI")  # buffered fake; không nên được dùng
    ev = _collect_stream({"topic": "đụ má", "moral": "x", "age_range": "6-8 tuổi",
                          "guardrail_enabled": True})
    assert not any(e["type"] == "token" for e in ev)
    assert any(e["type"] == "step" and e["stage"] == "input_check" and e["status"] == "blocked" for e in ev)
    assert [e for e in ev if e["type"] == "done"][-1]["status"] == "refused"


def test_stream_guardrail_on_success_has_steps_and_no_tokens():
    _override_generate("Ngày xưa có một chú thỏ tốt bụng. Bài học: hãy tử tế.")
    ev = _collect_stream({"topic": "tình bạn", "moral": "sẻ chia", "age_range": "6-8 tuổi",
                          "guardrail_enabled": True})
    stages = [e["stage"] for e in ev if e["type"] == "step"]
    assert "input_check" in stages and "generating" in stages and "output_check" in stages
    assert not any(e["type"] == "token" for e in ev)
    assert [e for e in ev if e["type"] == "done"][-1]["status"] == "success"


def test_length_short_passes_smaller_num_predict():
    captured = {}
    def fake_gen(prompt, system, **kwargs):
        captured.update(kwargs)
        return "Ngày xưa có một chú thỏ tốt bụng."
    app.dependency_overrides[generate_fn] = lambda: fake_gen
    client.post("/generate", json={"topic": "tình bạn", "moral": "sẻ chia", "age_range": "6-8 tuổi",
                                   "guardrail_enabled": True, "length": "short"})
    assert captured["num_predict"] == 300


def test_models_endpoint_returns_configured_names():
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert data["base"]["name"] == config.BASE_MODEL
    assert data["tuned"]["name"] == config.TUNED_MODEL
    # có nhãn + mô tả tiếng Việt
    assert data["base"]["label"] and data["base"]["desc"]
    assert data["tuned"]["label"] and data["tuned"]["desc"]

from fastapi.testclient import TestClient

import app.main as main_mod
from app import config
from app.main import app, generate_fn

client = TestClient(app)


def _override_generate(text: str):
    app.dependency_overrides[generate_fn] = lambda: (lambda prompt, system, model=None: text)


def _override_capture(captured: dict, text: str = "Ngày xưa có một chú thỏ tốt bụng."):
    def fake_gen(prompt, system, model=None):
        captured["model"] = model
        return text
    app.dependency_overrides[generate_fn] = lambda: fake_gen


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

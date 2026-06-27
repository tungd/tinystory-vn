from fastapi.testclient import TestClient

import app.main as main_mod
from app.main import app, generate_fn

client = TestClient(app)


def _override_generate(text: str):
    app.dependency_overrides[generate_fn] = lambda: (lambda prompt, system: text)


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

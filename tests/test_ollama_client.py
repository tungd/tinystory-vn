import json

import httpx
import pytest

import app.ollama_client as oc

# Keep a reference to the real httpx.Client before any patching occurs.
_RealClient = httpx.Client


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_generate_returns_message_content(monkeypatch):
    def handler(request):
        assert request.url.path == "/api/chat"
        body = request.read().decode()
        assert "system" in body and "Ngày xưa" not in body
        return httpx.Response(200, json={"message": {"content": "Ngày xưa có một chú thỏ."}})

    # Use _RealClient (not oc.httpx.Client) to avoid infinite recursion when the
    # monkeypatched lambda calls httpx.Client, which would call the lambda again.
    monkeypatch.setattr(oc.httpx, "Client", lambda **kw: _RealClient(transport=_mock_transport(handler), **{k: v for k, v in kw.items() if k != "transport"}))
    out = oc.generate(prompt="Viết truyện về tình bạn", system="Bạn là người kể truyện.")
    assert out == "Ngày xưa có một chú thỏ."


def test_generate_raises_on_http_error(monkeypatch):
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    monkeypatch.setattr(oc.httpx, "Client", lambda **kw: _RealClient(transport=_mock_transport(handler), **{k: v for k, v in kw.items() if k != "transport"}))
    with pytest.raises(oc.OllamaError):
        oc.generate(prompt="x", system="y")


def test_generate_stream_yields_content_pieces(monkeypatch):
    body = "\n".join([
        json.dumps({"message": {"content": "Ngày xưa "}}),
        json.dumps({"message": {"content": "có một chú thỏ."}}),
        json.dumps({"done": True}),
    ])

    def handler(request):
        assert request.url.path == "/api/chat"
        return httpx.Response(200, text=body)

    monkeypatch.setattr(oc.httpx, "Client",
        lambda **kw: _RealClient(transport=_mock_transport(handler),
                                 **{k: v for k, v in kw.items() if k != "transport"}))
    pieces = list(oc.generate_stream(prompt="x", system="y"))
    assert "".join(pieces) == "Ngày xưa có một chú thỏ."


def test_generate_payload_disables_thinking(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.read().decode()))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    monkeypatch.setattr(oc.httpx, "Client",
        lambda **kw: _RealClient(transport=_mock_transport(handler),
                                 **{k: v for k, v in kw.items() if k != "transport"}))
    oc.generate(prompt="x", system="y")
    assert seen.get("think") is False


def test_generate_stream_payload_disables_thinking(monkeypatch):
    seen = {}
    body = "\n".join([
        json.dumps({"message": {"content": "hello"}}),
        json.dumps({"done": True}),
    ])

    def handler(request):
        seen.update(json.loads(request.read().decode()))
        return httpx.Response(200, text=body)

    monkeypatch.setattr(oc.httpx, "Client",
        lambda **kw: _RealClient(transport=_mock_transport(handler),
                                 **{k: v for k, v in kw.items() if k != "transport"}))
    list(oc.generate_stream(prompt="x", system="y"))
    assert seen.get("think") is False


def test_generate_meta_returns_text_and_counts(monkeypatch):
    def handler(request):
        body = json.loads(request.read().decode())
        assert body.get("stream") is False
        return httpx.Response(
            200,
            json={
                "message": {"content": "A fox learned wisdom."},
                "prompt_eval_count": 30,
                "eval_count": 5,
            },
        )

    monkeypatch.setattr(
        oc.httpx,
        "Client",
        lambda **kw: _RealClient(
            transport=_mock_transport(handler),
            **{k: v for k, v in kw.items() if k != "transport"},
        ),
    )
    result = oc.generate_meta(prompt="Tell a fable", system="You are a storyteller.", seed=42)
    assert result["text"] == "A fox learned wisdom."
    assert result["input_tokens"] == 30
    assert result["output_tokens"] == 5
    assert isinstance(result["latency_ms"], int)
    assert result["latency_ms"] >= 0


def test_generate_meta_raises_on_http_error(monkeypatch):
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    monkeypatch.setattr(
        oc.httpx,
        "Client",
        lambda **kw: _RealClient(
            transport=_mock_transport(handler),
            **{k: v for k, v in kw.items() if k != "transport"},
        ),
    )
    with pytest.raises(oc.OllamaError):
        oc.generate_meta(prompt="x", system="y")


def test_generate_seed_included_in_options(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.read().decode()))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    monkeypatch.setattr(
        oc.httpx,
        "Client",
        lambda **kw: _RealClient(
            transport=_mock_transport(handler),
            **{k: v for k, v in kw.items() if k != "transport"},
        ),
    )
    oc.generate(prompt="x", system="y", seed=99)
    assert seen.get("options", {}).get("seed") == 99

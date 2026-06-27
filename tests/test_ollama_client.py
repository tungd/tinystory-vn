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

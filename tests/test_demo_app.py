from types import SimpleNamespace

from demo.judge import GoogleJudge
from demo.runtime import STUDIO_SPECS, StudioRuntime
from demo.server import CompareRequest, StoryRequest, compare_models, load_presets


def test_demo_registry_has_the_five_benchmark_directions():
    assert [spec.direction for spec in STUDIO_SPECS] == ["E1", "E2", "E3", "E4", "E5"]
    assert [spec.backend for spec in STUDIO_SPECS] == [
        "llama_completion",
        "mlx",
        "transformers_peft",
        "llama_repair",
        "llama_chat",
    ]


def test_demo_exposes_the_canonical_25_judge_cases():
    presets = load_presets()
    assert len(presets) == 25
    assert [preset["prompt_id"] for preset in presets] == [f"GJ{index:02d}" for index in range(1, 26)]
    assert all(preset["length"] == "medium" for preset in presets)
    assert all({"character", "setting", "challenge", "outcome", "teaching"} <= preset.keys() for preset in presets)


def test_demo_runtime_reports_artifacts_without_loading_processes():
    snapshots = StudioRuntime().snapshots()
    assert len(snapshots) == 5
    assert all(item["status"] in {"unloaded", "unavailable"} for item in snapshots)
    assert all(item["loaded"] is False for item in snapshots)


def test_e1_uses_the_full_five_field_benchmark_prompt():
    row = {
        "character": "a patient fox",
        "setting": "a quiet forest",
        "challenge": "the river rises",
        "outcome": "the fox finds a safe path",
        "teaching": "Kindness helps everyone",
        "length": "short",
    }
    prompt = StudioRuntime._model_prompt(STUDIO_SPECS[0], row)
    assert "Main character: a patient fox" in prompt
    assert "Setting: a quiet forest" in prompt
    assert "Challenge: the river rises" in prompt
    assert "Outcome: the fox finds a safe path" in prompt
    assert "Teaching/Moral: Kindness helps everyone" in prompt


def test_judge_is_explicitly_google_not_ollama(monkeypatch):
    monkeypatch.setenv("FABLE_JUDGE_BACKEND", "google")
    monkeypatch.setenv("FABLE_JUDGE_MODEL_ID", "demo-judge")
    monkeypatch.setenv("FABLE_JUDGE_API_KEY", "test-key")
    status = GoogleJudge().status()
    assert status["backend"] == "google"
    assert status["provider"] == "Google GenAI"
    assert "ollama" not in status["backend"].lower()
    assert status["model_id"] == "demo-judge"


def test_judge_schema_is_the_compact_four_score_protocol():
    schema = GoogleJudge._schema()
    assert set(schema["properties"]) == {
        "grammar",
        "creativity",
        "moral_clarity",
        "prompt_adherence",
    }
    assert all(value["type"] == "integer" for value in schema["properties"].values())
    assert all("reason" not in value for value in schema["properties"].values())


def test_judge_accepts_a_structured_flat_response_without_text():
    response = SimpleNamespace(
        text="",
        parsed={
            "grammar": 8,
            "creativity": 7,
            "moral_clarity": 9,
            "prompt_adherence": 8,
        },
    )
    scores, raw = GoogleJudge._scores_from_response(response)
    assert scores["overall"] == 8.0
    assert raw.startswith('{"grammar": 8')


def test_compare_runs_each_selected_model_and_keeps_per_model_failures(monkeypatch):
    calls = []

    def fake_run(model_id, payload, seed=None):
        calls.append((model_id, payload["prompt_id"] if "prompt_id" in payload else None, seed))
        if model_id == "e5-fable-q4":
            raise RuntimeError("runtime unavailable")
        return {"model_id": model_id, "story": "A complete story."}

    monkeypatch.setattr("demo.server.runtime.run", fake_run)
    request = CompareRequest(
        model_ids=["e1-slm-60m", "e5-fable-q4"],
        prompt=StoryRequest(character="A fox", seed=5410),
    )

    result = compare_models(request)

    assert calls == [("e1-slm-60m", None, 5410), ("e5-fable-q4", None, 5410)]
    assert result["completed"] == 1
    assert result["results"][0]["story"] == "A complete story."
    assert result["results"][1] == {"model_id": "e5-fable-q4", "error": "runtime unavailable"}

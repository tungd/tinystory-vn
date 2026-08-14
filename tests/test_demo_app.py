from pathlib import Path
from types import SimpleNamespace

import numpy as np

from demo.judge import GoogleJudge
from demo.runtime import STUDIO_SPECS, StudioRuntime
from demo.server import CompareRequest, StoryRequest, compare_models, load_presets
from app.prompt_en import SMOLLM2_TF1_SYSTEM_MESSAGE, TF1_FABLE_PROMPT
from scripts.convert_v16_mlx import convert_key_and_tensor


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
    assert prompt.endswith("\n<|story|>")


def test_e3_uses_the_original_tf1_system_and_prompt_contract():
    row = {
        "character": "a patient fox",
        "setting": "a quiet forest",
        "challenge": "the river rises",
        "outcome": "the fox finds a safe path",
        "teaching": "Kindness helps everyone",
        "length": "medium",
    }

    prompt = StudioRuntime._model_prompt(STUDIO_SPECS[2], row)

    assert prompt == (
        f"{SMOLLM2_TF1_SYSTEM_MESSAGE}\n\n"
        + TF1_FABLE_PROMPT.format(**row)
        + "\n\n"
    )


def test_e2_uses_the_v16_training_prompt_contract():
    row = {
        "character": "a tidy squirrel who labels every acorn",
        "setting": "a great oak during a busy autumn",
        "challenge": "a storm mixes all the stores",
        "outcome": "she creates a simple shared pantry",
        "teaching": "Good plans improve when we listen to others.",
        "length": "medium",
    }

    prompt = StudioRuntime._model_prompt(STUDIO_SPECS[1], row)

    assert prompt == (
        "<char> a tidy squirrel who labels every acorn </char>\n"
        "<moral> Good plans improve when we listen to others. </moral>\n"
        "<story>\n"
    )


def test_e2_uses_the_selected_v16_conditioning_checkpoint():
    e2 = STUDIO_SPECS[1]

    assert e2.name == "V16 conditioned"
    assert e2.artifact.name == "conditioning-mlx"
    assert e2.artifact.is_dir()


def test_v16_mlx_conversion_transposes_conv1d_weights_contiguously():
    source = np.arange(6, dtype=np.float32).reshape(2, 3)

    key, converted = convert_key_and_tensor(
        "transformer.h.0.attn.c_attn.weight", source
    )

    assert key == "model.h.0.attn.c_attn.weight"
    assert converted.flags.c_contiguous
    assert np.array_equal(converted, source.T)


def test_judge_request_is_an_evaluation_brief_not_a_generation_prompt():
    row = {
        "character": "a patient fox",
        "setting": "a quiet forest",
        "challenge": "the river rises",
        "outcome": "the fox finds a safe path",
        "teaching": "Kindness helps everyone",
        "length": "short",
    }

    prompt = StudioRuntime._judge_prompt(STUDIO_SPECS[0], row)

    assert "Main character: a patient fox" in prompt
    assert "Teaching/Moral: Kindness helps everyone" in prompt
    assert "Do not stop until" not in prompt
    assert "final line" not in prompt.lower()


def test_judge_does_not_score_the_literal_moral_footer():
    prompt = GoogleJudge._prompt(
        "A complete story.\n\nMoral: kindness matters.",
        "Teaching/Moral: kindness matters",
    )

    assert "final Moral line" not in prompt
    assert "literal `Moral:` footer" in prompt
    assert "do not reward or penalize" in prompt.lower()


def test_story_format_is_reported_separately_from_judge_scores():
    report = StudioRuntime._story_format(
        "A complete story.\n\nMoral: kindness matters.",
        "kindness matters",
    )

    assert report == {
        "has_moral_line": True,
        "moral_is_final": True,
        "moral_is_exact": True,
    }


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


def test_judge_rejects_a_structured_response_with_a_missing_axis():
    response = SimpleNamespace(
        text="",
        parsed={
            "grammar": 8,
            "creativity": 7,
            "moral_clarity": 9,
        },
    )

    scores, _ = GoogleJudge._scores_from_response(response)

    assert scores["prompt_adherence"] == 0


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


def test_compare_ui_has_a_direct_output_switcher():
    html = (Path(__file__).parents[1] / "demo/static/index.html").read_text(encoding="utf-8")

    assert "Quick switch" in html
    assert "data-compare-select" in html
    assert "data-compare-prev" in html
    assert "Judge selected output" in html
    assert "Format is reported separately" in html

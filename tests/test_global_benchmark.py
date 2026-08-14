import json

from app.prompt_en import (
    SMOLLM2_TF1_SYSTEM_MESSAGE,
    TF1_FABLE_PROMPT,
    build_fable_prompt,
)
from scripts.global_benchmark import model_prompt
from scripts import global_benchmark


ROW = {
    "character": "a patient fox",
    "setting": "a quiet forest",
    "challenge": "the river rises",
    "outcome": "the fox finds a safe path",
    "teaching": "Kindness helps everyone",
    "length": "medium",
}


def test_global_slm_runner_appends_the_story_separator():
    expected = build_fable_prompt(
        ROW["character"],
        ROW["setting"],
        ROW["challenge"],
        ROW["outcome"],
        ROW["teaching"],
        "Write a medium-length fable (about 200-260 words).",
    ) + "\n<|story|>"

    assert model_prompt("e2", ROW) == expected


def test_global_smollm2_runner_replays_the_training_context():
    assert model_prompt("e4", ROW) == (
        f"{SMOLLM2_TF1_SYSTEM_MESSAGE}\n\n"
        + TF1_FABLE_PROMPT.format(**ROW)
        + "\n\n"
    )


def test_global_judge_receives_an_evaluation_brief_not_a_generation_prompt(monkeypatch, tmp_path):
    generations = []
    for index in range(1, 26):
        prompt = {"prompt_id": f"GJ{index:02d}", **ROW}
        generations.append(
            {
                "prompt_id": prompt["prompt_id"],
                "prompt": prompt,
                "formatted_prompt": global_benchmark.prompt_text(prompt),
                "story": "A patient fox finds a safe path.",
            }
        )
    monkeypatch.setattr(global_benchmark, "CANDIDATES", {"e1": {}})
    monkeypatch.setattr(global_benchmark, "GEN_DIR", tmp_path)
    (tmp_path / "e1.jsonl").write_text(
        "\n".join(json.dumps(generation) for generation in generations) + "\n",
        encoding="utf-8",
    )

    item = global_benchmark.blind_items()[0]

    assert "Do not stop until" in item["formatted_prompt"]
    assert "Do not stop until" not in item["judge_prompt"]

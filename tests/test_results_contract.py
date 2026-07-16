import json
from pathlib import Path


def test_committed_results_match_results_panel_contract():
    summary = json.loads(Path("results/eval_summary.json").read_text())

    assert summary["model"] == "fable-64M"
    assert summary["params_M"] == 63.0
    assert summary["metrics"] == summary["objective"]["finetuned"] | {"n": 4}
    assert set(summary["objective"]) == {"base", "finetuned"}

"""Create SVG figures from automatic and human evaluation files."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"

AUTO_FILES = {
    "Base": RESULTS / "eval_base.json",
    "Strict Prompt": RESULTS / "eval_strict_prompt.json",
    "Strict + Post": RESULTS / "eval_strict_postprocess.json",
    "Base + Repair": RESULTS / "eval_base_repair.json",
    "SFT Clean 3K": RESULTS / "eval_sft_clean3k.json",
    "Failure LoRA": RESULTS / "eval_failure_lora.json",
    "Fluency SFT v1": RESULTS / "eval_fluency_sft_v1.json",
}

METRIC_LABELS = {
    "has_moral_footer_rate": "Moral footer",
    "moral_exact_rate": "Moral exact",
    "outcome_covered_rate": "Outcome",
    "clean_ending_rate": "Clean ending",
}

HUMAN_FINAL = RESULTS / "human_eval_final_5way_10.csv"
HUMAN_FLUENCY = RESULTS / "human_eval_fluency_sft_v1_10.csv"


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def load_auto() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for name, path in AUTO_FILES.items():
        if path.exists():
            rows[name] = json.loads(path.read_text(encoding="utf-8"))["summary"]
    return rows


def load_human(path: Path) -> dict[str, dict[str, float]]:
    fields = {
        "score_english_fluency_1_5": "Fluency",
        "score_prompt_adherence_1_5": "Adherence",
        "score_fable_structure_1_5": "Structure",
        "score_moral_clarity_1_5": "Moral",
        "score_child_safety_1_5": "Safety",
        "average_score": "Overall",
    }
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model_name"]
            for src, label in fields.items():
                raw = row.get(src, "").strip()
                if raw:
                    values[model][label].append(float(raw))
    return {
        model: {label: sum(nums) / len(nums) for label, nums in model_values.items()}
        for model, model_values in values.items()
    }


def grouped_bar_svg(
    title: str,
    rows: dict[str, dict[str, float]],
    metrics: list[str],
    labels: dict[str, str],
    max_value: float,
    path: Path,
) -> None:
    width = 1180
    height = 620
    margin_left = 90
    margin_right = 30
    margin_top = 70
    margin_bottom = 130
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    models = list(rows)
    group_w = chart_w / max(1, len(models))
    bar_gap = 4
    bar_w = (group_w - 18) / max(1, len(metrics))
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{xml_escape(title)}</text>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = margin_top + chart_h - (value / max_value) * chart_h
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#374151">{value:.1f}</text>')
    for model_index, model in enumerate(models):
        x0 = margin_left + model_index * group_w + 9
        for metric_index, metric in enumerate(metrics):
            value = rows[model].get(metric, 0.0)
            h = (value / max_value) * chart_h
            x = x0 + metric_index * (bar_w + bar_gap)
            y = margin_top + chart_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[metric_index % len(colors)]}"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#111827">{value:.2f}</text>')
        label_x = x0 + (len(metrics) * (bar_w + bar_gap)) / 2
        parts.append(
            f'<text x="{label_x:.1f}" y="{height - 78}" text-anchor="end" transform="rotate(-35 {label_x:.1f},{height - 78})" '
            f'font-family="Arial" font-size="13" fill="#111827">{xml_escape(model)}</text>'
        )
    legend_x = margin_left
    legend_y = height - 28
    for i, metric in enumerate(metrics):
        x = legend_x + i * 190
        parts.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{colors[i % len(colors)]}"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y}" font-family="Arial" font-size="13" fill="#111827">{xml_escape(labels[metric])}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    auto = load_auto()
    grouped_bar_svg(
        "Automatic Reliability Metrics",
        auto,
        list(METRIC_LABELS),
        METRIC_LABELS,
        1.0,
        OUT / "automatic_reliability.svg",
    )
    if HUMAN_FINAL.exists():
        human = load_human(HUMAN_FINAL)
        grouped_bar_svg(
            "Human Evaluation - Final 5-Way",
            human,
            ["Fluency", "Adherence", "Structure", "Moral", "Safety", "Overall"],
            {
                "Fluency": "Fluency",
                "Adherence": "Adherence",
                "Structure": "Structure",
                "Moral": "Moral",
                "Safety": "Safety",
                "Overall": "Overall",
            },
            5.0,
            OUT / "human_eval_final_5way.svg",
        )
    if HUMAN_FLUENCY.exists():
        human = load_human(HUMAN_FLUENCY)
        grouped_bar_svg(
            "Human Evaluation - Fluency SFT v1",
            human,
            ["Fluency", "Adherence", "Structure", "Moral", "Safety", "Overall"],
            {
                "Fluency": "Fluency",
                "Adherence": "Adherence",
                "Structure": "Structure",
                "Moral": "Moral",
                "Safety": "Safety",
                "Overall": "Overall",
            },
            5.0,
            OUT / "human_eval_fluency_sft_v1.svg",
        )
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()

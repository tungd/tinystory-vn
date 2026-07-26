#!/usr/bin/env python3
"""Render report figures from the no-retraining ablation summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "ablation_judge" / "summary.json"
OUTPUT = ROOT / "figures" / "ablation"

BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#16a34a"
GRAY = "#94a3b8"


def label_bars(axis: plt.Axes, bars, digits: int = 2, suffix: str = "") -> None:
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.{digits}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def condition_availability(data: dict) -> None:
    models = ["E1 · 60M", "E5 · 3B"]
    keys = ["e2", "e5"]
    x = np.arange(len(models))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    full = [data[key]["full"]["slot_coverage_5_mean"] for key in keys]
    two = [data[key]["two_slot"]["slot_coverage_5_mean"] for key in keys]
    bars_a = axes[0].bar(x - width / 2, full, width, label="Đủ 5 slot", color=BLUE)
    bars_b = axes[0].bar(x + width / 2, two, width, label="Character + teaching", color=GRAY)
    axes[0].set_title("Coverage đúng theo yêu cầu")
    axes[0].set_ylabel("Số slot đúng / 5")
    axes[0].set_ylim(0, 5.5)
    axes[0].set_xticks(x, models)
    axes[0].grid(axis="y", alpha=0.2)
    label_bars(axes[0], bars_a)
    label_bars(axes[0], bars_b)

    full = [data[key]["full"]["requested_causal_consistency"] for key in keys]
    two = [data[key]["two_slot"]["requested_causal_consistency"] for key in keys]
    bars_a = axes[1].bar(x - width / 2, full, width, label="Đủ 5 slot", color=BLUE)
    bars_b = axes[1].bar(x + width / 2, two, width, label="Character + teaching", color=GRAY)
    axes[1].set_title("Nhân quả theo chuỗi được yêu cầu")
    axes[1].set_ylabel("Điểm / 10")
    axes[1].set_ylim(0, 10.8)
    axes[1].set_xticks(x, models)
    axes[1].grid(axis="y", alpha=0.2)
    label_bars(axes[1], bars_a)
    label_bars(axes[1], bars_b)
    axes[1].legend(frameon=False, loc="upper left")

    fig.suptitle("Ablation về số điều kiện: cùng 25 đề và cùng seed", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT / "20_condition_availability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def counterfactual(data: dict) -> None:
    models = ["E1 · 60M", "E5 · 3B"]
    keys = ["e2", "e5"]
    x = np.arange(len(models))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    match = [100 * data[key]["pair_match_both_rate"] for key in keys]
    changed = [100 * data[key]["intervention_changes_story_rate"] for key in keys]
    bars_a = axes[0].bar(x - width / 2, match, width, label="Cả hai khớp", color=BLUE)
    bars_b = axes[0].bar(x + width / 2, changed, width, label="Can thiệp đổi truyện", color=GREEN)
    axes[0].set_title("Kết quả trên 10 cặp counterfactual")
    axes[0].set_ylabel("Tỷ lệ (%)")
    axes[0].set_ylim(0, 112)
    axes[0].set_xticks(x, models)
    axes[0].grid(axis="y", alpha=0.2)
    label_bars(axes[0], bars_a, digits=0, suffix="%")
    label_bars(axes[0], bars_b, digits=0, suffix="%")
    axes[0].legend(frameon=False, loc="upper left")

    means = [data[key]["counterfactual_sensitivity"] for key in keys]
    cis = [data[key]["counterfactual_sensitivity_95pct_ci"] for key in keys]
    errors = np.array(
        [[mean - ci[0] for mean, ci in zip(means, cis)], [ci[1] - mean for mean, ci in zip(means, cis)]]
    )
    bars = axes[1].bar(x, means, 0.52, color=[ORANGE, BLUE], yerr=errors, capsize=5)
    axes[1].set_title("Độ nhạy với thay đổi đúng một slot")
    axes[1].set_ylabel("Điểm / 10 (CI 95%)")
    axes[1].set_ylim(0, 10.8)
    axes[1].set_xticks(x, models)
    axes[1].grid(axis="y", alpha=0.2)
    label_bars(axes[1], bars)

    fig.suptitle(
        "Counterfactual evaluation: 5 cặp trait + 5 cặp outcome",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "21_counterfactual_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def repair_effect(data: dict) -> None:
    modes = ["Raw", "Sau repair"]
    x = np.arange(len(modes))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    exact = [100 * data["raw"]["exact_requested_moral_rate"], 100 * data["repaired"]["exact_requested_moral_rate"]]
    teaching = [100 * data["raw"]["teaching_covered"], 100 * data["repaired"]["teaching_covered"]]
    width = 0.34
    bars_a = axes[0].bar(x - width / 2, exact, width, label="Moral đúng nguyên teaching", color=BLUE)
    bars_b = axes[0].bar(x + width / 2, teaching, width, label="Teaching được thể hiện", color=GREEN)
    axes[0].set_title("Hợp đồng bài học")
    axes[0].set_ylabel("Tỷ lệ (%)")
    axes[0].set_ylim(0, 112)
    axes[0].set_xticks(x, modes)
    axes[0].grid(axis="y", alpha=0.2)
    label_bars(axes[0], bars_a, digits=0, suffix="%")
    label_bars(axes[0], bars_b, digits=0, suffix="%")
    axes[0].legend(frameon=False, loc="lower right")

    internal = [data["raw"]["internal_causal_consistency"], data["repaired"]["internal_causal_consistency"]]
    requested = [data["raw"]["requested_causal_consistency"], data["repaired"]["requested_causal_consistency"]]
    bars_a = axes[1].bar(x - width / 2, internal, width, label="Nhân quả nội tại", color=GRAY)
    bars_b = axes[1].bar(x + width / 2, requested, width, label="Nhân quả theo yêu cầu", color=ORANGE)
    axes[1].set_title("Độ nhất quán nhân quả")
    axes[1].set_ylabel("Điểm / 10")
    axes[1].set_ylim(0, 10.4)
    axes[1].set_xticks(x, modes)
    axes[1].grid(axis="y", alpha=0.2)
    label_bars(axes[1], bars_a)
    label_bars(axes[1], bars_b)
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle("E4 trước và sau repair trên cùng 25 truyện", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT / "22_e4_repair_effect.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    condition_availability(data["condition_availability"])
    counterfactual(data["counterfactual"])
    repair_effect(data["e3_repair"])
    print(f"wrote figures to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

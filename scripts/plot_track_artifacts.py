#!/usr/bin/env python3
"""Render per-track figures from archived experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "tracks"
BLUE = "#2F6BDE"
LIGHT_BLUE = "#74BDE0"
GREEN = "#1B9E77"
ORANGE = "#E66101"
GRAY = "#8FA1B8"
BLACK = "#111111"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def finish(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)


def plot_e2_training() -> None:
    source = ROOT / "tmp" / "td-drive-upload" / "E1-V16"
    pretrain = load_json(
        source / "training-states" / "pretrain_trainer_state.json"
    )["log_history"]
    conditioning = load_json(
        source
        / "training-states"
        / "conditioning_trainer_state_step800.json"
    )["log_history"]
    run = load_json(
        source / "repo-evidence" / "runs" / "v16" / "run.json"
    )
    final_eval = load_json(
        ROOT
        / "runs"
        / "v16"
        / "artifacts"
        / "conditioning"
        / "v16_final_eval.json"
    )

    pre_steps = np.array([row["step"] for row in pretrain])
    pre_loss = np.array([row["loss"] for row in pretrain])
    pre_lr = np.array([row["learning_rate"] for row in pretrain])
    cond_train = [row for row in conditioning if "loss" in row]
    cond_eval = [row for row in conditioning if "eval_loss" in row]
    cond_steps = np.array([row["step"] for row in cond_train])
    cond_loss = np.array([row["loss"] for row in cond_train])
    cond_lr = np.array([row["learning_rate"] for row in cond_train])

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.6))
    fig.suptitle(
        "E2 — Diễn biến huấn luyện V16",
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].plot(pre_steps, pre_loss, color=BLUE, linewidth=1.8)
    axes[0, 0].scatter(
        [pre_steps[-1]], [pre_loss[-1]], color=BLACK, s=24, zorder=3
    )
    axes[0, 0].annotate(
        f"{pre_loss[-1]:.3f}",
        (pre_steps[-1], pre_loss[-1]),
        xytext=(-34, 10),
        textcoords="offset points",
        fontweight="bold",
    )
    axes[0, 0].set(
        title="Tiền huấn luyện: cross-entropy loss",
        xlabel="Bước",
        ylabel="Loss",
    )

    axes[0, 1].plot(pre_steps, pre_lr, color=ORANGE, linewidth=1.8)
    axes[0, 1].set(
        title="Tiền huấn luyện: learning rate",
        xlabel="Bước",
        ylabel="Learning rate",
    )
    axes[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    axes[1, 0].plot(
        cond_steps,
        cond_loss,
        color=BLUE,
        linewidth=1.8,
        label="Train loss (log đến bước 800)",
    )
    axes[1, 0].scatter(
        [row["step"] for row in cond_eval],
        [row["eval_loss"] for row in cond_eval],
        color=ORANGE,
        marker="D",
        s=34,
        label="Eval loss đã lưu",
        zorder=3,
    )
    endpoint_step = int(run["curriculum"]["conditioning"]["steps"])
    endpoint_train = float(run["curriculum"]["conditioning"]["final_loss"])
    endpoint_eval = float(final_eval["eval_loss"])
    axes[1, 0].plot(
        [cond_steps[-1], endpoint_step],
        [cond_loss[-1], endpoint_train],
        color=GRAY,
        linestyle="--",
        linewidth=1.2,
        label="Đoạn không có log",
    )
    axes[1, 0].scatter(
        [endpoint_step],
        [endpoint_train],
        color=BLACK,
        s=28,
        zorder=4,
    )
    axes[1, 0].scatter(
        [endpoint_step],
        [endpoint_eval],
        color=GREEN,
        marker="D",
        s=38,
        zorder=4,
    )
    axes[1, 0].annotate(
        f"endpoint {endpoint_eval:.3f}",
        (endpoint_step, endpoint_eval),
        xytext=(-86, 12),
        textcoords="offset points",
        fontweight="bold",
    )
    axes[1, 0].set(
        title="Điều kiện hóa: train/eval loss",
        xlabel="Bước",
        ylabel="Loss",
    )
    axes[1, 0].legend(fontsize=8, loc="upper right")

    axes[1, 1].plot(cond_steps, cond_lr, color=ORANGE, linewidth=1.8)
    axes[1, 1].set(
        title="Điều kiện hóa: learning rate",
        xlabel="Bước",
        ylabel="Learning rate",
    )
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Nguồn: artifact huấn luyện V16. Log theo bước của pha điều kiện hóa kết thúc "
        "ở bước 800; điểm cuối tại bước 1.611 được đánh dấu riêng.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    finish(fig, "e2_training_curves.png")


def plot_e2_screen() -> None:
    run = load_json(
        ROOT
        / "tmp"
        / "td-drive-upload"
        / "E1-V16"
        / "repo-evidence"
        / "runs"
        / "v16"
        / "run.json"
    )
    models = run["screen"]["models"]
    order = [
        "v3-full",
        "v16-conditioned",
        "v16-causal-e1",
        "v16-causal-e2",
        "v16-causal-e3",
    ]
    labels = ["V3", "V16 cond.", "Causal E1", "Causal E2", "Causal E3"]
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    fig.suptitle(
        "E2 — Màn sàng lọc 20 đề cho năm checkpoint",
        fontsize=14,
        fontweight="bold",
    )
    width = 0.36
    fluency = [models[name]["fluency"] for name in order]
    moral = [models[name]["moral_delivery"] for name in order]
    axes[0].bar(x - width / 2, fluency, width, label="Fluency /10", color=BLUE)
    axes[0].bar(
        x + width / 2, moral, width, label="Moral delivery /10", color=GRAY
    )
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 8)
    axes[0].set(title="Chất lượng bề mặt", ylabel="Điểm trung bình")
    axes[0].legend()
    for index, value in enumerate(fluency):
        axes[0].text(
            index - width / 2,
            value + 0.12,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )

    pass_metrics = [
        ("Trait→choice", "trait_drives_choice", BLUE),
        ("Causal pass", "causal_pass", ORANGE),
        ("Strict pass", "strict_pass", BLACK),
    ]
    bar_width = 0.24
    for offset, (label, key, color) in enumerate(pass_metrics):
        values = [100 * models[name][key] for name in order]
        axes[1].bar(
            x + (offset - 1) * bar_width,
            values,
            bar_width,
            label=label,
            color=color,
        )
    axes[1].axhline(
        10,
        color="#B42318",
        linestyle="--",
        linewidth=1.2,
        label="Cổng causal 10%",
    )
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylim(0, 12)
    axes[1].set(title="Tỷ lệ vượt kiểm tra nhân quả", ylabel="Tỷ lệ (%)")
    axes[1].legend(fontsize=8, ncol=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "V16-conditioned tăng fluency 0,95 điểm so với V3 nhưng đạt 0% causal pass; "
        "không checkpoint V16 nào đạt cổng 10%.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    finish(fig, "e2_screen_metrics.png")


def plot_e2_tokenizer_metrics() -> None:
    labels = ["V1 · 29,9M\nBPE thô", "V2 · 63M\nMetaspace BPE"]
    x = np.arange(2)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    fig.suptitle(
        "E2 — Ảnh hưởng của tokenizer và cấu hình nền",
        fontsize=14,
        fontweight="bold",
    )

    width = 0.34
    axes[0].bar(
        x - width / 2, [0.389, 0.519], width, label="Distinct-1", color=BLUE
    )
    axes[0].bar(
        x + width / 2, [0.857, 0.922], width, label="Distinct-2", color=GREEN
    )
    axes[0].set(title="Đa dạng từ vựng", ylabel="Tỷ lệ", ylim=(0, 1.05))
    axes[0].legend(fontsize=8)

    axes[1].bar(x, [0.078, 0.028], color=[GRAY, BLUE], width=0.55)
    axes[1].set(
        title="Self-BLEU (thấp hơn tốt hơn)",
        ylabel="Điểm",
        ylim=(0, 0.09),
    )

    axes[2].bar(x, [82.9, 81.5], color=[GRAY, BLUE], width=0.55)
    axes[2].set(
        title="Flesch Reading Ease",
        ylabel="Điểm",
        ylim=(0, 90),
    )
    for axis, values, fmt in [
        (axes[0], None, None),
        (axes[1], [0.078, 0.028], "{:.3f}"),
        (axes[2], [82.9, 81.5], "{:.1f}"),
    ]:
        axis.set_xticks(x, labels)
        axis.spines[["top", "right"]].set_visible(False)
        if values is not None:
            for index, value in enumerate(values):
                axis.text(
                    index,
                    value + (0.003 if axis is axes[1] else 1.2),
                    fmt.format(value),
                    ha="center",
                    fontsize=8,
                )
    fig.text(
        0.5,
        0.01,
        "Metaspace sửa ranh giới từ; V2 tăng độ đa dạng và giảm lặp giữa các truyện "
        "mà gần như giữ nguyên độ dễ đọc.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    finish(fig, "e2_tokenizer_metrics.png")


def plot_e2_v3_failure_chain() -> None:
    labels = [
        "Nhân vật\nchủ động",
        "Xung đột\nphù hợp",
        "Trait→\nchoice",
        "Choice xử lý\nxung đột",
        "Hệ quả theo\nlựa chọn",
        "Plot suy ra\nmoral",
        "Kết thúc\nđược giải quyết",
    ]
    original = [96, 7, 1, 11, 29, 0, 65]
    swapped = [83, 21, 4, 21, 39, 0, 77]
    x = np.arange(len(labels))
    width = 0.36

    fig, axis = plt.subplots(figsize=(11.2, 4.4))
    fig.suptitle(
        "E2 — Bóc tách chuỗi nhân quả của V3 trên 100 đề",
        fontsize=14,
        fontweight="bold",
    )
    axis.bar(
        x - width / 2,
        original,
        width,
        color=BLUE,
        label="Điều kiện gốc",
    )
    axis.bar(
        x + width / 2,
        swapped,
        width,
        color=ORANGE,
        label="Hoán đổi riêng moral",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Tỷ lệ đạt (%)")
    axis.legend(ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Moral swap làm thay đổi cấu trúc ở 73% trường hợp, nhưng cả hai tập đều "
        "0% plot-entails-moral: mô hình nhạy với token moral nhưng chưa điều khiển "
        "được chuỗi sự kiện.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    finish(fig, "e2_v3_failure_chain.png")


def plot_e2_v10_causal_replay() -> None:
    steps = np.array([100, 200, 300, 400, 500, 600, 700, 800, 830])
    causal_loss = np.array(
        [3.404, 3.204, 3.083, 3.028, 2.994, 2.977, 2.970, 2.968, 2.968]
    )
    replay_loss = np.array(
        [1.5311, 1.5301, 1.5304, 1.5301, 1.5299, 1.5297, 1.5295, 1.5296, 1.5295]
    )
    labels = ["V3", "V10 · bước 300"]
    x = np.arange(2)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    fig.suptitle(
        "E2 — V10: học trên dữ liệu nhân quả có replay",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].plot(steps, causal_loss, color=BLUE, linewidth=2, label="Causal validation")
    axes[0].plot(steps, replay_loss, color=GRAY, linewidth=2, label="Replay validation")
    axes[0].axvline(300, color=ORANGE, linestyle="--", linewidth=1.2)
    axes[0].annotate(
        "checkpoint được chọn",
        (300, causal_loss[2]),
        xytext=(22, 22),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": ORANGE},
    )
    axes[0].set(title="Loss validation theo bước", xlabel="Bước", ylabel="Loss")
    axes[0].legend(fontsize=8)

    width = 0.25
    for offset, (name, values, color) in enumerate(
        [
            ("Fluency", [5.21, 5.08], BLUE),
            ("Moral delivery", [2.12, 2.13], ORANGE),
            ("Primary", [3.67, 3.60], GRAY),
        ]
    ):
        axes[1].bar(
            x + (offset - 1) * width,
            values,
            width,
            color=color,
            label=name,
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 6)
    axes[1].set(title="Đánh giá mù trên 100 đề", ylabel="Điểm /10")
    axes[1].legend(fontsize=8)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Causal loss giảm 3,404→2,968 nhưng causal pass của V3 và V10 cùng 2%; "
        "loss validation không dự báo được khả năng điều khiển diễn biến.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    finish(fig, "e2_v10_causal_replay.png")


def plot_e2_v11_class_token() -> None:
    labels = ["V3", "V11 có thẻ lớp", "V11 bỏ thẻ lớp"]
    x = np.arange(3)
    colors = [GRAY, ORANGE, BLUE]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2))
    fig.suptitle(
        "E2 — V11: huấn luyện và bóc tách thẻ lớp bài học",
        fontsize=14,
        fontweight="bold",
    )
    steps = [150, 300, 450, 600, 750, 900, 1011]
    causal_loss = [2.908, 2.749, 2.674, 2.636, 2.616, 2.610, 2.610]
    replay_loss = [1.5502, 1.5473, 1.5471, 1.5468, 1.5471, 1.5471, 1.5471]
    axes[0].plot(steps, causal_loss, color=BLUE, marker="o", label="Causal")
    axes[0].plot(steps, replay_loss, color=GRAY, marker="o", label="Replay")
    axes[0].axvline(600, color=ORANGE, linestyle="--", linewidth=1)
    axes[0].set(
        title="Validation loss",
        xlabel="Bước",
        ylabel="Loss",
        ylim=(1.4, 3.1),
    )
    axes[0].legend(fontsize=7)

    words = [252.15, 90.8, 247.8]
    axes[1].bar(x, words, color=colors)
    axes[1].set_xticks(x, labels)
    axes[1].set(title="Độ dài thân truyện", ylabel="Số từ trung bình", ylim=(0, 280))
    for index, value in enumerate(words):
        axes[1].text(index, value + 6, f"{value:.1f}", ha="center", fontsize=8)

    width = 0.24
    series = [
        ("Fluency", [5.30, 4.05, 5.05], BLUE),
        ("Moral delivery", [2.20, 1.60, 2.25], ORANGE),
        ("Primary", [3.75, 2.83, 3.65], GRAY),
    ]
    for offset, (name, values, color) in enumerate(series):
        axes[2].bar(
            x + (offset - 1) * width,
            values,
            width,
            color=color,
            label=name,
        )
    axes[2].set_xticks(x, labels)
    axes[2].set(title="Đánh giá trên cùng 20 đề", ylabel="Điểm /10", ylim=(0, 6))
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Cùng checkpoint bước 600: bỏ <moral_class> khôi phục độ dài và chất lượng "
        "gần V3; sai lệch đến từ token điều khiển, không phải toàn bộ trọng số.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    finish(fig, "e2_v11_class_token.png")


def plot_e2_v12_v13() -> None:
    metrics = ["Fluency", "Moral delivery", "Primary"]
    x = np.arange(3)
    v12 = np.array([0.12, 0.04, 0.08])
    v12_low = np.array([-0.08, -0.10, -0.065])
    v12_high = np.array([0.32, 0.18, 0.225])
    v13 = np.array([0.11, 0.01, 0.06])
    v13_low = np.array([0.02, -0.05, -0.005])
    v13_high = np.array([0.21, 0.08, 0.125])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    fig.suptitle(
        "E2 — Bóc tách tăng quy mô và DPO",
        fontsize=14,
        fontweight="bold",
    )
    offset = 0.11
    axes[0].errorbar(
        x - offset,
        v12,
        yerr=np.vstack([v12 - v12_low, v12_high - v12]),
        fmt="o",
        color=BLUE,
        capsize=4,
        label="V12: 98M − 63M",
    )
    axes[0].errorbar(
        x + offset,
        v13,
        yerr=np.vstack([v13 - v13_low, v13_high - v13]),
        fmt="D",
        color=ORANGE,
        capsize=4,
        label="V13: DPO − 98M",
    )
    axes[0].axhline(0, color=BLACK, linewidth=0.9)
    axes[0].set_xticks(x, metrics)
    axes[0].set(
        title="Chênh lệch bắt cặp và CI bootstrap 95%",
        ylabel="Chênh điểm /10",
        ylim=(-0.16, 0.38),
    )
    axes[0].legend(fontsize=8)

    causal_delta = [1.0, 0.0]
    bars = axes[1].bar(
        ["V12\n98M − 63M", "V13\nDPO − 98M"],
        causal_delta,
        color=[BLUE, ORANGE],
        width=0.55,
    )
    axes[1].errorbar(
        [0, 1],
        causal_delta,
        yerr=[[1.0, 0.0], [2.0, 0.0]],
        fmt="none",
        ecolor=BLACK,
        capsize=4,
    )
    axes[1].axhline(0, color=BLACK, linewidth=0.9)
    axes[1].set(
        title="Thay đổi causal pass",
        ylabel="Điểm phần trăm",
        ylim=(-1, 4),
    )
    axes[1].text(
        1,
        2.8,
        "Preference accuracy DPO: 54,4%\n(mức ngẫu nhiên: 50%)",
        ha="center",
        fontsize=8.5,
    )
    for bar, value in zip(bars, causal_delta):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.0f}",
            ha="center",
            fontsize=8,
        )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Tăng 63M→98M và DPO đều không tạo cải thiện causal pass đáng tin cậy "
        "trên 100 đề bắt cặp.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    finish(fig, "e2_v12_v13_ablation.png")


def plot_e2_v14_epochs() -> None:
    labels = ["V3", "E1", "E2", "E3", "E5", "E8"]
    fluency = [6.30, 4.75, 4.65, 4.90, 4.90, 4.80]
    moral = [2.40, 2.20, 1.90, 2.30, 2.65, 2.70]
    causal = [5, 5, 0, 0, 5, 5]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    fig.suptitle(
        "E2 — V14: tám epoch chỉ dùng dữ liệu nhân quả",
        fontsize=14,
        fontweight="bold",
    )
    width = 0.36
    axes[0].bar(x - width / 2, fluency, width, color=BLUE, label="Fluency")
    axes[0].bar(x + width / 2, moral, width, color=ORANGE, label="Moral delivery")
    axes[0].set_xticks(x, labels)
    axes[0].set(title="Chất lượng theo checkpoint", ylabel="Điểm /10", ylim=(0, 7))
    axes[0].legend(fontsize=8)
    axes[1].bar(x, causal, color=[GRAY] + [BLUE] * 5, width=0.58)
    axes[1].axhline(
        10,
        color="#B42318",
        linestyle="--",
        linewidth=1.2,
        label="Cổng 10%",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set(title="Causal pass", ylabel="Tỷ lệ (%)", ylim=(0, 12))
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Tám epoch giảm fluency khoảng 1,4–1,7 điểm; causal pass dao động 0–5% "
        "và không vượt baseline.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    finish(fig, "e2_v14_causal_epochs.png")


def plot_e2_v15_transfer() -> None:
    epochs = np.arange(1, 4)
    match_accuracy = [73.18, 74.56, 75.44]
    labels = ["V3", "E1", "E2", "E3"]
    fluency = [6.20, 4.85, 4.65, 4.60]
    causal = [5, 0, 0, 5]
    x = np.arange(4)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9))
    fig.suptitle(
        "E2 — V15: tín hiệu biểu diễn và khả năng chuyển sang sinh",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].plot(epochs, match_accuracy, color=BLUE, marker="o", linewidth=2)
    axes[0].axhline(50, color=GRAY, linestyle="--", linewidth=1, label="Ngẫu nhiên")
    axes[0].axhline(70, color=ORANGE, linestyle="--", linewidth=1, label="Cổng 70%")
    axes[0].set_xticks(epochs)
    axes[0].set(
        title="Story–moral matching held-out",
        xlabel="Epoch",
        ylabel="Accuracy (%)",
        ylim=(45, 80),
    )
    axes[0].legend(fontsize=7)

    axes[1].bar(x, fluency, color=[GRAY] + [BLUE] * 3, width=0.58)
    axes[1].set_xticks(x, labels)
    axes[1].set(title="Fluency khi sinh tự do", ylabel="Điểm /10", ylim=(0, 7))

    axes[2].bar(x, causal, color=[GRAY] + [ORANGE] * 3, width=0.58)
    axes[2].axhline(10, color="#B42318", linestyle="--", linewidth=1)
    axes[2].set_xticks(x, labels)
    axes[2].set(title="Causal pass", ylabel="Tỷ lệ (%)", ylim=(0, 12))

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Đầu phân loại học được quan hệ story–moral trên held-out, nhưng tín hiệu đó "
        "không chuyển thành điều khiển diễn biến khi sinh từng token.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    finish(fig, "e2_v15_representation_transfer.png")


def plot_e4_human_eval() -> None:
    systems = [
        "Base FP16",
        "Strict prompt",
        "Strict + post.",
        "Base + repair",
        "Fluency SFT",
    ]
    metrics = {
        "Trôi chảy": [3.40, 3.70, 3.70, 3.40, 3.00],
        "Bám đề": [4.00, 4.10, 4.10, 4.30, 3.70],
        "Cấu trúc": [4.00, 3.50, 3.60, 4.10, 4.00],
        "Bài học": [3.70, 3.80, 5.00, 5.00, 3.70],
        "An toàn": [4.80, 4.60, 4.60, 4.80, 4.80],
    }
    overall = [3.98, 3.94, 4.20, 4.32, 3.84]
    colors = [BLUE, GREEN, ORANGE, "#D62828", "#7C3AED"]
    x = np.arange(len(systems))
    width = 0.14

    fig, axis = plt.subplots(figsize=(11.2, 4.8))
    fig.suptitle(
        "E4 — Đánh giá thủ công năm cấu hình cuối",
        fontsize=14,
        fontweight="bold",
    )
    for index, (label, values) in enumerate(metrics.items()):
        axis.bar(
            x + (index - 2) * width,
            values,
            width,
            label=label,
            color=colors[index],
        )
    axis.plot(
        x,
        overall,
        color=BLACK,
        marker="D",
        linewidth=1.5,
        markersize=5,
        label="Tổng",
        zorder=4,
    )
    for index, value in enumerate(overall):
        axis.text(
            index,
            value + 0.11,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
    axis.set_xticks(x, systems, rotation=12, ha="right")
    axis.set_ylim(0, 5.35)
    axis.set_ylabel("Điểm /5")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=6, loc="upper center", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Nguồn: bảng human evaluation E4 đã lưu, 10 đề bài; "
        "Base + Repair có tổng điểm cao nhất 4,32/5.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    finish(fig, "e4_human_eval.png")


def plot_e3_ablation() -> None:
    labels = ["Base\n0M", "A: q,v all-30\n0,92M", "B: q,v last-10\n0,31M", "C: all-linear\n4,88M"]
    ppl = [9.52, 4.82, 5.46, 3.84]
    judge = [5.73, 6.70, 5.94, 6.87]
    colors = [GRAY, LIGHT_BLUE, ORANGE, BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    fig.suptitle(
        "E3 — Bóc tách vị trí LoRA trên SmolLM2-135M",
        fontsize=14,
        fontweight="bold",
    )
    x = np.arange(len(labels))
    axes[0].bar(x, ppl, color=colors)
    axes[0].set_xticks(x, labels)
    axes[0].set(title="Validation perplexity (thấp hơn tốt hơn)", ylabel="PPL")
    axes[0].set_ylim(0, 10.5)
    axes[1].bar(x, judge, color=colors)
    axes[1].set_xticks(x, labels)
    axes[1].set(
        title="Điểm LLM-as-judge nội bộ (cao hơn tốt hơn)", ylabel="Điểm /10"
    )
    axes[1].set_ylim(0, 8)
    for axis, values in zip(axes, [ppl, judge]):
        axis.spines[["top", "right"]].set_visible(False)
        for index, value in enumerate(values):
            axis.text(
                index,
                value + 0.14,
                f"{value:.2f}",
                ha="center",
                fontsize=9,
                fontweight="bold" if index == 3 else "normal",
            )
    fig.text(
        0.5,
        0.01,
        "Nguồn: bảng kết quả E3 đã lưu; cùng tập held-out, cấu hình sinh và judge nội bộ. "
        "Không có trainer trace để dựng loss curve.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    finish(fig, "e3_lora_ablation.png")


def plot_e5_training() -> None:
    state = load_json(
        ROOT / "hoangndl" / "outputs" / "checkpoint-339" / "trainer_state.json"
    )
    history = state["log_history"]
    train = [row for row in history if "loss" in row]
    evaluation = [row for row in history if "eval_loss" in row]
    steps = np.array([row["step"] for row in train])
    loss = np.array([row["loss"] for row in train])
    lr = np.array([row["learning_rate"] for row in train])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    fig.suptitle(
        "E5 — QLoRA Llama 3.2 3B, run ba epoch",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].plot(steps, loss, color=BLUE, linewidth=1.8, label="Train loss")
    axes[0].scatter(
        [row["step"] for row in evaluation],
        [row["eval_loss"] for row in evaluation],
        color=ORANGE,
        marker="D",
        s=42,
        label="Eval loss đã lưu",
        zorder=3,
    )
    axes[0].set(title="Loss theo bước", xlabel="Bước", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(steps, lr, color=ORANGE, linewidth=1.8)
    axes[1].set(
        title="Learning rate theo bước", xlabel="Bước", ylabel="Learning rate"
    )
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    for axis in axes:
        axis.axvline(113, color=GRAY, linestyle="--", linewidth=0.9)
        axis.axvline(226, color=GRAY, linestyle="--", linewidth=0.9)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].annotate(
        "Eval 0,488",
        (113, evaluation[0]["eval_loss"]),
        xytext=(-36, 24),
        textcoords="offset points",
        fontsize=8,
    )
    axes[0].annotate(
        "Eval 0,453",
        (226, evaluation[1]["eval_loss"]),
        xytext=(-22, 26),
        textcoords="offset points",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.01,
        "Nguồn: checkpoint-339/trainer_state.json (67 điểm train, eval tại bước 113 và 226). "
        "Artifact không lưu eval tại bước 339.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    finish(fig, "e5_training_curves.png")


def main() -> None:
    configure_style()
    plot_e2_training()
    plot_e2_screen()
    plot_e2_tokenizer_metrics()
    plot_e2_v3_failure_chain()
    plot_e2_v10_causal_replay()
    plot_e2_v11_class_token()
    plot_e2_v12_v13()
    plot_e2_v14_epochs()
    plot_e2_v15_transfer()
    plot_e4_human_eval()
    plot_e3_ablation()
    plot_e5_training()
    for path in sorted(OUT.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()

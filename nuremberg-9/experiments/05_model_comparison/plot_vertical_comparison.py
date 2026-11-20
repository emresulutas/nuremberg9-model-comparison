"""Render vertical comparison figures from the saved Nuremberg-9 score report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MODELS = ("churro", "resnet18", "resnet50", "swin_tiny")
NAMES = ("CHURRO", "ResNet18", "ResNet50", "Swin-Tiny")
COLORS = ("#5069a0", "#087568", "#6a8392", "#ad654a")
METRICS = ("macro_f1", "balanced_accuracy", "accuracy")
METRIC_NAMES = ("Macro-F1", "Balanced accuracy", "Accuracy")
INK = "#182f3a"
MUTED = "#536f7a"
PAPER = "#f1f6f5"
GRID = "#d5e3e4"


def heading(fig, title, subtitle, legend=True, models=MODELS):
    fig.patch.set_facecolor(PAPER)
    fig.text(0.08, 0.947, title, fontsize=26, weight="bold", color=INK)
    fig.text(0.08, 0.910, subtitle, fontsize=10.5, color=MUTED)
    if legend:
        handles = [Patch(facecolor=color, label=name)
                   for key, name, color in zip(MODELS, NAMES, COLORS, strict=True) if key in models]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.077, 0.892),
                   frameon=False, ncol=len(handles), labelcolor=INK, fontsize=10)


def axes_style(ax):
    ax.set_facecolor(PAPER)
    ax.spines[:].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=9)


def percentage_axis(ax, ticks):
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))


def panel_title(ax, title, detail):
    ax.text(0, 1.10, title, transform=ax.transAxes, fontsize=15, weight="bold", color=INK)
    ax.text(0, 1.042, detail, transform=ax.transAxes, fontsize=9.5, color=MUTED)


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, facecolor=PAPER)
    plt.close(fig)


def final_summary(scores, output):
    order = sorted(MODELS, key=lambda model: scores["models"][model]["macro_f1"], reverse=True)
    names = dict(zip(MODELS, NAMES, strict=True))
    heights = [100 * scores["models"][model]["macro_f1"] for model in order]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=PAPER)
    fig.text(.08, .867, f"{names[order[0]]} leads Nuremberg-9", fontsize=25, weight="bold", color=INK)
    fig.text(.08, .79,
             f"Macro-F1  ·  Full test set  ·  {scores['n_test_lines']:,} lines, {len(scores['scribes'])} writers",
             fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=.115, right=.95, top=.718, bottom=.14)
    axes_style(ax)
    ax.set_ylim(0, 108)
    percentage_axis(ax, range(0, 101, 25))
    ax.set_xticks(range(len(order)), [names[model] for model in order])
    ax.tick_params(axis="x", pad=12, labelcolor=INK, labelsize=11)
    colors = [COLORS[1] if i == 0 else COLORS[0] if model == "churro" else "#829ba5"
              for i, model in enumerate(order)]
    bars = ax.bar(range(len(order)), heights, width=.59, color=colors)
    labels = ax.bar_label(bars, labels=[f"{score:.2f}%" for score in heights], padding=6,
                          color=INK, fontsize=13)
    labels[0].set_weight("bold")
    labels[0].set_color("#075e54")
    save(fig, output, "00-final-sonuc")


def model_scores(scores, output):
    fig, axs = plt.subplots(2, 1, figsize=(11, 11.3), sharex=True, sharey=True)
    heading(fig, "Writer identification scores", "Nuremberg-9 Letterbooks  ·  One prediction per handwritten line")
    fig.subplots_adjust(left=.10, right=.95, top=.780, bottom=.095, hspace=.54)
    center = np.arange(len(METRICS)) * 1.55
    width = .275
    for ax, section, title, count in zip(
        axs, (scores, scores["unseen_pages"]), ("Full test set", "Unseen pages"),
        (len(scores["scribes"]), len(scores["unseen_pages"]["scribes_with_test_lines"])), strict=True
    ):
        axes_style(ax)
        panel_title(ax, title, f"{section['n_test_lines']:,} test lines  ·  {count} writers")
        for i, (key, color) in enumerate(zip(MODELS, COLORS, strict=True)):
            heights = [100 * section["models"][key][m] for m in METRICS]
            bars = ax.bar(center + (i - 1.5) * width, heights, width=width * .83, color=color)
            ax.bar_label(bars, labels=[f"{v:.2f}%" for v in heights], padding=4,
                         fontsize=8.5, color=INK, weight="semibold")
        ax.set_ylim(0, 110)
        ax.set_xticks(center, METRIC_NAMES)
        ax.tick_params(axis="x", labelbottom=True, labelcolor=INK, pad=9)
        percentage_axis(ax, range(0, 101, 25))
    fig.text(.10, .039, "Unseen pages contributed no lines to the training or validation split.",
             fontsize=9, color=MUTED)
    save(fig, output, "01-model-scores-vertical")


def difference(scores, output):
    compared = MODELS[1:]
    fig, axs = plt.subplots(2, 1, figsize=(10.8, 10.4), sharey=True)
    heading(fig, "Score differences from CHURRO",
            "Percentage points  ·  Zero marks the CHURRO score on the same test subset", models=compared)
    fig.subplots_adjust(left=.10, right=.95, top=.777, bottom=.095, hspace=.53)
    center = np.arange(len(METRICS)) * 1.46
    width = .29
    values = [v for section in (scores, scores["unseen_pages"])
              for model in compared for v in section["differences_from_churro_pp"][model].values()]
    low = min(-1, np.floor(min(values)) - .8)
    high = max(1, np.ceil(max(values)) + .8)
    for ax, section, title, count in zip(
        axs, (scores, scores["unseen_pages"]), ("Full test set", "Unseen pages"),
        (len(scores["scribes"]), len(scores["unseen_pages"]["scribes_with_test_lines"])), strict=True
    ):
        axes_style(ax)
        panel_title(ax, title, f"{section['n_test_lines']:,} test lines  ·  {count} writers")
        ax.axhline(0, linewidth=1, color=INK)
        for i, (key, color) in enumerate(zip(compared, COLORS[1:], strict=True)):
            heights = [section["differences_from_churro_pp"][key][m] for m in METRICS]
            bars = ax.bar(center + (i - 1) * width, heights, width=width * .81, color=color)
            ax.bar_label(bars, labels=[f"{v:+.2f}" for v in heights], padding=4,
                         fontsize=9, color=INK, weight="semibold")
        ax.set_ylim(low, high)
        ax.set_xticks(center, METRIC_NAMES)
        ax.tick_params(axis="x", labelcolor=INK, pad=8)
        ax.set_ylabel("Percentage points", color=MUTED, fontsize=9)
    fig.text(.10, .040, "The differences describe these runs on their shared test splits.",
             fontsize=9, color=MUTED)
    save(fig, output, "02-churro-differences-vertical")


def writer_recall(scores, output):
    fig, axs = plt.subplots(3, 3, figsize=(11.4, 12), sharey=True)
    heading(fig, "Writer-by-writer recall", "Full test set  ·  Share of each writer’s lines correctly identified")
    fig.subplots_adjust(left=.085, right=.955, top=.790, bottom=.102, hspace=.49, wspace=.31)
    baseline = scores["models"]["churro"]
    shared = set(scores["line_split_scribes"])
    for i, (ax, scribe) in enumerate(zip(axs.flat, scores["scribes"], strict=True)):
        axes_style(ax)
        panel_title(ax, f"S{scribe}{'*' if scribe in shared else ''}",
                    f"{baseline['support_per_scribe'][i]:,} test lines")
        heights = [100 * scores["models"][key]["recall_per_scribe"][i] for key in MODELS]
        bars = ax.bar(range(4), heights, width=.57, color=COLORS)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in heights], padding=3, fontsize=8, color=INK)
        ax.set_ylim(0, 112)
        ax.set_xticks([])
        percentage_axis(ax, (0, 50, 100))
    fig.text(.085, .055, "* S11 and S12 share pages with training or validation. Their test samples are small.",
             fontsize=9, color=MUTED)
    save(fig, output, "03-writer-recall-vertical")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    comparison = ROOT / "reports/comparison"
    parser.add_argument("--scores", type=Path, default=comparison / "metrics/scores.json")
    parser.add_argument("--output-dir", type=Path, default=comparison)
    args = parser.parse_args()
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    final_summary(scores, args.output_dir)
    model_scores(scores, args.output_dir)
    difference(scores, args.output_dir)
    writer_recall(scores, args.output_dir)


if __name__ == "__main__":
    main()

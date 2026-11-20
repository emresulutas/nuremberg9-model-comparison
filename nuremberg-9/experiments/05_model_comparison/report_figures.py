"""Paper figures for the four-model Nuremberg Letterbooks comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402

MODEL_NAMES = {"churro": "CHURRO", "resnet18": "ResNet18", "resnet50": "ResNet50", "swin_tiny": "Swin-Tiny"}
COLORS = ("#315583", "#137A75", "#A5512B", "#75548F")
MARKERS = ("o", "s", "^", "D")
INK = "#172635"
GRID = "#DCE4EB"


def save(fig, out: Path) -> None:
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def score_table(scope: dict, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 0.38 * len(scope["models"]) + 0.82))
    ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=10)
    ax.axis("off")
    cells = []
    for name, scores in scope["models"].items():
        change = scope["differences_from_churro_pp"].get(name)
        delta = "Ref." if change is None else f"{change['macro_f1']:+.2f}"
        cells.append([MODEL_NAMES.get(name, name), f"{scores['macro_f1']:.2%}", delta,
                      f"{scores['balanced_accuracy']:.2%}", f"{scores['accuracy']:.2%}"])
    table = ax.table(cellText=cells, cellLoc="center", colLoc="center", bbox=(0, 0, 1, 1),
                     colLabels=("Model", "Macro-F1", "Macro-F1 Δ\nvs CHURRO (pp)", "Balanced\naccuracy", "Accuracy"),
                     colWidths=(0.21, 0.17, 0.23, 0.22, 0.17))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#EAF0F5" if row == 0 else ("#F6F8FB" if row % 2 else "white"))
        cell.get_text().set_color(INK)
        if column == 0:
            cell.get_text().set_ha("left")
        if row == 0:
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(8.5)
    save(fig, out)


def recall_plot(report: dict, out: Path) -> None:
    models = report["models"]
    scribes = report["scribes"]
    support = models["churro"]["support_per_scribe"]
    shared = set(report["line_split_scribes"])
    names = [f"S{s}{'*' if s in shared else ''}  (n={n:,})" for s, n in zip(scribes, support, strict=True)]
    fig, ax = plt.subplots(figsize=(7.4, 4.05))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.24)
    y = np.arange(len(scribes))
    offsets = np.linspace(-0.31, 0.31, len(models)) if len(models) > 1 else [0]
    for i, ((model, scores), offset) in enumerate(zip(models.items(), offsets, strict=True)):
        ax.scatter(scores["recall_per_scribe"], y + offset, color=COLORS[i % len(COLORS)],
                   marker=MARKERS[i % len(MARKERS)], s=17, label=MODEL_NAMES.get(model, model), zorder=3)
    ax.set(yticks=y, yticklabels=names, ylim=(len(y) - 0.45, -0.55), xlim=(-0.025, 1.025), xlabel="Test recall",
           xticks=np.linspace(0, 1, 5))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="both", color=GRID, linewidth=0.7)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Test recall by writer", loc="left", fontsize=11, weight="bold", pad=11)
    fig.legend(ncol=min(len(models), 4), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.065))
    if shared:
        fig.text(0.5, 0.015, "* Some pages also contributed training or validation lines.",
                 fontsize=8, ha="center", color="#516577")
    save(fig, out)


def confusion_plot(report: dict, out: Path) -> None:
    models = report["models"]
    names = [f"S{s}" for s in report["scribes"]]
    columns = min(2, len(models))
    rows = (len(models) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.95 * columns + 0.45, 3.55 * rows),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    used_axes = []
    for ax, (name, scores) in zip(axes.flat, models.items()):
        counts = np.asarray(scores["confusion_matrix"], dtype=np.int64)
        percentages = counts / counts.sum(axis=1, keepdims=True) * 100
        heatmap = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
        for (row, column), value in np.ndenumerate(percentages):
            rounded = round(value)
            if rounded:
                ax.text(column, row, f"{rounded:d}", ha="center", va="center", fontsize=7,
                        color="white" if value >= 60 else INK)
        ticks = np.arange(len(names))
        ax.set(xticks=ticks, xticklabels=names, yticks=ticks, yticklabels=names)
        ax.tick_params(length=0, labelsize=7.5)
        ax.spines[:].set_visible(False)
        ax.set_title(MODEL_NAMES.get(name, name), fontsize=10, weight="bold", loc="left")
        used_axes.append(ax)
    for ax in list(axes.flat)[len(used_axes):]:
        ax.remove()
    colorbar = fig.colorbar(heatmap, ax=used_axes, shrink=0.80, ticks=[0, 25, 50, 75, 100], pad=0.035)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    colorbar.set_label("Share of true writer's lines", fontsize=8)
    fig.supxlabel("Predicted writer", fontsize=9)
    fig.supylabel("True writer", fontsize=9)
    save(fig, out)


def render_figures(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK}):
        score_table(report, f"Full test set  ·  {report['n_test_lines']:,} lines, {len(report['scribes'])} writers",
                    out / "test_scores.png")
        subset = report["unseen_pages"]
        score_table(subset,
                    f"Unseen test pages  ·  {subset['n_test_lines']:,} lines, {len(subset['scribes_with_test_lines'])} writers",
                    out / "test_scores_unseen_pages.png")
        recall_plot(report, out / "recall_per_scribe.png")
        confusion_plot(report, out / "confusion.png")

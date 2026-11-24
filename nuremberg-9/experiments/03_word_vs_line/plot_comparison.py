"""Render the CHURRO-3B comparison from saved metrics without training."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


FIGURE_NAME = "churro-3b-word-vs-line.png"


def plot_comparison(metrics, out: Path) -> None:
    variants = (
        ("word_level", "Single word", "#547280"),
        ("line_vote", "Word votes per line", "#087c70"),
        ("line_head_same_lines", "Whole line (Balanced Probe)", "#405a9b"),
    )
    words = sum(metrics["word_level"]["support"])
    lines = sum(metrics["line_vote"]["support"])
    seed = metrics["run"]["config"]["seed"]
    fig, (scores, recall) = plt.subplots(2, 1, figsize=(11, 7))
    fig.subplots_adjust(top=.73, bottom=.15, left=.08, right=.98, hspace=.47)
    fig.text(.08, .95, "CHURRO-3B · Word vs Line", fontsize=23, weight="bold")
    fig.text(.08, .90, "Frozen visual encoder, block 16 · Two linear classifiers · Nuremberg-9 test",
             fontsize=11, color="#425668")
    fig.text(.08, .86, f"Seed {seed} · Single-word scores use {words:,} words"
             f" · Both line methods use the same {lines:,} lines", fontsize=10)

    width = .24
    score_keys = ("accuracy", "balanced_accuracy", "macro_f1")
    scribes = metrics["scribes"]
    for index, (key, label, color) in enumerate(variants):
        values = metrics[key]
        offset = (index - 1) * width
        bars = scores.bar(np.arange(3) + offset, [values[k] for k in score_keys],
                          width, color=color, label=label)
        scores.bar_label(bars, labels=[f"{values[k]:.1%}" for k in score_keys], padding=3, fontsize=9)
        recall.bar(np.arange(len(scribes)) + offset, values["recall_per_scribe"], width, color=color)

    scores.set(xticks=range(3), xticklabels=("Accuracy", "Balanced accuracy", "Macro-F1"))
    fig.legend(*scores.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.07, .83),
               ncol=3, frameon=False)
    scores.set_title("Overall test scores", loc="left", weight="bold")
    recall.set(xticks=range(len(scribes)), xticklabels=[f"S{s}" for s in scribes])
    recall.set_title("Recall by writer", loc="left", weight="bold")
    for ax in (scores, recall):
        ax.set(ylim=(0, 1.12), yticks=np.arange(0, 1.01, .25))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)

    fig.text(.08, .074, "The same word classifier is used for single-word predictions and word voting."
             " Word voting averages the words’ log-probabilities.", fontsize=9)
    fig.text(.08, .042, "The whole-line classifier takes the entire line as input."
             " S10, S11 and S12 have only 32, 14 and 6 test lines.", fontsize=9)
    fig.savefig(out / FIGURE_NAME, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    for path in sorted(Path(__file__).resolve().parent.glob("results/words*/metrics.json")):
        plot_comparison(json.loads(path.read_text(encoding="utf-8")), path.parent)

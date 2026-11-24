"""Analyze saved baseline predictions using the CHURRO whole-line and word protocols."""
from __future__ import annotations

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import binomtest

EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP / "05_model_comparison"))
from compare import scores
from data import SCRIBES

LINE_SEEDS = (42, 43, 44, 45, 46)
WORD_SEEDS = (42, 43, 44)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
LABELS = {"resnet18": "ResNet18", "resnet50": "ResNet50", "swin_tiny": "Swin-Tiny"}


def load(folder, split):
    with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
        page, line, truth, prob = (archive[f"{split}_{field}"] for field in ("page_id", "line_id", "y", "prob"))
        log_key = f"{split}_log_prob"
        logp = archive[log_key] if log_key in archive else np.log(np.clip(prob, 1e-9, 1))
    return list(zip(page.tolist(), line.tolist(), strict=True)), truth, logp


def votes(data):
    keys, truth, logp = data
    groups = {}
    for row, key in enumerate(keys):
        groups.setdefault(key, []).append(row)
    labels, means = [], []
    for members in groups.values():
        y = truth[members]
        if np.any(y != y[0]):
            raise ValueError("Conflicting word labels on a line")
        labels.append(y[0])
        means.append(logp[members].mean(0))
    return list(groups), np.asarray(labels), np.stack(means)


def select(data, keys, truth):
    saved_keys, saved_truth, pred = data
    index = {key: row for row, key in enumerate(saved_keys)}
    if len(index) != len(saved_keys):
        raise ValueError("Duplicate line identifiers")
    rows = [index[key] for key in keys]
    if not np.array_equal(saved_truth[rows], truth):
        raise ValueError("Word and whole-line labels differ")
    return pred[rows]


def metric(y, arrays):
    return {k: v for k, v in scores(y, arrays, list(SCRIBES)).items() if k in METRICS}


def average(rows):
    return {key: {"mean": float(np.mean([row[key] for row in rows])),
                  "sd": float(np.std([row[key] for row in rows], ddof=1))} for key in METRICS}


def chart(out, name, title, labels, categories, values, metric_names=None):
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.55), 4.4), layout="constrained")
    x = np.arange(len(labels))
    width = .72 / len(categories)
    for i, (category, vals, color) in enumerate(zip(categories, values, ("#547280", "#087c70", "#465e9b"), strict=False)):
        bars = ax.bar(x + (i - (len(categories) - 1) / 2) * width, vals, width, color=color, label=category)
        ax.bar_label(bars, labels=[f"{v:.1%}" if metric_names else str(v) for v in vals], padding=3, fontsize=9)
    ax.set(title=title, xticks=x, xticklabels=labels)
    ax.spines[["top", "right"]].set_visible(False)
    if metric_names:
        ax.set(ylim=(0, 1.15))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    else:
        ax.set(ylim=(0, max(max(v) for v in values) * 1.15 or 1), ylabel="Test lines")
    if len(categories) > 1:
        ax.legend(frameon=False, ncol=len(categories), loc="lower right")
    fig.savefig(out / name, dpi=300)
    plt.close(fig)


def analyze(model, baseline, out):
    name = LABELS[model]
    line = {seed: baseline if seed == 42 else out / f"line_seed_{seed}" for seed in LINE_SEEDS}
    word = {seed: out / f"word_seed_{seed}" for seed in WORD_SEEDS}
    aug = {seed: out / f"aug_seed_{seed}" for seed in LINE_SEEDS}
    report = {"model": model, "line_seeds": LINE_SEEDS, "word_seeds": WORD_SEEDS}
    report["augmentation"] = {
        group: average([metric(truth, pred) for _, truth, pred in (load(p, "test") for p in runs.values())])
        for group, runs in (("original", line), ("augmented", aug))
    }
    lp, wp, labels = {}, {}, {}
    for split in ("val", "test"):
        groups = {seed: votes(load(path, split)) for seed, path in word.items()}
        keys, truth, _ = groups[42]
        labels[split] = truth
        lp[split] = {seed: select(load(path, split), keys, truth) for seed, path in line.items()}
        wp[split] = {seed: select(group, keys, truth) for seed, group in groups.items()}

    y = labels["test"]
    yv = labels["val"]
    main_line, main_word = lp["test"][42], wp["test"][42]
    _, yt, raw_words = load(word[42], "test")
    report["word_vs_line"] = {
        "test_lines": len(y), "test_words": len(yt),
        "single_word": metric(yt, raw_words), "word_vote": metric(y, main_word),
        "whole_line": metric(y, main_line),
    }
    el, ew = main_line.argmax(1) != y, main_word.argmax(1) != y
    only_line, only_word, both = (int(m.sum()) for m in (el & ~ew, ew & ~el, el & ew))
    discordant = only_line + only_word
    report["error_overlap"] = {
        "only_whole_line_wrong": only_line, "only_word_vote_wrong": only_word, "both_wrong": both,
        "both_right": int((~(el | ew)).sum()),
        "mcnemar_exact_p": float(binomtest(only_line, discordant).pvalue) if discordant else 1.0,
    }
    for key, pairs in (
        ("line_vs_line", combinations(lp["test"].values(), 2)),
        ("word_vs_word", combinations(wp["test"].values(), 2)),
        ("word_vs_line", product(wp["test"].values(), lp["test"].values())),
    ):
        stats = []
        for left, right in pairs:
            a, b = left.argmax(1), right.argmax(1)
            wrong_a, wrong_b = a != y, b != y
            stats.append(((a != b).mean(), (wrong_a & wrong_b).sum() / max(1, (wrong_a | wrong_b).sum())))
        report["error_overlap"][key] = {
            "pairs": len(stats), "disagreement_mean": float(np.mean([r[0] for r in stats])),
            "error_jaccard_mean": float(np.mean([r[1] for r in stats])),
        }

    ensembles = {
        "whole_line": [metric(y, p) for p in lp["test"].values()],
        "word_vote": [metric(y, p) for p in wp["test"].values()],
        "line_plus_line_control": [metric(y, a + b) for a, b in combinations(lp["test"].values(), 2)],
        "word_plus_word_control": [metric(y, a + b) for a, b in combinations(wp["test"].values(), 2)],
        "half_and_half": [], "validation_weight": [],
    }
    weights = []
    for i, j in product(LINE_SEEDS, WORD_SEEDS):
        grid = np.linspace(0, 1, 21)
        def blend(split, weight):
            return (1 - weight) * lp[split][i] + weight * wp[split][j]

        weight = max(grid, key=lambda v: (round(metric(yv, blend("val", v))["balanced_accuracy"], 6), -abs(v - .5)))
        ensembles["half_and_half"].append(metric(y, blend("test", .5)))
        ensembles["validation_weight"].append(metric(y, blend("test", weight)))
        weights.append({"line_seed": i, "word_seed": j, "word_weight_from_validation": float(weight)})
    report["ensemble"] = {key: {"runs_or_pairs": len(rows), **average(rows)} for key, rows in ensembles.items()}
    report["ensemble"]["chosen_weights"] = weights

    metric_labels = ("Accuracy", "Balanced accuracy", "Macro-F1")
    chart(out, "augmentation.png", f"{name} · Train-line augmentation · Five matched seeds",
          metric_labels, ("Original", "Augmented"),
          [[report["augmentation"][group][key]["mean"] for key in METRICS] for group in ("original", "augmented")], True)
    chart(out, "word-vs-line.png", f"{name} · {len(yt):,} words or {len(y):,} shared lines · Seed 42",
          metric_labels, ("Single word", "Word votes per line", "Whole line"),
          [[report["word_vs_line"][group][key] for key in METRICS] for group in ("single_word", "word_vote", "whole_line")], True)
    chart(out, "error-overlap.png", f"{name} · Whole-line versus word-vote errors · {len(y):,} lines",
          ("Only whole line\nwrong", "Only word vote\nwrong", "Both wrong"), ("Errors",),
          [[only_line, only_word, both]])
    combo = ("whole_line", "word_vote", "half_and_half", "validation_weight")
    chart(out, "word-line-ensemble.png", f"{name} · Whole-line and word-vote ensemble · {len(y):,} lines",
          ("Whole line", "Word vote", "50/50 ensemble", "Validation-weighted\nensemble"),
          ("Macro-F1", "Balanced accuracy"),
          [[report["ensemble"][k][score]["mean"] for k in combo] for score in ("macro_f1", "balanced_accuracy")], True)
    temp = out / "details.json.tmp"
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(out / "details.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=LABELS)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.model, args.baseline, args.out)

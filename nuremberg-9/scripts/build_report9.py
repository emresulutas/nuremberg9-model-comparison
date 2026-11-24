"""Refresh the Nuremberg-9 paper report from saved scores and predictions."""
from __future__ import annotations

import csv
import json
from os.path import relpath
from pathlib import Path
from runpy import run_path
from shutil import copy2
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
N9 = ROOT / "nuremberg-9"
EXP = N9 / "experiments"
REPORT = N9 / "report"
FITS = EXP / "05_model_comparison/results/whole_line"
LOCAL_RESULTS = ROOT / "Nuremberg-9-Karsilastirma/Sonuclar"
REPEATS = EXP / "06_resnet18_vs_churro/results"
DETAILS = EXP / "07_model_details/results"
CHURRO = EXP / "01_balanced_probe/results/weighted_oversample"
LABELS = {"churro": "CHURRO-3B", "resnet18": "ResNet18", "resnet50": "ResNet50", "swin_tiny": "Swin-Tiny"}
COLORS = {"churro": "#405a9b", "resnet18": "#087c70", "resnet50": "#78919d", "swin_tiny": "#aa7655"}
sys.path.insert(0, str(EXP / "05_model_comparison"))
from line_diagnostics import render as render_line_diagnostics


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def link(page, label, target, image=False):
    prefix = "!" if image else ""
    return f"{prefix}[{label}]({Path(relpath(target, page.parent)).as_posix()})"


def publish(page, label, source, filename, image=False):
    page.parent.mkdir(parents=True, exist_ok=True)
    target = page.parent / filename
    copy2(source, target)
    return link(page, label, target, image=image)


def scores_csv(page, subsets, models):
    with (page.parent / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_subset", "test_lines", "model", "macro_f1", "balanced_accuracy", "accuracy"])
        for label, subset in subsets:
            for name in models:
                score = subset["models"][name]
                writer.writerow([label, subset["n_test_lines"], LABELS[name],
                                 score["macro_f1"], score["balanced_accuracy"], score["accuracy"]])


def write(page, text):
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(text.strip() + "\n", encoding="utf-8")


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def score_table(scores, models):
    rows = ["| Model | Macro-F1 | Balanced accuracy | Accuracy |", "|---|---:|---:|---:|"]
    for name in models:
        score = scores[name]
        rows.append(f"| {LABELS[name]} | {score['macro_f1']:.2%} | {score['balanced_accuracy']:.2%} | {score['accuracy']:.2%} |")
    return "\n".join(rows)


def model_figure(name, fit, page):
    scores = fit["test"]
    support = scores["support_per_scribe"]
    writers = [f"S{s}\n(n={n:,})" for s, n in zip(fit["scribes"], support)]
    history = fit["history"]
    fig, recall = plt.subplots(figsize=(9, 4), layout="constrained")
    recall.bar(writers, scores["recall_per_scribe"], color=COLORS[name])
    recall.set(title=f"{LABELS[name]} · Test recall by writer · {sum(support):,} whole lines", ylim=(0, 1.05))
    recall.yaxis.set_major_formatter(PercentFormatter(1))
    recall.tick_params(axis="x", labelsize=8)
    save(fig, page.parent / "test-recall.png")

    fig, (loss, val) = plt.subplots(1, 2, figsize=(10.4, 4), layout="constrained")
    fig.suptitle(f"{LABELS[name]} · Whole-line training curves · Nuremberg-9", weight="bold")
    val.plot([r["epoch"] for r in history], [r["val_macro_f1"] for r in history], color=COLORS[name])
    val.axvline(fit["best_epoch"], color="#425668", linestyle=":", label=f"Selected epoch {fit['best_epoch']}")
    val.set(title="Validation Macro-F1 by epoch", xlabel="Epoch", ylim=(0, 1.02))
    val.yaxis.set_major_formatter(PercentFormatter(1))
    val.legend(loc="lower right", frameon=False)
    loss.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], color=COLORS[name])
    loss.axvline(fit["best_epoch"], color="#425668", linestyle=":")
    loss.set(title="Training loss by epoch", xlabel="Epoch", ylabel="Training loss")
    save(fig, page.parent / "training-curves.png")

    counts = np.asarray(scores["confusion_matrix"])
    rates = counts / np.maximum(1, counts.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(6.8, 5.5), layout="constrained")
    heat = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for row, col in zip(*np.nonzero(counts)):
        ax.text(col, row, str(counts[row, col]), ha="center", va="center",
                color="white" if rates[row, col] > .5 else "#152737", fontsize=8)
    ticks = [f"S{s}" for s in fit["scribes"]]
    ax.set(title=f"{LABELS[name]} · Confusion matrix · {sum(support):,} whole test lines",
           xticks=range(len(ticks)), yticks=range(len(ticks)), xticklabels=ticks, yticklabels=ticks,
           xlabel="Predicted writer", ylabel="True writer")
    fig.colorbar(heat, ax=ax, format=PercentFormatter(1), label="Share of the true writer's lines")
    save(fig, page.parent / "test-confusion-matrix.png")


def write_models(results):
    models = REPORT / "01-models"
    write(models / "README.md", "# Individual models\n\n" + "\n".join(
        f"- [{label}]({name}/README.md)" for name, label in LABELS.items()))
    page = models / "churro/README.md"
    aug = EXP / "02_augmentation"
    word = EXP / "03_word_vs_line/results"
    overlap = EXP / "04_error_overlap/results"
    write(page, f"""
# CHURRO-3B

We froze the visual encoder and trained a linear classifier on whole-line features from block 16. This block was selected in the earlier seven-writer validation study. The nine-writer run used class weighting and oversampling to give rare writers more representation.

Square-root oversampling increased how often rare writers appeared during training. Effective-number weights were computed after sampling so that the two corrections would not both use the original class frequencies. Those two choices were made together. The nine-writer study did not estimate their separate effects.

## Experiments

| Experiment | Why we ran it | What it showed | Files |
|---|---|---|---|
| Balanced Probe | Could frozen CHURRO features distinguish nine writers with uneven class sizes? | The whole-line classifier reached 88.66% Macro-F1 on 7,636 test lines. | {publish(page, "Training curves", CHURRO / "blk16_curves.png", "training-curves.png")} · {publish(page, "Confusion matrix", CHURRO / "blk16_confusion.png", "test-confusion-matrix.png")} |
| Augmentation | Would additional copies of rare writers' training lines help? | Across five seeds, Macro-F1 rose from 88.2% to 89.2% and balanced accuracy fell from 90.5% to 90.0%. The reference run uses the original lines. | {publish(page, "Augmentation", aug / "results/compare.png", "augmentation.png")} |
| Word vs Line | Did word crops or whole lines give better writer predictions? | Word voting reached 90.7% Macro-F1 and whole lines 90.0% on the same 7,094 lines in the seed-42 run. | {publish(page, "Word vs Line", word / "words/churro-3b-word-vs-line.png", "word-vs-line.png")} |
| Error Overlap | Did the whole-line classifier and word voting fail on the same lines? | Only the whole-line classifier missed 127 lines, only word voting missed 88, and both missed 151. | {publish(page, "Error overlap", overlap / "error_sets.png", "error-overlap.png")} · {publish(page, "Word and line ensemble", overlap / "ensemble/compare_bars.png", "word-line-ensemble.png")} |

{publish(page, "CHURRO-3B whole-line test recall", CHURRO / "blk16_recall.png", "whole-line-recall.png", image=True)}
""")
    if results and (results / "churro/explanations/01_S1.png").is_file():
        with page.open("a", encoding="utf-8") as f:
            f.write("\n" + publish(page, "Grad-CAM and attention example", results / "churro/explanations/01_S1.png", "grad-cam-and-attention.png") + "\n")

    descriptions = {
        "resnet18": "A residual CNN with 18 layers. It tests how a compact ImageNet-initialized model does when all weights are adapted to the Nuremberg task.",
        "resnet50": "A residual CNN with 50 layers. It provides a deeper residual baseline under the same data split and training protocol.",
        "swin_tiny": "A hierarchical vision transformer with shifted windows. It provides a transformer baseline alongside the two residual networks.",
    }
    for name, intro in descriptions.items():
        page = models / name / "README.md"
        lines = [f"# {LABELS[name]}", intro,
                 "The model received a masked grayscale image of a whole line. Its weights were fine-tuned from ImageNet initialization. The checkpoint was selected by validation Macro-F1.",
                 "[Shared training and test conditions](../../02-model-comparison/README.md)"]
        if results and (fit_path := results / name / "fit/metrics.json").is_file():
            fit = read_json(fit_path)
            model_figure(name, fit, page)
            lines += [score_table({name: fit["test"]}, [name]),
                      "![Training loss and validation Macro-F1](training-curves.png)",
                      "[Full-test recall by writer](test-recall.png)",
                      "[Test confusion matrix](test-confusion-matrix.png)"]
        if results and (sample := results / name / "explanations/01_S1.png").is_file():
            method = "Grad-CAM and attention" if name == "swin_tiny" else "Grad-CAM"
            filename = "grad-cam-and-attention.png" if name == "swin_tiny" else "grad-cam.png"
            lines.append(publish(page, f"{method} example on a shared test line", sample, filename))
        if (detail_path := DETAILS / name / "details.json").is_file():
            detail = read_json(detail_path)
            aug = detail["augmentation"]
            word = detail["word_vs_line"]
            errors = detail["error_overlap"]
            ensemble = detail["ensemble"]["half_and_half"]
            lines += [
                "## Augmentation, words and line voting",
                "The follow-up reuses CHURRO-3B's word crops and training-only augmentation recipe. "
                "For this backbone the word and line variants are both fine-tuned from ImageNet weights.",
                "| Question | Result | Figure |\n|---|---|---|\n"
                f"| Do augmented training lines help? | Mean Macro-F1 over five seeds was {aug['original']['macro_f1']['mean']:.2%} "
                f"with original lines and {aug['augmented']['macro_f1']['mean']:.2%} with added copies. | "
                f"{publish(page, 'Augmentation', detail_path.parent / 'augmentation.png', 'augmentation.png')} |\n"
                f"| Does word voting improve line predictions? | On the same {word['test_lines']:,} lines, the whole-line model reached "
                f"{word['whole_line']['macro_f1']:.2%} Macro-F1 and word voting reached {word['word_vote']['macro_f1']:.2%}. | "
                f"{publish(page, 'Word vs Line', detail_path.parent / 'word-vs-line.png', 'word-vs-line.png')} |\n"
                f"| Do the two methods make different errors? | Only the whole-line model missed {errors['only_whole_line_wrong']} lines, "
                f"only word voting missed {errors['only_word_vote_wrong']}, and both missed {errors['both_wrong']}. | "
                f"{publish(page, 'Error overlap', detail_path.parent / 'error-overlap.png', 'error-overlap.png')} |\n"
                f"| Does pooling the predictions help? | The 50/50 combination reached "
                f"{ensemble['macro_f1']['mean']:.2%} mean Macro-F1 across {ensemble['runs_or_pairs']} pairs. | "
                f"{publish(page, 'Word and line ensemble', detail_path.parent / 'word-line-ensemble.png', 'word-line-ensemble.png')} |",
                publish(page, "Full experiment scores", detail_path, "details.json"),
            ]
        write(page, "\n\n".join(lines))


def write_comparison(results, scores):
    page = REPORT / "02-model-comparison/README.md"
    lines = ["# Four-model comparison",
             "The four trained classifiers were compared on the same masked whole lines. Each gives one writer prediction per line. The whole-line protocol avoids restricting the comparison to lines whose words could be segmented.",
             "Macro-F1 averages F1 across writers, which gives rare writers the same weight as common writers. The held-out-page subset excludes pages used in training or validation.",
             "All four runs use the same train, validation and test split, class weighting and square-root sampling. The ImageNet-initialized baselines fine-tune their encoders. CHURRO trains a linear classifier on frozen block-16 features. The checkpoint for each run is selected on validation Macro-F1.",
             "The frozen CHURRO probe and the fine-tuned Swin-Tiny are different training setups. Their scores compare the resulting classifiers. They do not isolate the effects of architecture, pretraining or fine-tuning."]
    if scores:
        options = (("All 7,636 test lines · nine writers", scores),
                   ("7,583 lines from unseen pages · seven writers", scores["unseen_pages"]))
        for title, subset in options:
            order = sorted(LABELS, key=lambda name: subset["models"][name]["macro_f1"], reverse=True)
            lines.extend((f"## {title}", score_table(subset["models"], order)))
        comparison = results / "comparison"
        for label, target, filename, image in (
            ("Macro-F1 overview", comparison / "00-final-sonuc.png", "final-result.png", True),
            ("Scores across both test sets", comparison / "01-model-scores-vertical.png", "both-test-sets.png", False),
            ("Confusion matrices", comparison / "metrics/confusion.png", "confusion-matrices.png", False),
            ("Shared Grad-CAM and attention panel", comparison / "explanations/line_explanations.png", "grad-cam-and-attention.png", False),
        ):
            if target.is_file():
                lines.append(publish(page, label, target, filename, image))
        scores_csv(page, (("full_test", scores), ("unseen_pages", scores["unseen_pages"])), LABELS)
        lines.append("[Scores CSV](scores.csv)")
    lines.append("All four classifiers were trained with nine target labels. The unseen-page test excludes every page that supplied any training or validation line. This removed all 14 S11 lines, all 7 S12 lines, and 32 lines from other writers. The seven represented labels are used for Macro-F1 on this subset. The models can still predict any of the nine labels.")
    write(page, "\n\n".join(lines))


def writer_f1(matrix):
    counts = np.array(matrix)
    support = counts.sum(axis=1)
    denominator = support + counts.sum(axis=0)
    f1 = np.divide(2 * counts.diagonal(), denominator, out=np.zeros(len(support)), where=denominator > 0)
    return f1[support > 0]


def pair_summary_figure(page, means):
    """Explain the repeat experiment and show its result without seed-by-seed detail."""
    fig = plt.figure(figsize=(12, 6.35), facecolor="white")
    ink, muted = "#183146", "#486173"
    fig.text(.05, .915, "Did ResNet18’s lead repeat?", fontsize=25, weight="bold", color=ink)
    fig.text(.05, .851, "Nuremberg-9 · One writer prediction per handwritten line",
             fontsize=12, color=muted)

    notes = (
        ("Why we checked", "ResNet18 led in the first whole-line comparison.\nOne run can be sensitive to the training seed."),
        ("What we repeated", "Five runs per model, seeds 42–46.\nThe training, validation and test splits stayed fixed."),
        ("How the models learned", "ResNet18 updated the whole network.\nCHURRO kept the encoder frozen and trained its classifier."),
    )
    for index, (heading, body) in enumerate(notes):
        y = .69 - index * .19
        fig.text(.05, y, heading, fontsize=13, weight="bold", color=ink)
        fig.text(.05, y - .095, body, fontsize=10.5, linespacing=1.45, color=muted)

    table = fig.add_axes([.54, .235, .405, .47])
    table.set(xlim=(0, 1), ylim=(0, 1))
    table.axis("off")
    table.text(0, .98, "Mean Macro-F1 across five runs", weight="bold", fontsize=14, color=ink)
    for xpos, model in ((.62, "churro"), (.89, "resnet18")):
        table.text(xpos, .77, LABELS[model], ha="center", weight="bold", fontsize=10.5, color=COLORS[model])
    for index, (label, details) in enumerate(means.items()):
        y = .59 - index * .34
        table.axhline(y + .11, color="#d9e4e9", linewidth=.8)
        table.text(0, y, label, va="center", fontsize=11, weight="bold", color=ink)
        table.text(0, y - .13, f"{details['lines']:,} lines · {details['writers']} writers",
                   fontsize=9, color=muted)
        for xpos, model in ((.62, "churro"), (.89, "resnet18")):
            table.text(xpos, y, f"{details[model]:.2%}", ha="center", va="center",
                       fontsize=13, color=COLORS[model], weight="bold")
    fig.text(.05, .09, "The lead persisted across the five runs. To explain its cause, change one training factor at a time.",
             fontsize=11, color=ink, weight="bold")
    save(fig, page.parent / "experiment-overview.png")


def write_pair(results, scores):
    page = REPORT / "03-resnet18-vs-churro/README.md"
    lines = ["# ResNet18 vs CHURRO-3B",
             "ResNet18 led the first whole-line comparison. We asked whether it would still lead after repeating training, and which writers and lines accounted for the first-run difference.",
             "Both models kept the same train, validation and test splits and made one writer prediction per whole line. CHURRO used frozen visual features and a linear head. ResNet18 was fine-tuned end to end. The repeat changes the training seed. Measuring the effect of an individual training choice would require a separate controlled experiment."]
    if scores:
        subset = scores["unseen_pages"]
        writers = subset["scribes_with_test_lines"]
        page.parent.mkdir(parents=True, exist_ok=True)
        scores_csv(page, (("full_test", scores), ("unseen_pages", subset)), ("churro", "resnet18"))
        fig, ax = plt.subplots(figsize=(9.4, 4.5), layout="constrained")
        x = np.arange(len(writers))
        for name, offset in (("churro", -.18), ("resnet18", .18)):
            values = writer_f1(subset["models"][name]["confusion_matrix"])
            ax.bar(x + offset, values, .36, color=COLORS[name], label=LABELS[name])
        ax.set(title="ResNet18 vs CHURRO-3B · F1 on 7,583 test lines from unseen pages",
               xticks=x, xticklabels=[f"S{i}" for i in writers], ylim=(0, 1.04), ylabel="F1")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="lower right", frameon=False)
        save(fig, page.parent / "unseen-page-writer-f1.png")
        lines += ["## Original runs · seed 42", score_table(subset["models"], ["churro", "resnet18"]),
                  "![F1 by writer on unseen pages](unseen-page-writer-f1.png)", "[Both test sets in CSV](scores.csv)"]
        predictions = results / "resnet18/fit/predictions.npz"
        if predictions.is_file():
            compare = run_path(str(EXP / "05_model_comparison/compare.py"))
            rows, truth, churro = compare["load_predictions"](CHURRO / "blk16_predictions.npz", len(scores["scribes"]))
            resnet = compare["align_predictions"](predictions, rows, truth, len(scores["scribes"]))
            held_out = compare["unseen_page_mask"](read_json(N9 / "docs/split_v2_9scribes.json"), rows)
            rc = resnet.argmax(1)[held_out] == truth[held_out]
            cc = churro.argmax(1)[held_out] == truth[held_out]
            only_resnet = np.count_nonzero(rc & ~cc)
            only_churro = np.count_nonzero(cc & ~rc)
            both_wrong = np.count_nonzero(~cc & ~rc)
            fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
            bars = ax.bar(["Only ResNet18\ncorrect", "Only CHURRO-3B\ncorrect", "Both wrong"],
                          [only_resnet, only_churro, both_wrong],
                          color=[COLORS["resnet18"], COLORS["churro"], "#78919d"])
            ax.bar_label(bars, padding=4)
            ax.set(title="Where the two classifiers made errors\n7,583 unseen-page test lines · Seed 42",
                   ylabel="Test lines", ylim=(0, 1.17 * max(only_resnet, only_churro, both_wrong, 1)))
            save(fig, page.parent / "which-model-was-correct.png")
            lines.append(f"On the same unseen-page lines, only ResNet18 was correct on {only_resnet} lines "
                         f"and only CHURRO-3B was correct on {only_churro} lines. "
                         "Predictions were aligned by page and line ID.")
            lines.append("[Prediction disagreements and shared errors](which-model-was-correct.png)")
    if (repeat_file := REPEATS / "scores.json").is_file():
        expanded = read_json(repeat_file)
        keysets = (("churro", "churro", "churro_s"), ("resnet18", "resnet18_s42", "resnet18_s"))
        table = ["| Test set | Model | Runs | Mean Macro-F1 | Sample standard deviation |",
                 "|---|---|---:|---:|---:|"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
        fig.suptitle("ResNet18 vs CHURRO-3B · Whole-line Macro-F1 across training seeds", weight="bold")
        means = {}
        for ax, (label, subset) in zip(axes, (("All test lines", expanded), ("Unseen pages", expanded["unseen_pages"]))):
            writers = subset["scribes_with_test_lines"] if label == "Unseen pages" else expanded["scribes"]
            means[label] = {"lines": subset["n_test_lines"], "writers": len(writers)}
            for name, original, prefix in keysets:
                values = np.array([subset["models"][k]["macro_f1"] for k in [original, *(f"{prefix}{s}" for s in range(43, 47))]])
                means[label][name] = values.mean()
                table.append(f"| {label} | {LABELS[name]} | {len(values)} | {values.mean():.2%} | {values.std(ddof=1):.2%} |")
                ax.plot(range(42, 47), values, "o-", color=COLORS[name], label=LABELS[name])
            ax.set(title=label, xlabel="Training seed", xticks=range(42, 47))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].legend(frameon=False)
        save(fig, page.parent / "macro-f1-by-seed.png")
        pair_summary_figure(page, means)
        lines.insert(3, "![Why we repeated the experiment and the result](experiment-overview.png)")
        lines += ["## Repeated training · seeds 42–46", "\n".join(table),
                  "![Whole-line Macro-F1 by training seed](macro-f1-by-seed.png)",
                  publish(page, "Metrics for each seed", repeat_file, "scores-by-seed.json")]
    else:
        lines += ["## Repeated training · seeds 42–46",
                  "The repeat summary has not been generated. This page currently reports the original seed-42 comparison."]
    write(page, "\n\n".join(lines))


def main():
    results = next((p for p in (FITS, LOCAL_RESULTS) if (p / "comparison/metrics/scores.json").is_file()), None)
    scores = read_json(results / "comparison/metrics/scores.json") if results else None
    write_models(results)
    write_comparison(results, scores)
    if results:
        render_line_diagnostics(results, REPORT)
    write_pair(results, scores)
    guide = """
# Nuremberg-9 paper results

| Start here | Use in the paper |
|---|---|
| [Individual models](01-models/README.md) | Methods and supplementary results. Each model page explains the training setup and links to its figures. |
| [Four-model comparison](02-model-comparison/README.md) | Results. Main scores and figures for the whole-line writer identification task. |
| [ResNet18 vs CHURRO-3B](03-resnet18-vs-churro/README.md) | Discussion. Writer-level differences, repeats and limits on what the experiment can explain. |

[Open the results in a browser](BASLA.html)

The PNG figures and result tables are included in these folders. The source runs for this export are in nuremberg-9/experiments.
"""
    if (REPORT / "04-writer-lines/README.md").is_file():
        guide += "\n[See one handwritten line from each writer](04-writer-lines/README.md)\n"
    write(REPORT / "README.md", guide)
    print(f"Updated {REPORT}")


if __name__ == "__main__":
    main()

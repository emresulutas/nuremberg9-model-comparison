"""Render diagnostics from the four saved Nuremberg-9 whole-line classifiers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from compare import REFERENCE, ROOT, SPLIT, align_predictions, load_predictions, scores, unseen_page_mask
from four_model_errors import draw, draw_intersections, exact_counts

LABELS = {"churro": "CHURRO-3B", "resnet18": "ResNet18", "resnet50": "ResNet50", "swin_tiny": "Swin-Tiny"}


def save(fig, out, name):
    fig.savefig(out / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def per_writer(matrix):
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = matrix.diagonal()
    precision = np.divide(tp, predicted, where=predicted > 0, out=np.zeros(len(tp)))
    recall = np.divide(tp, support, where=support > 0, out=np.zeros(len(tp)))
    f1 = np.divide(2 * tp, support + predicted, where=support + predicted > 0, out=np.zeros(len(tp)))
    return support, precision, recall, f1


def writer_plot(out, title, subset, scribes):
    support, *metrics = subset
    present = np.flatnonzero(support)
    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(10, 4.3), layout="constrained")
    for offset, label, values, color in zip((-.25, 0, .25), ("Precision", "Recall", "F1"),
                                            metrics, ("#587482", "#287f70", "#4a63a3"), strict=True):
        ax.bar(x + offset, values[present], .24, label=label, color=color)
    labels = [f"S{scribes[i]}\n(n={support[i]:,})" for i in present]
    ax.set(title=f"{title} · Precision, recall and F1 by writer\n{support.sum():,} test lines from unseen pages",
           xticks=x, xticklabels=labels, ylim=(0, 1.08))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower left", ncol=3, frameon=False)
    save(fig, out, "line-scores-by-writer.png")


def error_plot(out, title, matrix, scribes):
    wrong = matrix.copy()
    np.fill_diagonal(wrong, 0)
    pairs = sorted(zip(*np.nonzero(wrong)), key=lambda pair: int(wrong[pair]), reverse=True)[:8]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    if pairs:
        pairs.reverse()
        counts = [int(wrong[i, j]) for i, j in pairs]
        bars = ax.barh([f"S{scribes[i]} → S{scribes[j]}" for i, j in pairs], counts, color="#567486")
        ax.bar_label(bars, padding=4)
        ax.set_xlim(0, max(counts) * 1.18)
    ax.set(title=f"{title} · Most frequent writer confusions\n{matrix.sum():,} unseen-page lines · True → predicted",
           xlabel="Incorrect test lines")
    save(fig, out, "common-line-errors.png")
    return [(scribes[i], scribes[j], int(wrong[i, j])) for i, j in reversed(pairs)]


def confidence_and_pages(out, title, prob, truth, pages):
    pred = prob.argmax(1)
    correct = pred == truth
    confidence = prob.max(1)
    boundaries = np.array([0, .50, .70, .80, .90, .95, .99, 1.01])
    groups = np.digitize(confidence, boundaries[1:-1])
    labels = ("<50", "50–69", "70–79", "80–89", "90–94", "95–98", "99–100")
    fig, (ax, page_ax) = plt.subplots(1, 2, figsize=(12.6, 4.6), layout="constrained")
    fig.suptitle(f"{title} · Confidence and page variation · {len(truth):,} unseen-page test lines")
    counts = np.bincount(groups, minlength=len(labels))
    rate = np.divide(np.bincount(groups, weights=correct, minlength=len(labels)), counts,
                     where=counts > 0, out=np.zeros(len(labels)))
    use = np.flatnonzero(counts)
    bars = ax.bar(use, rate[use], color="#287f70")
    ax.bar_label(bars, labels=[f"n={counts[i]:,}" for i in use], padding=4, fontsize=8)
    ax.set(title="Observed accuracy by predicted confidence", xlabel="Highest model probability (%)",
           xticks=range(len(labels)), xticklabels=labels, ylabel="Observed accuracy", ylim=(0, 1.12))
    ax.tick_params(axis="x", labelsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    _, page_index = np.unique(pages, return_inverse=True)
    page_counts = np.bincount(page_index)
    page_accuracy = np.bincount(page_index, weights=correct) / page_counts
    use_page = page_counts >= 10
    if use_page.any():
        page_ax.hist(page_accuracy[use_page], bins=np.linspace(0, 1, 11), color="#4a63a3")
    page_ax.set(title=f"Accuracy across pages · {use_page.sum()} pages with ≥10 test lines",
                xlabel="Page accuracy", ylabel="Pages", xlim=(0, 1.01))
    page_ax.xaxis.set_major_formatter(PercentFormatter(1))
    save(fig, out, "line-confidence-and-pages.png")

    n_high_wrong = int(((confidence >= .95) & ~correct).sum())
    return n_high_wrong


def render_model(name, out, y, prob, mask, pages, scribes):
    out.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(scores(y[mask], prob[mask], scribes)["confusion_matrix"])
    subset = per_writer(matrix)
    writer_plot(out, LABELS[name], subset, scribes)
    pairs = error_plot(out, LABELS[name], matrix, scribes)
    high_wrong = confidence_and_pages(out, LABELS[name], prob[mask], y[mask], pages[mask])
    rows = []
    for label, selected in (("full_test", np.ones(len(y), dtype=bool)), ("unseen_pages", mask)):
        m = np.asarray(scores(y[selected], prob[selected], scribes)["confusion_matrix"])
        for writer, support, p, r, f in zip(scribes, *per_writer(m), strict=True):
            if support:
                rows.append((label, f"S{writer}", support, p, r, f))
    write_csv(out / "line-scores-by-writer.csv", ("test_subset", "writer", "lines", "precision", "recall", "f1"), rows)
    top = f" The largest was S{pairs[0][0]} predicted as S{pairs[0][1]} on {pairs[0][2]} lines." if pairs else ""
    with (out / "README.md").open("a", encoding="utf-8") as f:
        f.write(f"""

## Whole-line diagnostic figures

The three charts use the same {mask.sum():,} lines from pages excluded from training and validation.

- [Precision, recall and F1 by writer](line-scores-by-writer.png) shows who was identified well. Sample counts are under each writer. [Writer scores for both test sets](line-scores-by-writer.csv)
- [Most frequent writer confusions](common-line-errors.png) shows where the errors went.{top}
- [Confidence and page variation](line-confidence-and-pages.png) shows how accuracy changes across probability ranges and pages. The model made {high_wrong} errors with at least 95% predicted confidence. Probability here is the model's raw score.
""")
    return subset[3]


def render_comparison(out, y, prob, mask, scribes, writer_f1, rows_by_id):
    out.mkdir(parents=True, exist_ok=True)
    support = np.bincount(y[mask], minlength=len(scribes))
    ids = np.flatnonzero(support)
    models = list(prob)
    f1 = np.array([writer_f1[name][ids] for name in models])
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    fig.colorbar(ax.imshow(f1, cmap="Blues", vmin=.75, vmax=1), ax=ax, format=PercentFormatter(1), label="F1")
    for i, j in np.ndindex(f1.shape):
        ax.text(j, i, f"{f1[i,j]:.1%}", ha="center", va="center",
                color="white" if f1[i,j] > .91 else "#163143",
                weight="bold" if f1[i,j] == f1[:,j].max() else "normal")
    ax.set(title=f"Four whole-line models · F1 by writer · {mask.sum():,} unseen-page test lines",
           xticks=range(len(ids)), xticklabels=[f"S{scribes[i]}\n(n={support[i]:,})" for i in ids],
           yticks=range(len(models)), yticklabels=[LABELS[m] for m in models])
    save(fig, out, "line-models-by-writer.png")

    names = [LABELS[name] for name in models]
    full = exact_counts(y, list(prob.values()))
    unseen = exact_counts(y[mask], [p[mask] for p in prob.values()])
    save(draw(unseen, names), out, "line-model-error-overlap.png")

    save(draw_intersections(unseen, names), out, "shared-error-intersections.png")

    rows = []
    for subset, counts in (("full_test", full), ("unseen_pages", unseen)):
        for bits, count in enumerate(counts):
            wrong = [n for i, n in enumerate(names) if bits & (1 << i)]
            correct = [n for i, n in enumerate(names) if not bits & (1 << i)]
            rows.append((subset, " + ".join(wrong), " + ".join(correct), len(wrong), int(count)))
    write_csv(out / "line-model-error-overlap.csv", ("test_subset", "wrong_models", "correct_models", "models_wrong", "lines"), rows)
    predictions = np.column_stack([p.argmax(1) for p in prob.values()])
    case_rows = (
        (page, line, bool(mask[i]), f"S{scribes[y[i]]}", *(f"S{scribes[j]}" for j in predictions[i]),
         " + ".join(name for name, j in zip(names, predictions[i], strict=True) if j != y[i]))
        for i, (page, line) in enumerate(rows_by_id)
        if np.any(predictions[i] != y[i])
    )
    write_csv(out / "line-error-cases.csv", ("page_id", "line_id", "unseen_page", "true_writer",
                                            *models, "wrong_models"), case_rows)
    with (out / "README.md").open("a", encoding="utf-8") as f:
        f.write("\n\n## Whole-line error analysis\n\n"
                "- [F1 by writer for all four models](line-models-by-writer.png)\n"
                "- [Shared and single-model errors at a glance](line-model-error-overlap.png) · [All exact intersections for both test sets](line-model-error-overlap.csv)\n"
                "- [Every shared error group by model name](shared-error-intersections.png)\n"
                "- [Find the actual lines and each model's predicted writer](line-error-cases.csv)\n\n"
                f"On unseen pages, {unseen[15]:,} lines were misclassified by all four models and "
                f"{unseen[0]:,} were correctly classified by all four. The figure also shows lines that only one model got right. "
                "The counts describe complementary predictions from the four existing classifiers.\n")


def render(results, report):
    split = json.loads((ROOT / SPLIT).read_text(encoding="utf-8"))
    scribes = sorted(split["scribes"])
    rows, y, base = load_predictions(ROOT / REFERENCE, len(scribes))
    paths = {name: results / name / "fit/predictions.npz" for name in LABELS if name != "churro"}
    if not all(p.is_file() for p in paths.values()):
        return
    prob = {"churro": base, **{name: align_predictions(p, rows, y, len(scribes)) for name,p in paths.items()}}
    mask = unseen_page_mask(split, rows)
    pages = np.array([page for page, _ in rows])
    f1 = {name: render_model(name, report / "01-models" / name, y, p, mask, pages, scribes)
          for name, p in prob.items()}
    render_comparison(report / "02-model-comparison", y, prob, mask, scribes, f1, rows)

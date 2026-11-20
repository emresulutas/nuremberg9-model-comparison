"""Figures and tables for the report (matplotlib only, no torch, so they also run on the laptop).

Scribe classes are named S1, S2, ... after the scribe numbers in the XML. Every function takes plain
numbers/arrays and writes one PNG, so nuremberg-7/scripts/build_report.py can call them for any experiment.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

SPLITS = ("train", "val", "test")


def scribe_colors(n: int):
    return [plt.cm.tab10(i % 10) for i in range(n)]


def _save(fig, out) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def table_png(header, rows, out, title=None, best_cols=(), col_widths=None) -> None:
    """Render a table as an image. best_cols: indices of numeric columns whose maximum is highlighted."""
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(header)), 0.45 * (len(rows) + 2) + (0.5 if title else 0)))
    ax.axis("off")
    cells = [[c if isinstance(c, str) else f"{c:.3f}" if isinstance(c, float) else str(c) for c in r] for r in rows]
    tab = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center", colWidths=col_widths)
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.4)
    for j in range(len(header)):
        tab[0, j].set_facecolor("#dfe6f0")
        tab[0, j].set_text_props(weight="bold")
    for j in best_cols:
        vals = [r[j] for r in rows if isinstance(r[j], float)]
        if vals:
            for i, r in enumerate(rows, start=1):
                if isinstance(r[j], float) and r[j] == max(vals):
                    tab[i, j].set_facecolor("#d6efd6")
                    tab[i, j].set_text_props(weight="bold")
    if title:
        ax.set_title(title, fontsize=12, pad=8)
    _save(fig, out)


# ---------------------------------------------------------------- data figures

def pies_lines_per_scribe(summary: dict, scribes, out) -> None:
    cols = scribe_colors(len(scribes))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, sp in zip(axes, SPLITS):
        counts = [summary[sp][str(s)]["lines"] for s in scribes]
        ax.pie(counts, colors=cols, startangle=90, counterclock=False,
               autopct=lambda p: f"{p:.1f}%" if p >= 4 else "", pctdistance=0.75,
               wedgeprops={"edgecolor": "white"})
        ax.set_title(f"{sp}: {sum(counts):,} lines")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in cols],
               [f"S{s} (train {summary['train'][str(s)]['lines']:,} / val {summary['val'][str(s)]['lines']:,} / "
                f"test {summary['test'][str(s)]['lines']:,})" for s in scribes],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Share of lines per scribe in each split")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def bars_lines_pages(summary: dict, scribes, out) -> None:
    cols = scribe_colors(3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(scribes))
    for ax, key in zip(axes, ("lines", "pages")):
        for k, sp in enumerate(SPLITS):
            vals = [summary[sp][str(s)][key] for s in scribes]
            bars = ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=sp, color=cols[k])
            ax.bar_label(bars, fontsize=7)
        ax.set_xticks(x, [f"S{s}" for s in scribes])
        ax.set_title(f"{key} per scribe and split")
        if key == "lines":
            ax.set_yscale("log")
        ax.legend()
    _save(fig, out)


def band_heatmap(stats: dict, scribes, out) -> None:
    bands = sorted({b for s in scribes for b in stats["scribes"][str(s)]["lines_by_band"]})
    mat = np.array([[stats["scribes"][str(s)]["lines_by_band"].get(b, 0) for b in bands] for s in scribes], float)
    fig, ax = plt.subplots(figsize=(1.2 * len(bands) + 3, 0.6 * len(scribes) + 2))
    im = ax.imshow(mat / mat.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{int(mat[i, j]):,}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] / mat[i].sum() > 0.6 else "black")
    ax.set_xticks(range(len(bands)), bands)
    ax.set_yticks(range(len(scribes)), [f"S{s}" for s in scribes])
    ax.set_title("Lines of each scribe per book (colour = share of the scribe's lines)\n"
                 "scribes are tied to books -> a model could cheat via the book")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, out)


# ---------------------------------------------------------------- result figures

def per_class_metrics(cm) -> list[dict]:
    cm = np.asarray(cm, float)
    out = []
    for i in range(len(cm)):
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({"precision": prec, "recall": rec, "f1": f1, "support": int(support)})
    return out


def confusion(cm, names, out, title="Confusion matrix") -> None:
    cm = np.asarray(cm, float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, sub, fmt in ((axes[0], cm, "counts", "{:.0f}"), (axes[1], norm, "share of true class (recall)", "{:.2f}")):
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title}: {sub}")
    fig.colorbar(im, ax=axes, shrink=0.8)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def outcome_pies(cm, names, out, title="Test lines per scribe: correct vs wrong") -> None:
    cm = np.asarray(cm, float)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.0))
    for i, ax in enumerate(np.atleast_1d(axes)):
        right, total = cm[i, i], cm[i].sum()
        ax.pie([right, total - right], colors=["#4c9f70", "#d9534f"], startangle=90, counterclock=False,
               autopct=lambda p: f"{p:.0f}%" if p >= 5 else "", wedgeprops={"edgecolor": "white"})
        ax.set_title(f"{names[i]}\n(n={int(total)})", fontsize=9)
    fig.suptitle(title + "  (green = correct)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def recall_bars(recalls: dict, names, out, title="Recall per scribe (test)") -> None:
    """recalls: {series name: [recall per scribe]}"""
    series = list(recalls)
    cols = scribe_colors(len(series))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(names)), 4.2))
    x = np.arange(len(names))
    for k, s in enumerate(series):
        ax.bar(x + (k - (len(series) - 1) / 2) * width, recalls[s], width, label=s, color=cols[k])
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall")
    ax.set_title(title)
    ax.legend(ncol=min(len(series), 5), fontsize=8)
    _save(fig, out)


def layer_bars(rows: dict, baseline: dict, out) -> None:
    """rows: {layer: {val_balanced_accuracy, val_macro_f1, test_balanced_accuracy, test_macro_f1}}"""
    layers = list(rows)
    keys = [("val_balanced_accuracy", "val balanced acc"), ("val_macro_f1", "val macro-F1"),
            ("test_balanced_accuracy", "test balanced acc"), ("test_macro_f1", "test macro-F1")]
    cols = scribe_colors(4)
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(layers)), 4.4))
    x = np.arange(len(layers))
    for k, (key, label) in enumerate(keys):
        bars = ax.bar(x + (k - 1.5) * 0.2, [rows[l][key] for l in layers], 0.2, label=label, color=cols[k])
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    if baseline:
        ax.axhline(baseline["test"]["balanced_accuracy"], color="grey", ls="--", lw=1)
        ax.text(-0.48, baseline["test"]["balanced_accuracy"] + 0.012, "band shortcut baseline (test bAcc)",
                ha="left", fontsize=8, color="dimgrey", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_xticks(x, layers)
    ax.set_ylim(0, 1.05)
    ax.set_title("Which encoder depth separates scribes best?")
    ax.legend(ncol=4, fontsize=8, loc="lower center")
    _save(fig, out)


def band_accuracy(y, pred, page_ids, out, baseline=None, title="Test accuracy per book") -> dict:
    """Accuracy and balanced accuracy of the test lines, split by book (Band2..Band5)."""
    bands = np.array([str(p).split("/")[0] for p in page_ids])
    y, pred = np.asarray(y), np.asarray(pred)
    res = {}
    for b in sorted(set(bands)):
        m = bands == b
        recalls = [(pred[m & (y == c)] == c).mean() for c in np.unique(y[m])]
        res[b] = {"lines": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean()),
                  "balanced_accuracy": float(np.mean(recalls))}
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(res))
    ax.bar(x - 0.2, [v["accuracy"] for v in res.values()], 0.4, label="accuracy")
    ax.bar(x + 0.2, [v["balanced_accuracy"] for v in res.values()], 0.4, label="balanced acc (scribes in that book)")
    ax.set_xticks(x, [f"{b}\n(n={v['lines']})" for b, v in res.items()])
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, out)
    return res


def page_accuracy_hist(y, pred, page_ids, out, title="Test: accuracy per page") -> None:
    per_page = defaultdict(list)
    for yy, pp, pid in zip(y, pred, page_ids):
        per_page[str(pid)].append(yy == pp)
    acc = np.array([np.mean(v) for v in per_page.values()])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(acc, bins=np.linspace(0, 1, 11), color="#4c78a8", edgecolor="white")
    ax.set_xlabel("share of correctly classified lines on the page")
    ax.set_ylabel("pages")
    ax.set_title(f"{title} ({len(acc)} pages, median {np.median(acc):.2f})")
    _save(fig, out)


# ---------------------------------------------------------------- AUC, curves, controlled test

def ovr_scores(y, prob) -> dict:
    """One-vs-rest ROC-AUC and average precision (PR-AUC) per scribe, plus their macro mean."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, prob = np.asarray(y), np.asarray(prob)
    auc, ap = [], []
    for c in range(prob.shape[1]):
        t = (y == c).astype(int)
        if t.sum() in (0, len(t)):
            auc.append(float("nan"))
            ap.append(float("nan"))
            continue
        auc.append(float(roc_auc_score(t, prob[:, c])))
        ap.append(float(average_precision_score(t, prob[:, c])))
    return {"roc_auc": auc, "average_precision": ap,
            "roc_auc_macro": float(np.nanmean(auc)), "average_precision_macro": float(np.nanmean(ap))}


def roc_pr_curves(y, prob, names, out, title="") -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve
    y, prob = np.asarray(y), np.asarray(prob)
    sc = ovr_scores(y, prob)
    cols = scribe_colors(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for c, name in enumerate(names):
        t = (y == c).astype(int)
        if t.sum() in (0, len(t)):
            continue
        fpr, tpr, _ = roc_curve(t, prob[:, c])
        pr, rc, _ = precision_recall_curve(t, prob[:, c])
        axes[0].plot(fpr, tpr, color=cols[c], label=f"{name} (AUC {sc['roc_auc'][c]:.3f})")
        axes[1].plot(rc, pr, color=cols[c], label=f"{name} (AP {sc['average_precision'][c]:.3f})")
    axes[0].plot([0, 1], [0, 1], color="grey", ls=":")
    axes[0].set(title=f"{title} ROC, one scribe vs rest", xlabel="false positive rate", ylabel="true positive rate")
    axes[1].set(title=f"{title} precision-recall", xlabel="recall", ylabel="precision", ylim=(0, 1.02))
    for ax in axes:
        ax.legend(fontsize=8)
    _save(fig, out)


def multi_scribe_pages(y, pred, page_ids):
    """Controlled test: only test pages that hold lines of at least two scribes. Paper, ink and scan are
    the same inside such a page, so a model that works here is reading the hand, not the book."""
    y, pred = np.asarray(y), np.asarray(pred)
    pages = defaultdict(list)
    for i, p in enumerate(page_ids):
        pages[str(p)].append(i)
    chosen = [ids for ids in pages.values() if len(set(y[ids])) >= 2]
    if not chosen:
        return None
    idx = np.array([i for ids in chosen for i in ids])
    yy, pp = y[idx], pred[idx]
    recalls = [(pp[yy == c] == c).mean() for c in np.unique(yy)]
    return {"pages": len(chosen), "lines": int(len(idx)), "accuracy": float((pp == yy).mean()),
            "balanced_accuracy": float(np.mean(recalls)), "scribes": [int(c) for c in np.unique(yy)]}

"""Figures for the word-level blk16 maps (used by explain_words.py and redraw_words.py).

An item is a dict like in explain_plots.py (crop, evidence, cam, attention, scribe, kind, title) plus
"word", "pred" (class index of the prediction) and "probs" (probability of every scribe, in class order).
Per run and word two pictures:
  word_<word>.png   WHO: per scribe (row) three crops, next to each crop the probability the head gives to every scribe
  maps_<word>.png   WHERE: per scribe one crop with evidence / Grad-CAM / attention (top 15 % patches in yellow)
Over all words: word_accuracy.png (scribe x word heatmap), word_confusion.png, word_per_scribe.png (seaborn).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import seaborn as sns             # noqa: E402

from .explain_plots import METHODS, TOP_SHOW, ink_overlap, top_mask   # noqa: E402

GREEN, RED, GREY = "#2a7f3f", "#c0392b", "#b8b8b8"
PANELS = ("word", "evidence", "Grad-CAM", "attention")


def slug(word: str) -> str:
    """File-name safe version of a word (the long z 'ʒ' becomes 'z')."""
    return word.replace("ʒ", "z")


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_who(items: list[dict], word: str, scribes: list[int], run: str, path: Path, per_scribe: int = 3) -> None:
    """Per scribe (row) per example: the crop, and how likely the head thinks each of the 7 scribes wrote it."""
    n = len(scribes)
    fig = plt.figure(figsize=(per_scribe * 5.4 + 1.4, n * 1.45 + 1.4))
    gs = fig.add_gridspec(n, per_scribe * 2, width_ratios=[1.25, 1.75] * per_scribe, wspace=0.12, hspace=0.45)
    for c in range(n):
        all_c = [it for it in items if it["scribe"] == c]
        right = sum(it["pred"] == c for it in all_c)
        for j in range(per_scribe):
            ax_img, ax_bar = fig.add_subplot(gs[c, 2 * j]), fig.add_subplot(gs[c, 2 * j + 1])
            _clean(ax_img)
            if j >= len(all_c):
                ax_bar.axis("off")
                continue
            it = all_c[j]
            h, w = it["crop"].shape[:2]
            ax_img.imshow(it["crop"], extent=(0, w, h, 0), aspect="equal")
            if c == 0:
                ax_img.set_title(f"example {j + 1}", fontsize=10, fontweight="bold", pad=6, loc="left")
            if j == 0:
                ax_img.set_ylabel(f"S{scribes[c]}\n{right}/{len(all_c)} right", rotation=0, ha="right", va="center",
                                  fontsize=10, fontweight="bold", labelpad=8)
            p = np.asarray(it["probs"])
            colors = [GREY] * n
            colors[it["pred"]] = GREEN if it["pred"] == c else RED
            colors[c] = GREEN
            ax_bar.barh(range(n), p, color=colors, height=0.75)
            ax_bar.set_ylim(n - 0.5, -0.5)
            ax_bar.set_xlim(0, 1.18)
            ax_bar.set_yticks(range(n), [f"S{s}" for s in scribes], fontsize=6.5)
            ax_bar.tick_params(length=0, pad=1)
            ax_bar.set_xticks([])
            for sp in ax_bar.spines.values():
                sp.set_visible(False)
            for k, v in enumerate(p):
                if v >= 0.05:
                    ax_bar.text(v + 0.02, k, f"{v * 100:.0f}%", va="center", fontsize=6.5)
            ok = it["pred"] == c
            ax_bar.set_title(f"guess: S{scribes[it['pred']]} ({p[it['pred']] * 100:.0f}%)  " + ("RIGHT" if ok else "WRONG"),
                             fontsize=8, color=GREEN if ok else RED, fontweight="bold", pad=2)
    fig.suptitle(f'THE WORD "{word}" - {run} head', fontsize=15, fontweight="bold", y=1.0)
    fig.text(0.5, 0.955, "Every row = the scribe who really wrote it (left). Each row shows 3 different examples of this word by that scribe.\n"
             "Bars = how likely the head thinks each scribe wrote the crop. Green = true scribe, red = the head's wrong top guess.",
             ha="center", va="top", fontsize=9)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def draw_gallery(items: list[dict], words: list[str], scribes: list[int], run: str, path: Path) -> None:
    """The picture matrix: rows = scribes, columns = the 5 words, one crop per cell with the head's guess under it."""
    n = len(scribes)
    fig, axes = plt.subplots(n, len(words), figsize=(len(words) * 2.5 + 1.3, n * 1.5 + 1.3), squeeze=False)
    for c in range(n):
        for j, word in enumerate(words):
            ax = axes[c, j]
            _clean(ax)
            if c == 0:
                ax.set_title(word, fontsize=14, fontweight="bold", pad=6)
            if j == 0:
                ax.set_ylabel(f"written by\nS{scribes[c]}", rotation=0, ha="right", va="center", fontsize=11, fontweight="bold", labelpad=8)
            its = [it for it in items if it["word"] == word and it["scribe"] == c]
            if not its:
                continue
            it = its[0]
            h, w = it["crop"].shape[:2]
            ax.imshow(it["crop"], extent=(0, w, h, 0), aspect="equal")
            ok = it["pred"] == c
            ax.set_xlabel(f"guess S{scribes[it['pred']]} ({it['probs'][it['pred']] * 100:.0f}%)" + ("" if ok else f"\ntrue S{scribes[c]}: {it['probs'][c] * 100:.0f}%"),
                          fontsize=8.5, color=GREEN if ok else RED, fontweight="bold", labelpad=2)
    fig.suptitle(f"{run} head: one example per scribe and word. Green = scribe guessed right, red = wrong (with the share it gave the true scribe)",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def draw_maps(items: list[dict], word: str, scribes: list[int], run: str, path: Path) -> None:
    """Per scribe one crop and where the head looks: evidence, Grad-CAM, attention (top patches in yellow)."""
    n = len(scribes)
    fig, axes = plt.subplots(n, 4, figsize=(9.5, n * 1.3 + 1.0), squeeze=False)
    for c in range(n):
        its = [it for it in items if it["scribe"] == c]
        for r in range(4):
            ax = axes[c, r]
            _clean(ax)
            if c == 0:
                ax.set_title(PANELS[r], fontsize=10, fontweight="bold", pad=4)
            if not its:
                continue
            it = its[0]
            rows, cols = it["evidence"].shape
            h, w = it["crop"].shape[:2]
            ax.imshow(it["crop"], extent=(0, w, h, 0), aspect="equal")
            if r > 0:
                rgba = np.zeros((rows, cols, 4), np.float32)
                rgba[top_mask(it[METHODS[r - 1][0]], TOP_SHOW)] = (1.0, 0.85, 0.0, 0.6)
                ax.imshow(rgba, extent=(0, w, h, 0), interpolation="nearest", aspect="equal")
        if its:
            it = its[0]
            ok = it["pred"] == c
            axes[c, 0].set_ylabel(f"written by S{scribes[c]}\nguess: S{scribes[it['pred']]} "
                                  f"({it['probs'][it['pred']] * 100:.0f}%)", rotation=0, ha="right", va="center", fontsize=9,
                                  color=GREEN if ok else RED, fontweight="bold", labelpad=8)
    fig.suptitle(f'Word "{word}" - {run} head: where the head looks (yellow = top {int(TOP_SHOW * 100)} % patches)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def accuracy_table(items: list[dict], words: list[str], n_scribes: int) -> tuple[np.ndarray, np.ndarray]:
    """(accuracy, count) arrays of shape (scribes, words); NaN accuracy where there is no example."""
    acc = np.full((n_scribes, len(words)), np.nan)
    cnt = np.zeros((n_scribes, len(words)), dtype=int)
    for wi, word in enumerate(words):
        for c in range(n_scribes):
            its = [it for it in items if it["word"] == word and it["scribe"] == c]
            cnt[c, wi] = len(its)
            if its:
                acc[c, wi] = float(np.mean([it["pred"] == c for it in its]))
    return acc, cnt


def confusion(items: list[dict], n_scribes: int) -> np.ndarray:
    m = np.zeros((n_scribes, n_scribes), dtype=int)
    for it in items:
        m[it["scribe"], it["pred"]] += 1
    return m


def _right_total(items: list[dict], words: list[str], n: int) -> tuple[np.ndarray, np.ndarray]:
    """(right, total) of shape (scribes + 1, words + 1): last row / column = over all scribes / all words."""
    right = np.zeros((n + 1, len(words) + 1), dtype=int)
    total = np.zeros_like(right)
    for it in items:
        c, w, ok = it["scribe"], words.index(it["word"]), int(it["pred"] == it["scribe"])
        for r_, c_ in ((c, w), (c, len(words)), (n, w), (n, len(words))):
            right[r_, c_] += ok
            total[r_, c_] += 1
    return right, total


def draw_accuracy(items: list[dict], words: list[str], scribes: list[int], run: str, path: Path) -> None:
    """One heatmap: how often the scribe is found from one word (large % + right/all)."""
    n = len(scribes)
    right, total = _right_total(items, words, n)
    pct = np.where(total > 0, 100.0 * right / np.maximum(total, 1), np.nan)
    labels = np.array([[f"{pct[i, j]:.0f}%\n{right[i, j]}/{total[i, j]}" if total[i, j] else "" for j in range(len(words) + 1)]
                       for i in range(n + 1)])
    fig, ax = plt.subplots(figsize=(9.5, 7))
    sns.heatmap(pct, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=100, linewidths=1, linecolor="white", ax=ax,
                annot_kws={"fontsize": 11}, cbar_kws={"label": "% of crops with the right scribe", "shrink": 0.8})
    ax.set_xticklabels([*words, "ALL WORDS"], fontsize=11)
    ax.set_yticklabels([*[f"S{s}" for s in scribes], "ALL SCRIBES"], rotation=0, fontsize=11)
    ax.axhline(n, color="black", lw=2)
    ax.axvline(len(words), color="black", lw=2)
    ax.set_xlabel("word", fontsize=11)
    ax.set_ylabel("scribe who wrote the word", fontsize=11)
    ax.set_title(f"{run} head: is the scribe found from ONE word?  (chance = {100 / n:.0f}%)", fontsize=12, loc="left")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def draw_confusion(items: list[dict], scribes: list[int], run: str, path: Path) -> None:
    """Rows = true scribe, columns = guessed scribe, cell = share of that scribe's crops."""
    n = len(scribes)
    cm = confusion(items, n)
    pct = 100.0 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
    labels = np.array([[f"{pct[i, j]:.0f}%" if cm[i, j] else "" for j in range(n)] for i in range(n)])
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    sns.heatmap(pct, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=100, linewidths=1, linecolor="white", ax=ax,
                annot_kws={"fontsize": 10}, cbar_kws={"label": "% of the scribe's crops", "shrink": 0.8})
    ax.set_xticklabels([f"S{s}" for s in scribes], fontsize=10)
    ax.set_yticklabels([f"S{s}" for s in scribes], rotation=0, fontsize=10)
    ax.set_xlabel("scribe the head guessed", fontsize=11)
    ax.set_ylabel("scribe who wrote the word", fontsize=11)
    ax.set_title(f"{run} head: who gets mistaken for whom (all 5 words together)", fontsize=11, loc="left")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def draw_per_scribe(items: list[dict], scribes: list[int], run: str, path: Path) -> None:
    """Bars: share of right crops per scribe, with chance and the mean over scribes."""
    n = len(scribes)
    acc = [100.0 * np.mean([it["pred"] == c for it in items if it["scribe"] == c]) for c in range(n)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = [sum(it["scribe"] == c for it in items) for c in range(n)]
    sns.barplot(x=[f"S{s} ({k} crops)" for s, k in zip(scribes, counts)], y=acc, color="#3b78b4", ax=ax)
    ax.tick_params(axis="x", labelsize=8)
    for k, v in enumerate(acc):
        ax.text(k, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)  # a 0 % bar is still a number, not missing data
    ax.axhline(100 / n, color=RED, ls="--", lw=1.5)
    ax.text(n - 0.5, 100 / n + 1.5, f"chance {100 / n:.0f}%", color=RED, ha="right", fontsize=9)
    ax.axhline(np.mean(acc), color="black", ls=":", lw=1.5)
    ax.text(-0.45, np.mean(acc) + 1.5, f"mean over scribes {np.mean(acc):.0f}%", fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of crops with the right scribe")
    ax.set_xlabel("scribe who wrote the word")
    ax.set_title(f"{run} head, single words: accuracy per scribe (5 words together)", fontsize=11, loc="left")
    sns.despine(ax=ax)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def overlap_by_word(items: list[dict], words: list[str]) -> dict:
    """Share of the top 20 % patches that lie on ink, per word and method (chance = ink share of all patches)."""
    out = {}
    for word in words:
        its = [it for it in items if it["word"] == word]
        if not its:
            continue
        s = ink_overlap(its)
        out[word] = {"chance": float(np.mean(s["chance_ink_share"])),
                     **{k: float(np.nanmean(s[k])) for k, *_ in METHODS}}
    return out

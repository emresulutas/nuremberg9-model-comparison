"""Figures and numbers for the blk16 maps (used by explain_blk16.py and redraw_blk16.py, no GPU needed).

A line item is a dict: crop (H x W x 3 uint8 array, W = 14 * patch columns, H = 14 * patch rows), evidence,
cam, attention (patch-grid maps, shape (rows, cols)), title, scribe (class index), kind ("correct"/"mistake").

Per method the figure shows the line with its TOP patches highlighted (yellow) and, right under it, the
plain map as a thin strip. The number version asks: of the top patches, what share lies on ink?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from PIL import Image             # noqa: E402

PATCH = 14
TOP_SHOW = 0.15      # share of patches highlighted in the pictures
TOP_STAT = 0.20      # share of patches used for the ink-overlap numbers
METHODS = (
    ("evidence", "coolwarm", True, "evidence: pushes towards the predicted scribe"),
    ("cam", "inferno", False, "Grad-CAM (block 8 -> blk16 head)"),
    ("attention", "viridis", False, "attention received (block 16)"),
)


def top_mask(m: np.ndarray, frac: float, positive_only: bool = True) -> np.ndarray:
    """Boolean mask of the highest `frac` of patches. Patches with a value of 0 or less are never selected
    (Grad-CAM is mostly 0 after its ReLU, so a plain threshold would select the whole line); the mask can
    therefore cover fewer than `frac` of the patches."""
    k = max(1, int(round(frac * m.size)))
    thr = np.partition(m.ravel(), -k)[-k]
    return (m >= thr) & (m > 0)


def ink_patches(crop: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Patches that hold ink: a pixel is ink when it is clearly darker than the paper (median grey);
    a patch counts when at least 10 % of its pixels are ink."""
    gray = np.asarray(Image.fromarray(crop).convert("L"), dtype=np.float32)
    ink = (gray < np.median(gray) - 40).astype(np.float32)
    frac = np.asarray(Image.fromarray(ink).resize((cols, rows), Image.BOX))
    return frac >= 0.10


def ink_overlap(items: list[dict], frac: float = TOP_STAT) -> dict:
    """Per method: share of the top patches that lie on ink, against the share of ink patches overall."""
    out: dict = {"top_fraction": frac, "chance_ink_share": [], **{k: [] for k, *_ in METHODS}, "scribe": []}
    for it in items:
        rows, cols = it["evidence"].shape
        ink = ink_patches(it["crop"], rows, cols)
        out["chance_ink_share"].append(float(ink.mean()))
        out["scribe"].append(int(it["scribe"]))
        for key, _, positive, _ in METHODS:
            out[key].append(float(ink[top_mask(it[key], frac, positive)].mean()) if top_mask(it[key], frac, positive).any() else float("nan"))
    return out


def draw_scribe(items: list[dict], scribe_name: str, run: str, path: Path) -> None:
    """Per line and method: the line with its top patches in yellow, a thin strip with the plain map below.
    Both keep the true proportions of the line (heights in inches follow the line's aspect ratio)."""
    heights = []
    for it in items:
        h, w = it["crop"].shape[:2]
        height = float(np.clip(12.4 * h / w, 0.55, 2.2))
        heights += [height, 0.3 * height] * len(METHODS)
    fig, axes = plt.subplots(len(heights), 1, figsize=(13, sum(heights) + 0.55 * len(heights) / 2 + 0.8),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    for n, it in enumerate(items):
        rows, cols = it["evidence"].shape
        h, w = it["crop"].shape[:2]
        for r, (key, cmap, positive, label) in enumerate(METHODS):
            idx = 2 * (len(METHODS) * n + r)
            ax, strip = axes[idx, 0], axes[idx + 1, 0]
            m = it[key]
            ax.imshow(it["crop"], extent=(0, w, h, 0))
            rgba = np.zeros((rows, cols, 4), np.float32)
            rgba[top_mask(m, TOP_SHOW, positive)] = (1.0, 0.85, 0.0, 0.55)
            ax.imshow(rgba, extent=(0, w, h, 0), interpolation="nearest")
            ax.axis("off")
            head = it["title"] + chr(10) if r == 0 else ""
            ax.set_title(f"{head}{label}: yellow = top {int(TOP_SHOW * 100)} % of patches (positive only)", fontsize=6.5, pad=2, loc="left")
            lim = max(float(np.abs(m).max()), 1e-12)
            strip.imshow(m, cmap=cmap, vmin=-lim if positive else 0, vmax=lim, extent=(0, w, 0.3 * h, 0),
                         interpolation="nearest")
            strip.axis("off")
    fig.suptitle(f"blk16 maps ({run} head), scribe {scribe_name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_overlap(stats: dict, scribe_names: list[str], run: str, path: Path) -> dict:
    """Bars: share of the top patches on ink, per method, overall and per scribe; dashed = chance."""
    scribe = np.array(stats["scribe"])
    summary = {"top_fraction": stats["top_fraction"], "chance": float(np.mean(stats["chance_ink_share"]))}
    for key, *_ in METHODS:
        summary[key] = float(np.nanmean(stats[key]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 3]})
    names = [lbl.split(":")[0].split(" (")[0] for _, _, _, lbl in METHODS]
    ax[0].bar(names, [summary[k] for k, *_ in METHODS], color=["#d95f02", "#7570b3", "#1b9e77"])
    ax[0].axhline(summary["chance"], color="grey", ls="--")
    ax[0].text(0.02, summary["chance"] + 0.01, "chance (ink share of all patches)", fontsize=7, color="grey", transform=ax[0].get_yaxis_transform())
    ax[0].set(ylim=(0, 1), ylabel="share of top patches on ink", title="all shown lines")
    width = 0.27
    for j, (key, *_rest) in enumerate(METHODS):
        vals = [np.nanmean(np.array(stats[key])[scribe == c]) if (scribe == c).any() else 0 for c in range(len(scribe_names))]
        ax[1].bar(np.arange(len(scribe_names)) + (j - 1) * width, vals, width, label=names[j], color=["#d95f02", "#7570b3", "#1b9e77"][j])
    chance = [np.mean(np.array(stats["chance_ink_share"])[scribe == c]) if (scribe == c).any() else 0 for c in range(len(scribe_names))]
    ax[1].plot(np.arange(len(scribe_names)), chance, "k_", markersize=26, label="chance")
    ax[1].set_xticks(range(len(scribe_names)), scribe_names)
    ax[1].set(ylim=(0, 1), title=f"per scribe ({run} head, top {int(stats['top_fraction'] * 100)} % patches)")
    ax[1].legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return summary


README = """# blk16 maps (evidence, Grad-CAM, attention)

Made by `nuremberg-7/experiments/01_churro_probe/explain_blk16.py`; redrawn without GPU by `redraw_blk16.py`.
One picture per scribe: `blk16_S<id>.png`. For every line three blocks (one per method). In each block the
line is shown with the **top 15 % of patches in yellow** ("the model looked / pushed here") and under it
the plain map as a thin strip (evidence: blue = towards the other scribe, red = towards the predicted one;
Grad-CAM and attention: dark = low, bright = high).

- **evidence** (exact): contribution of every patch to (predicted scribe minus runner-up / true scribe).
- **Grad-CAM**: gradient of the same score with respect to the output of block 8 (through blocks 9-16 and the
  head), channel-weighted, positive part only.
- **attention received**: how much attention block 16 pays to each patch on average (all queries, all heads).
  Independent of the head, identical for `plain` and `weighted`.

`ink_overlap.png` / `ink_overlap.json`: of the top 20 % patches of each map, what share lies on ink; the dashed
line is the share of ink patches in the whole line (what random patches would give).
`check.json`: numerical checks and the older `ink_correlation`. `maps.npz` + `maps_meta.json`: the raw maps
and line images, so figures can be redrawn on any machine.
Lines are the most confident correct / wrong test lines of the plain run, the same lines for both heads.
"""


def save_maps(out: Path, items: list[dict]) -> None:
    """Raw maps and line images (compressed), so that figures can be redrawn without the GPU."""
    arrays = {}
    for j, it in enumerate(items):
        arrays[f"crop_{j}"] = it["crop"]
        for key, *_ in METHODS:
            arrays[f"{key}_{j}"] = it[key]
    np.savez_compressed(out / "maps.npz", **arrays)
    import json
    (out / "maps_meta.json").write_text(json.dumps(
        [{k: it[k] for k in ("title", "scribe", "kind")} for it in items], indent=1), encoding="utf-8")


def load_maps(out: Path) -> list[dict]:
    import json
    data = np.load(out / "maps.npz")
    meta = json.loads((out / "maps_meta.json").read_text(encoding="utf-8"))
    return [{**m, "crop": data[f"crop_{j}"], **{k: data[f"{k}_{j}"] for k, *_ in METHODS}} for j, m in enumerate(meta)]


def render(out: Path, run: str, items: list[dict], scribes: list[int], check: dict) -> None:
    """All pictures and numbers of one run from the line items."""
    import json
    for c in range(len(scribes)):
        its = [it for it in items if it["scribe"] == c]
        if its:
            draw_scribe(its, f"S{scribes[c]}", run, out / f"blk16_S{scribes[c]}.png")
    stats = ink_overlap(items)
    summary = draw_overlap(stats, [f"S{s}" for s in scribes], run, out / "ink_overlap.png")
    (out / "ink_overlap.json").write_text(json.dumps({"summary": summary, **stats}, indent=1), encoding="utf-8")
    check = {**check, "ink_overlap_top20": summary}
    (out / "check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")

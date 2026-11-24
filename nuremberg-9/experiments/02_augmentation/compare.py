"""Plan step 4, part 4: without vs with augmentation over several seeds (mean +- std), from the saved metrics.

    python nuremberg-9/experiments/02_augmentation/compare.py
Reads results/seeds/{base,aug}_s<seed>/blk16_metrics.json, writes results/compare.json and results/compare.png.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402


def load(folder: Path, kind: str):
    runs = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(str(folder / f"{kind}_s*" / "blk16_metrics.json")))]
    if not runs:
        raise FileNotFoundError(f"no {kind}_s*/blk16_metrics.json in {folder}")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="nuremberg-9/experiments/02_augmentation/results")
    args = ap.parse_args()
    root = Path(args.dir)
    runs = {k: load(root / "seeds", k) for k in ("base", "aug")}
    scribes = runs["base"][0]["scribes"]
    support = runs["base"][0]["test"]["support"]
    out = {"scribes": scribes, "test_lines": support, "seeds": {k: len(v) for k, v in runs.items()}}
    for k, rs in runs.items():
        for m in ("balanced_accuracy", "macro_f1", "accuracy"):
            v = np.array([r["test"][m] for r in rs])
            out[f"{k}_{m}"] = [float(v.mean()), float(v.std())]
        rec = np.array([r["test"]["recall_per_scribe"] for r in rs])
        out[f"{k}_recall_mean"], out[f"{k}_recall_std"] = rec.mean(0).tolist(), rec.std(0).tolist()
    json.dump(out, open(root / "compare.json", "w", encoding="utf-8"), indent=1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8.5), gridspec_kw={"height_ratios": [1, 2.2]})
    names = ["balanced accuracy", "macro-F1", "accuracy"]
    for i, (k, c) in enumerate((("base", "#7f7f7f"), ("aug", "#2a7fbf"))):
        for j, m in enumerate(("balanced_accuracy", "macro_f1", "accuracy")):
            mu, sd = out[f"{k}_{m}"]
            ax[0].bar(j + (i - 0.5) * 0.38, mu, 0.36, yerr=sd, color=c, capsize=3, label=("without" if k == "base" else "with augmentation") if j == 0 else None)
            ax[0].text(j + (i - 0.5) * 0.38, mu + sd + 0.004, f"{mu:.3f}", ha="center", fontsize=9)
    ax[0].set(xticks=range(3), xticklabels=names, ylim=(0.8, 1.0), title="test, mean +- std over seeds (bars: std)")
    ax[0].legend(loc="lower right")
    x = np.arange(len(scribes))
    for i, (k, c) in enumerate((("base", "#7f7f7f"), ("aug", "#2a7fbf"))):
        mu, sd = np.array(out[f"{k}_recall_mean"]), np.array(out[f"{k}_recall_std"])
        ax[1].bar(x + (i - 0.5) * 0.38, mu, 0.36, yerr=sd, color=c, capsize=2)
        for xi, m in zip(x + (i - 0.5) * 0.38, mu):
            ax[1].text(xi, 0.03, f"{m:.2f}", ha="center", rotation=90, color="white", fontsize=8)
    ax[1].set(xticks=x, xticklabels=[f"S{s}\nn={n}" for s, n in zip(scribes, support)], ylim=(0, 1.05),
              ylabel="recall", title="recall per scribe (S10, S11, S12: very few test lines)")
    fig.tight_layout()
    fig.savefig(root / "compare.png", dpi=130)
    for k in ("base", "aug"):
        print(k, {m: [round(v, 3) for v in out[f"{k}_{m}"]] for m in ("balanced_accuracy", "macro_f1", "accuracy")})
    print(f"wrote {root / 'compare.png'}")


if __name__ == "__main__":
    main()

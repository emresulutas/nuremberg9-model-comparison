"""Ensemble of the line model and the word-vote model, tested properly.

    python nuremberg-9/experiments/04_error_overlap/ensemble.py

Ensemble = weighted mean of log-probabilities:  score = w * (word vote) + (1 - w) * (line model);  w = 0 is the line model alone, w = 1 the word vote alone.
Everything uses the 5 line-model seeds and 3 word-head seeds that already exist, no new training:
  - 15 pairs (every line seed with every word seed), so the result is a mean and a spread, not one lucky pair
  - weight grid 0, 0.05, ... 1: scored on validation and on test for every pair
  - the weight is picked on VALIDATION only (best balanced accuracy, ties go to the weight closest to 0.5), then scored on test
  - controls: line+line and word+word ensembles at weight 0.5 (a gain from mixing models of one kind is only seed averaging)
Figures and ensemble_summary.csv go to the experiment's results/ensemble directory by default.
The ensemble.json is saved directly under results.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import (C_LINE, C_WORD, ROOT, group_lines, line_probs, provenance, read_words, vote,   # noqa: E402
                     word_head_logp)

C_ENS, C_TUNED, C_CTRL = "#3a9d5d", "#1f6b3a", "#9aa5b4"


def bacc(y, p) -> float:
    return float(balanced_accuracy_score(y, p))


def mf1(y, p) -> float:
    return float(f1_score(y, p, average="macro", zero_division=0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="features/churro_v2_words")
    ap.add_argument("--split-json", default="nuremberg-9/docs/split_v2_9scribes.json")
    ap.add_argument("--line-seeds", default="nuremberg-9/experiments/02_augmentation/results/seeds/base_s{}/blk16_predictions.npz")
    ap.add_argument("--line-seed-ids", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--word-main", default="nuremberg-9/experiments/03_word_vs_line/results/words")
    ap.add_argument("--word-seeds", default="nuremberg-9/experiments/03_word_vs_line/results/words_s{}")
    ap.add_argument("--word-seed-ids", nargs="+", type=int, default=[43, 44])
    ap.add_argument("--step", type=float, default=0.05, help="weight grid step")
    ap.add_argument("--results", default="nuremberg-9/experiments/04_error_overlap/results")
    ap.add_argument("--out", help="Figure and table directory, defaults to the ensemble subdirectory of --results")
    args = ap.parse_args()
    res_dir = ROOT / args.results
    out = ROOT / args.out if args.out else res_dir / "ensemble"
    out.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    scribes = sorted(json.load(open(ROOT / args.split_json, encoding="utf-8"))["scribes"])
    K = len(scribes)
    names = [f"S{s}" for s in scribes]

    # ---------- data: the same lines for every model ----------
    W = {sp: read_words(ROOT / args.features, sp) for sp in ("val", "test")}
    groups = {sp: group_lines(W[sp]["page_id"], W[sp]["line_id"]) for sp in W}
    keys = {sp: list(groups[sp]) for sp in W}
    y = {sp: np.array([W[sp]["label"][groups[sp][k][0]] for k in keys[sp]]) for sp in W}
    heads = {42: ROOT / args.word_main / "head.npz"}
    heads.update({s: ROOT / args.word_seeds.format(s) / "head.npz" for s in args.word_seed_ids})
    WV = {s: {sp: vote(word_head_logp(W[sp], p), groups[sp], keys[sp]) for sp in W} for s, p in heads.items()}
    LP = {}
    for s in args.line_seed_ids:
        LP[s] = {}
        for sp in W:
            prob, yy = line_probs(ROOT / args.line_seeds.format(s), sp, keys[sp])
            assert (yy == y[sp]).all()
            LP[s][sp] = np.log(np.clip(prob, 1e-9, 1.0))
    yv, yt = y["val"], y["test"]
    print(f"lines: val {len(yv)}, test {len(yt)} | line seeds {list(LP)} | word seeds {list(WV)}")

    # ---------- weight grid for every (line seed, word seed) pair ----------
    ws = np.linspace(0, 1, int(round(1 / args.step)) + 1)
    half = int(np.argmin(np.abs(ws - 0.5)))
    pairs = [(i, j) for i in LP for j in WV]
    P, G = len(pairs), len(ws)
    val_b, test_b, val_f, test_f = (np.zeros((P, G)) for _ in range(4))
    for p, (i, j) in enumerate(pairs):
        for g, w in enumerate(ws):
            pv = (w * WV[j]["val"] + (1 - w) * LP[i]["val"]).argmax(1)
            pt = (w * WV[j]["test"] + (1 - w) * LP[i]["test"]).argmax(1)
            val_b[p, g], test_b[p, g] = bacc(yv, pv), bacc(yt, pt)
            val_f[p, g], test_f[p, g] = mf1(yv, pv), mf1(yt, pt)

    def pick(curve):                                  # best on validation, ties -> closest to 0.5
        return max(range(G), key=lambda g: (round(float(curve[g]), 6), -abs(ws[g] - 0.5)))

    g_pair = [pick(val_b[p]) for p in range(P)]
    g_global = pick(val_b.mean(0))
    tuned_b = np.array([test_b[p, g_pair[p]] for p in range(P)])
    tuned_f = np.array([test_f[p, g_pair[p]] for p in range(P)])
    print(f"weight on words picked on validation: global {ws[g_global]:.2f}; per pair min {ws[g_pair].min():.2f}, "
          f"median {np.median(ws[g_pair]):.2f}, max {ws[g_pair].max():.2f}")

    # ---------- single models and controls ----------
    line_pred = {i: LP[i]["test"].argmax(1) for i in LP}
    word_pred = {j: WV[j]["test"].argmax(1) for j in WV}
    cfg = {
        "line model (5 seeds)": [(bacc(yt, p), mf1(yt, p)) for p in line_pred.values()],
        "word vote (3 seeds)": [(bacc(yt, p), mf1(yt, p)) for p in word_pred.values()],
    }
    ll = [(LP[a]["test"] + LP[b]["test"]).argmax(1) for a, b in combinations(LP, 2)]
    ww = [(WV[a]["test"] + WV[b]["test"]).argmax(1) for a, b in combinations(WV, 2)]
    cfg["line + line, 50/50\n(control, 10 pairs)"] = [(bacc(yt, p), mf1(yt, p)) for p in ll]
    cfg["word + word, 50/50\n(control, 3 pairs)"] = [(bacc(yt, p), mf1(yt, p)) for p in ww]
    cfg["line + word, 50/50\n(15 pairs)"] = list(zip(test_b[:, half], test_f[:, half]))
    cfg["line + word, weight\npicked on val (15 pairs)"] = list(zip(tuned_b, tuned_f))
    colors = [C_LINE, C_WORD, C_CTRL, C_CTRL, C_ENS, C_TUNED]

    # ---------- figure 1: weight curve ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, (vb, tb, label) in zip(ax, ((val_b, test_b, "balanced accuracy"), (val_f, test_f, "macro-F1"))):
        a.fill_between(ws, tb.min(0), tb.max(0), color=C_ENS, alpha=0.18, label="test, min to max over 15 pairs")
        a.plot(ws, tb.mean(0), color=C_ENS, lw=2.2, label="test, mean of 15 pairs")
        a.plot(ws, vb.mean(0), color="#555555", lw=1.5, ls="--", label="validation, mean of 15 pairs")
        a.axvline(0.5, color="grey", lw=0.8, ls=":")
        a.scatter([ws[g_global]], [vb.mean(0)[g_global]], marker="*", s=180, color="black", zorder=5,
                  label=f"weight picked on validation: {ws[g_global]:.2f}")
        a.set(xlabel="weight on the word vote   (0 = line model only, 1 = word vote only)", ylabel=label,
              title=f"{label} against the ensemble weight")
        a.legend(fontsize=8, loc="lower center")
    fig.suptitle("How the CHURRO-3B ensemble weight changes scores")
    fig.tight_layout()
    fig.savefig(out / "weight_curve.png", dpi=130)
    plt.close(fig)

    # ---------- figure 2: single models vs ensembles ----------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, (a, label) in enumerate(zip(ax, ("balanced accuracy", "macro-F1"))):
        data = [[v[k] for v in vals] for vals in cfg.values()]
        means = [float(np.mean(d)) for d in data]
        a.bar(range(1, len(data) + 1), means, 0.6, color=colors, alpha=0.85)
        for x, (d, m) in enumerate(zip(data, means), start=1):
            a.text(x, m + 0.0008, f"{m:.3f} +- {np.std(d):.3f}", ha="center", va="bottom", fontsize=9, weight="bold")
        a.set_ylim(min(means) - 0.012, max(means) + 0.008)
        a.set_xticks(range(1, len(data) + 1))
        a.set_xticklabels([textwrap.fill(n.replace("\n", " "), 15) for n in cfg], fontsize=8)
        a.set(title=f"test {label}: mean over seeds (+- spread between seeds)", ylabel=label)
    fig.suptitle("CHURRO-3B · Whole-line and word-vote ensembles")
    fig.tight_layout()
    fig.savefig(out / "compare_bars.png", dpi=130)
    plt.close(fig)

    # ---------- figure 3: recall per scribe ----------
    labs = list(range(K))
    r_line = np.array([recall_score(yt, p, labels=labs, average=None, zero_division=0) for p in line_pred.values()])
    r_word = np.array([recall_score(yt, p, labels=labs, average=None, zero_division=0) for p in word_pred.values()])
    ens_pred = {}
    for p, (i, j) in enumerate(pairs):
        ens_pred[(i, j)] = (0.5 * WV[j]["test"] + 0.5 * LP[i]["test"]).argmax(1)
    r_ens = np.array([recall_score(yt, p, labels=labs, average=None, zero_division=0) for p in ens_pred.values()])
    n_test = np.bincount(yt, minlength=K)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(K)
    for off, arr, col, lab in ((-0.27, r_line, C_LINE, "line model (mean of 5 seeds)"),
                               (0.0, r_word, C_WORD, "word vote (mean of 3 seeds)"),
                               (0.27, r_ens, C_ENS, "line + word 50/50 (mean of 15 pairs)")):
        ax.bar(x + off, arr.mean(0), 0.26, yerr=arr.std(0), color=col, label=lab, capsize=2, error_kw={"lw": 0.8})
    ax.set(xticks=x, xticklabels=[f"{n}\nn={c}" for n, c in zip(names, n_test)], ylim=(0, 1.1), ylabel="recall",
           title="Recall per scribe on the test lines (bars: mean, whisker: spread over seeds; S10 to S12 have very few lines)")
    ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("CHURRO-3B · Whole-line and word-vote recall by writer")
    fig.tight_layout()
    fig.savefig(out / "per_scribe_recall.png", dpi=130)
    plt.close(fig)

    # ---------- figure 4: lines fixed and lines broken by the 50/50 ensemble ----------
    fx = {"vs line model": [], "vs word vote": []}
    bk = {"vs line model": [], "vs word vote": []}
    for (i, j), pe in ens_pred.items():
        for name, base in (("vs line model", line_pred[i]), ("vs word vote", word_pred[j])):
            fx[name].append(int(((base != yt) & (pe == yt)).sum()))
            bk[name].append(int(((base == yt) & (pe != yt)).sum()))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels, means, lo, hi, cols = [], [], [], [], []
    for name in fx:
        for kind, d, col in (("fixed", fx[name], C_ENS), ("broken", bk[name], "#c0504d")):
            labels.append(f"{kind}\n{name}")
            means.append(np.mean(d))
            lo.append(np.mean(d) - min(d))
            hi.append(max(d) - np.mean(d))
            cols.append(col)
    ax.bar(range(4), means, yerr=[lo, hi], color=cols, capsize=4)
    for i, m in enumerate(means):
        ax.text(i + 0.28, m, f"{m:.0f}", ha="left", va="center", fontsize=11, weight="bold")
    ax.set(xticks=range(4), xticklabels=labels, ylabel="test lines",
           title="What the 50/50 ensemble changes (mean over 15 pairs, whisker: min to max)\nfixed = was wrong, now right;  broken = was right, now wrong")
    fig.suptitle("CHURRO-3B · Errors changed by the whole-line and word-vote ensemble")
    fig.tight_layout()
    fig.savefig(out / "fixed_vs_broken.png", dpi=130)
    plt.close(fig)

    # ---------- numbers ----------
    rows = []
    for name, vals in cfg.items():
        a = np.array(vals)
        rows.append([name.replace("\n", " "), len(a), *[round(float(v), 4) for v in (a[:, 0].mean(), a[:, 0].std(), a[:, 0].min(), a[:, 0].max(),
                                                                                    a[:, 1].mean(), a[:, 1].std())]])
    with open(out / "ensemble_summary.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "n", "bAcc_mean", "bAcc_std", "bAcc_min", "bAcc_max", "macroF1_mean", "macroF1_std"])
        wr.writerows(rows)
    for r in rows:
        print(f"{r[0]:<44} n={r[1]:<3} bAcc {r[2]:.3f} +- {r[3]:.3f} (min {r[4]:.3f}, max {r[5]:.3f})   macro-F1 {r[6]:.3f} +- {r[7]:.3f}")
    print("fixed / broken (mean):", {k: round(float(np.mean(v)), 1) for k, v in fx.items()}, {k: round(float(np.mean(v)), 1) for k, v in bk.items()})
    print("recall per scribe, line / word / ens:")
    for n, a, b, c in zip(names, r_line.mean(0), r_word.mean(0), r_ens.mean(0)):
        print(f"  {n}: {a:.2f} / {b:.2f} / {c:.2f}")
    with open(res_dir / "ensemble.json", "w", encoding="utf-8") as fh:
        json.dump({"weights": ws.tolist(), "pairs": [list(p) for p in pairs], "val_bacc": val_b.tolist(), "test_bacc": test_b.tolist(),
                   "val_f1": val_f.tolist(), "test_f1": test_f.tolist(), "weight_picked_global": float(ws[g_global]),
                   "weight_picked_per_pair": [float(ws[g]) for g in g_pair], "summary_rows": rows,
                   "fixed": fx, "broken": bk, "scribes": scribes, "run": provenance(vars(args))}, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

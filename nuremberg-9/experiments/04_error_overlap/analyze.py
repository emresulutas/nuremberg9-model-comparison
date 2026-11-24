"""Next goal: do the word-vote model and the line model make the same mistakes? (are they complementary?)

    python nuremberg-9/experiments/04_error_overlap/analyze.py

Two models, same test lines (the lines that could be cut into words):
  line model   step-3 head on whole-line features (5 seeds are available from the augmentation study)
  word model   word head, the words of a line vote (mean log-probability); seeds 42, 43, 44
Figures, tables and overlap.json go to the experiment's results directory by default.

What is measured
  1 error sets      lines wrong for the line model (A), for the word model (B), A and B, neither; oracle (at least one right)
  2 similarity      Cohen's kappa of the predicted scribes, McNemar test, share of "both wrong" lines with the same wrong scribe
  3 line types      accuracy of both models by scribe, words per line, line-model confidence
  4 seed control    how often two seeds of the SAME model disagree, compared with word vs line
The ensemble itself (weight search, 15 seed pairs, controls) is in ensemble.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from scipy.special import log_softmax                    # noqa: E402
from scipy.stats import binomtest                                 # noqa: E402
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score   # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.common.runinfo import provenance                         # noqa: E402

LAYER = "blk16"
C_LINE, C_WORD = "#2b6cb0", "#dd8452"


# ---------- loading ----------
def read_words(folder: Path, split: str) -> dict:
    parts = [np.load(f, allow_pickle=False) for f in sorted(folder.glob(f"{split}_S*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in (LAYER, "label", "page_id", "line_id", "word")}


def group_lines(page_id, line_id) -> dict:
    groups: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(page_id.tolist(), line_id.tolist())):
        groups.setdefault(key, []).append(i)
    return groups


def word_head_logp(words: dict, head_path: Path) -> np.ndarray:
    h = np.load(head_path)
    z = ((words[LAYER].astype(np.float32) - h["mean"]) / h["std"]) @ h["W"].T + h["b"]
    return log_softmax(z, axis=1)


def vote(logp, groups, keys) -> np.ndarray:
    return np.stack([logp[groups[k]].mean(0) for k in keys])


def line_probs(path: Path, split: str, keys) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    where = {k: i for i, k in enumerate(zip(d[f"{split}_page_id"].tolist(), d[f"{split}_line_id"].tolist()))}
    rows = np.array([where[k] for k in keys])
    return d[f"{split}_prob"][rows], d[f"{split}_y"][rows]


# ---------- small helpers ----------
def scores(y, pred) -> dict:
    return {"accuracy": float((pred == y).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def table_png(path: Path, title: str, header: list, rows: list) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(header)), 0.42 * (len(rows) + 1) + 0.9))
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left")
    t = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.45)
    t.auto_set_column_width(list(range(len(header))))
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#d9e2ef")
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def terciles(v, fmt="{:.0f}"):
    e1, e2 = np.quantile(v, [1 / 3, 2 / 3])
    idx = np.digitize(v, [e1, e2])
    return idx, [f"low (<{fmt.format(e1)})", f"middle", f"high (>={fmt.format(e2)})"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="features/churro_v2_words")
    ap.add_argument("--split-json", default="nuremberg-9/docs/split_v2_9scribes.json")
    ap.add_argument("--line-main", default="nuremberg-9/experiments/01_balanced_probe/results/weighted_oversample/blk16_predictions.npz")
    ap.add_argument("--line-seeds", default="nuremberg-9/experiments/02_augmentation/results/seeds/base_s{}/blk16_predictions.npz")
    ap.add_argument("--line-seed-ids", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--word-main", default="nuremberg-9/experiments/03_word_vs_line/results/words")
    ap.add_argument("--word-seeds", default="nuremberg-9/experiments/03_word_vs_line/results/words_s{}",
                    help="folders of the extra word-head seeds (head.npz inside)")
    ap.add_argument("--word-seed-ids", nargs="+", type=int, default=[43, 44])
    ap.add_argument("--results", default="nuremberg-9/experiments/04_error_overlap/results")
    ap.add_argument("--out", help="Figure and table directory, defaults to --results")
    args = ap.parse_args()
    out, res_dir = ROOT / (args.out or args.results), ROOT / args.results
    out.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    scribes = sorted(json.load(open(ROOT / args.split_json, encoding="utf-8"))["scribes"])
    K = len(scribes)
    names = [f"S{s}" for s in scribes]

    # ---------- word side: head of each seed applied to the saved word features ----------
    feat = ROOT / args.features
    W = {sp: read_words(feat, sp) for sp in ("val", "test")}
    groups = {sp: group_lines(W[sp]["page_id"], W[sp]["line_id"]) for sp in W}
    keys = {sp: list(groups[sp]) for sp in W}
    y = {sp: np.array([W[sp]["label"][groups[sp][k][0]] for k in keys[sp]]) for sp in W}
    word_heads = {42: ROOT / args.word_main / "head.npz"}
    word_heads.update({s: ROOT / args.word_seeds.format(s) / "head.npz" for s in args.word_seed_ids})
    wvote = {s: {sp: vote(word_head_logp(W[sp], p), groups[sp], keys[sp]) for sp in W} for s, p in word_heads.items()}

    # ---------- line side, aligned to the same lines ----------
    L_main, L_val = {}, {}
    L_main["test"], y_chk = line_probs(ROOT / args.line_main, "test", keys["test"])
    assert (y_chk == y["test"]).all()
    L_main["val"], y_chk = line_probs(ROOT / args.line_main, "val", keys["val"])
    assert (y_chk == y["val"]).all()
    line_seed_probs = {s: line_probs(ROOT / args.line_seeds.format(s), "test", keys["test"])[0] for s in args.line_seed_ids}
    eps = 1e-9
    lp = {sp: np.log(np.clip(L_main[sp], eps, 1.0)) for sp in L_main}      # line model, log-probabilities
    yt, yv = y["test"], y["val"]
    n_lines = len(yt)
    print(f"test lines used: {n_lines} of 7636 (the rest could not be cut into words)")

    # ---------- 1 error sets (main pair: line seed 42 vs word seed 42) ----------
    pl, pw = lp["test"].argmax(1), wvote[42]["test"].argmax(1)
    el, ew = pl != yt, pw != yt
    both, only_l, only_w, neither = el & ew, el & ~ew, ew & ~el, ~el & ~ew
    b, c = int(only_l.sum()), int(only_w.sum())
    pval = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    oracle_pred = np.where(~el | ~ew, yt, pl)
    ov = {
        "n_lines": n_lines, "line_wrong": int(el.sum()), "word_wrong": int(ew.sum()),
        "both_wrong": int(both.sum()), "only_line_wrong": b, "only_word_wrong": c, "both_right": int(neither.sum()),
        "oracle_accuracy": float((~both).mean()), "oracle_balanced_accuracy": float(balanced_accuracy_score(yt, oracle_pred)),
        "kappa_predicted_scribe": float(cohen_kappa_score(pl, pw)),
        "mcnemar_p": pval,
        "error_jaccard": float(both.sum() / max(1, (el | ew).sum())),
        "both_wrong_same_wrong_scribe": float((pl[both] == pw[both]).mean()) if both.any() else None,
        "agree_on_scribe": float((pl == pw).mean()),
    }
    rows = [
        ["both right", ov["both_right"], f"{ov['both_right'] / n_lines:.1%}"],
        ["only the line model wrong", b, f"{b / n_lines:.1%}"],
        ["only the word model wrong", c, f"{c / n_lines:.1%}"],
        ["both wrong (intersection)", ov["both_wrong"], f"{ov['both_wrong'] / n_lines:.1%}"],
        ["line model wrong (set A)", ov["line_wrong"], f"{ov['line_wrong'] / n_lines:.1%}"],
        ["word model wrong (set B)", ov["word_wrong"], f"{ov['word_wrong'] / n_lines:.1%}"],
        ["at least one right (oracle accuracy)", n_lines - ov["both_wrong"], f"{ov['oracle_accuracy']:.1%}"],
        ["both wrong, same wrong scribe", int((pl[both] == pw[both]).sum()),
         f"{ov['both_wrong_same_wrong_scribe']:.0%} of both-wrong" if both.any() else "-"],
        ["Cohen's kappa (predicted scribe)", "", f"{ov['kappa_predicted_scribe']:.3f}"],
        ["McNemar p (only-A vs only-B)", "", f"{pval:.3f}"],
    ]
    table_png(out / "overlap_table.png", f"CHURRO-3B · Whole-line vs word-vote errors on {n_lines:,} test lines (seed 42)",
              ["", "lines", "share"], rows)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ra, rb = np.sqrt(ov["line_wrong"]), np.sqrt(ov["word_wrong"])
    scale = 1.0 / max(ra, rb)
    ra, rb = ra * scale, rb * scale
    gap = 0.75 * (ra + rb) if ov["both_wrong"] else ra + rb + 0.2
    ax.add_patch(plt.Circle((0, 0), ra, color=C_LINE, alpha=0.45))
    ax.add_patch(plt.Circle((gap, 0), rb, color=C_WORD, alpha=0.45))
    ax.text(-ra * 0.55, 0, f"only line\nwrong\n{b}", ha="center", va="center")
    ax.text(gap + rb * 0.55, 0, f"only word\nwrong\n{c}", ha="center", va="center")
    ax.text((gap + ra - rb) / 2, 0, f"both\n{ov['both_wrong']}", ha="center", va="center", weight="bold")
    ax.text(-ra, ra + 0.08, f"line model wrong: {ov['line_wrong']}", color=C_LINE, ha="left")
    ax.text(gap + rb, rb + 0.08, f"word model wrong: {ov['word_wrong']}", color=C_WORD, ha="right")
    ax.set(xlim=(-ra - 0.1, gap + rb + 0.1), ylim=(-1.15, 1.35), aspect="equal", title="CHURRO-3B · Whole-line vs word-vote errors (schematic)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "error_sets.png", dpi=130)
    plt.close(fig)

    # ---------- 3 line types ----------
    n_words = np.array([len(groups["test"][k]) for k in keys["test"]])
    conf = L_main["test"].max(1)
    panels = []
    panels.append(("scribe", np.array([names[i] for i in yt]), names, None))
    for label, v, fmt in (("words per line", n_words, "{:.0f}"), ("line-model confidence", conf, "{:.2f}")):
        idx, lab = terciles(v, fmt)
        panels.append((label, np.array([lab[i] for i in idx]), lab, None))
    csv_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (label, grp, order, _) in zip(axes.ravel(), panels):
        xs, accl, accw = [], [], []
        for j, g in enumerate(order):
            m = grp == g
            if not m.any():
                continue
            accl.append(float((~el[m]).mean()))
            accw.append(float((~ew[m]).mean()))
            xs.append(f"{g}\nn={int(m.sum())}")
            csv_rows.append([label, g, int(m.sum()), round(accl[-1], 4), round(accw[-1], 4),
                             int((el & ew)[m].sum()), int((el & ~ew)[m].sum()), int((ew & ~el)[m].sum())])
        pos = np.arange(len(xs))
        ax.bar(pos - 0.2, accl, 0.4, color=C_LINE, label="line model")
        ax.bar(pos + 0.2, accw, 0.4, color=C_WORD, label="word vote")
        ax.set(xticks=pos, xticklabels=xs, ylim=(0, 1.05), title=f"accuracy by {label}")
        ax.tick_params(axis="x", labelsize=7)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("CHURRO-3B · Whole-line vs word-vote accuracy by line type", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "by_line_type.png", dpi=120)
    plt.close(fig)
    with open(out / "by_line_type.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["grouped_by", "group", "n_lines", "acc_line_model", "acc_word_vote", "both_wrong", "only_line_wrong", "only_word_wrong"])
        wr.writerows(csv_rows)

    # ---------- 5 seed control ----------
    pred_all = {f"line s{s}": line_seed_probs[s].argmax(1) for s in line_seed_probs}
    pred_all.update({f"word s{s}": wvote[s]["test"].argmax(1) for s in wvote})
    kinds = {"line vs line (same model, other seed)": [], "word vs word (same model, other seed)": [],
             "word vs line": []}
    for a, b2 in combinations(pred_all, 2):
        ka, kb = a.split()[0], b2.split()[0]
        key = "word vs line" if ka != kb else f"{ka} vs {ka} (same model, other seed)"
        ea, eb = pred_all[a] != yt, pred_all[b2] != yt
        kinds[key].append((float((pred_all[a] != pred_all[b2]).mean()), float((ea & eb).sum() / max(1, (ea | eb).sum()))))
    seed_json = {k: {"pairs": len(v), "disagree_mean": float(np.mean([x[0] for x in v])) if v else None,
                     "error_jaccard_mean": float(np.mean([x[1] for x in v])) if v else None} for k, v in kinds.items()}
    result = {"overlap": ov, "seed_control": seed_json,
              "scribes": scribes, "line_seeds": args.line_seed_ids, "word_seeds": [42] + args.word_seed_ids,
              "run": provenance(vars(args))}
    with open(res_dir / "overlap.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)

    print(json.dumps(ov, indent=1))
    print(json.dumps(seed_json, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

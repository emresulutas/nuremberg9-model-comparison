"""Plan step 3: linear head on the frozen CHURRO features for 9 scribes, with imbalance handling.

    python nuremberg-9/experiments/01_balanced_probe/train_probe.py --device cpu

Features: the cache of the 7 old scribes (features/churro_v1) plus the new scribes S11 and S12
(features/churro_v2_new, made by nuremberg-9/scripts/run_extract_new_scribes.sh). Class numbers follow the
sorted scribe list of the split file: 1 2 6 7 8 9 10 11 12 -> 0..8.

Imbalance handling (TRAIN only, val and test are never touched):
  --weights   none | inverse | effective   class weights in the cross entropy. `effective` (default) is the
              class-balanced loss of Cui et al. 2019: weight ~ 1 / ((1 - beta^n) / (1 - beta)), softer than
              plain inverse frequency (S12 gets about 33 x the weight of S1 instead of 400 x at beta 0.999).
  --oversample none | sqrt | balanced   how often rare scribes are drawn per epoch. `sqrt` (default): the share
              of a scribe in the draws is proportional to sqrt(n), `balanced`: equal share for every scribe.
Both are on by default. The weights are computed from the counts after oversampling (expected draws per epoch),
so a rare scribe is not corrected twice. Augmentation is plan step 4 and not part of this script.

One run = one depth (default blk16, the winner of the 7-scribe study). Early stopping and the plateau scheduler
watch validation macro-F1 as before. S11 and S12 have only 14 and 7 validation lines, so that signal is noisy.
Outputs: <out>/<layer>_metrics.json, _head.npz, _predictions.npz, _curves.png, _recall.png, _confusion.png.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402
import torch.nn.functional as F          # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.common.runinfo import provenance          # noqa: E402

# reuse the loader and the metrics of the 7-scribe study instead of copying them (folder name has a hyphen)
_spec = importlib.util.spec_from_file_location(
    "probe7", ROOT / "nuremberg-7" / "experiments" / "01_churro_probe" / "train_probe.py")
probe7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe7)

THIN = 30        # test lines below this: the recall of that scribe is a very noisy estimate


def load_all(main: Path, extra: Path, split: str) -> tuple[dict, int]:
    a, da = probe7.load_split(main, split)
    b, db = probe7.load_split(extra, split)
    keys = [k for k in a if k in b]
    return {k: np.concatenate([a[k], b[k]]) for k in keys}, da + db


def class_weights(counts: torch.Tensor, mode: str, beta: float) -> torch.Tensor:
    """Per-class loss weights, scaled so that the average weight per training line is 1."""
    counts = counts.double()
    if mode == "none":
        return torch.ones_like(counts).float()
    if mode == "inverse":
        w = 1.0 / counts.clamp_min(1)
    else:
        w = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, dtype=torch.double), counts.clamp_min(1)))
    return (w * counts.sum() / (w * counts).sum()).float()


def draw_probs(y: torch.Tensor, counts: torch.Tensor, mode: str) -> torch.Tensor | None:
    """Per-line draw probability: class share ~ n^q (q = 1 plain, 0.5 sqrt, 0 balanced), split evenly inside the class."""
    if mode == "none":
        return None
    q = {"sqrt": 0.5, "balanced": 0.0}[mode]
    p = counts.double().clamp_min(1)[y] ** (q - 1.0)
    return (p / p.sum()).float()


def plot_recall(recall, support, scribes, path, title):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [f"S{s}\nn={n}" for s, n in zip(scribes, support)]
    colors = ["#c0504d" if n < THIN else "#4f81bd" for n in support]
    bars = ax.bar(labels, recall, color=colors)
    for b, r in zip(bars, recall):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.015, f"{r:.2f}", ha="center", fontsize=10)
    ax.set(ylim=(0, 1.1), ylabel="recall", title=title)
    ax.set_xlabel(f"red bars: fewer than {THIN} test lines, very noisy estimate", color="#c0504d")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, scribes, path, title):
    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    names = [f"S{s}" for s in scribes]
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
           xlabel="predicted", ylabel="true", title=title)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="share of the true scribe's lines")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_layer(layer, data, scribes, args, out_dir: Path) -> dict:
    n_classes = len(scribes)
    dev = args.device
    xs = {sp: torch.tensor(data[sp][layer].astype(np.float32)) for sp in ("train", "val", "test")}
    ys = {sp: torch.tensor(data[sp]["label"]).long() for sp in ("train", "val", "test")}
    mean, std = xs["train"].mean(0), xs["train"].std(0).clamp_min(1e-6)         # statistics from train only
    xs = {sp: ((x - mean) / std).to(dev) for sp, x in xs.items()}
    counts = torch.bincount(ys["train"], minlength=n_classes)
    ys = {sp: y.to(dev) for sp, y in ys.items()}
    probs = draw_probs(ys["train"].cpu(), counts, args.oversample)
    # with oversampling the loss only has to fix what sampling left over: weights use the expected drawn counts
    # (otherwise a rare scribe is corrected twice and its few lines get memorised)
    drawn = counts.double() if probs is None else torch.bincount(ys["train"].cpu(), weights=probs.double(), minlength=n_classes) * len(probs)
    weight = class_weights(drawn, args.weights, args.beta).to(dev)
    probs = probs.to(dev) if probs is not None else None
    print(f"train lines per class:  {counts.tolist()}")
    print(f"expected draws / epoch: {[round(d) for d in drawn.tolist()]}")
    print(f"class weights:          {[round(w, 2) for w in weight.tolist()]}")

    torch.manual_seed(args.seed)
    model = torch.nn.Linear(xs["train"].shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=args.lr_patience)
    history = {"train_loss": [], "val_loss": [], "val_balanced_acc": [], "val_macro_f1": [], "lr": []}
    best = {"val_f1": -1.0, "epoch": 0, "state": None}
    n_train = len(ys["train"])
    for epoch in range(1, args.epochs + 1):
        model.train()
        if probs is None:
            order = torch.randperm(n_train, device=dev)
        else:
            order = torch.multinomial(probs, n_train, replacement=True)           # rare scribes drawn more often
        total = 0.0
        for i in range(0, n_train, args.batch):
            idx = order[i:i + args.batch]
            loss = F.cross_entropy(model(xs["train"][idx]), ys["train"][idx], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_logits = model(xs["val"])
            val_loss = F.cross_entropy(val_logits, ys["val"]).item()
        val_pred = val_logits.argmax(1).cpu().numpy()
        val_true = ys["val"].cpu().numpy()
        history["train_loss"].append(total / n_train)
        history["val_loss"].append(val_loss)
        history["val_balanced_acc"].append(float(balanced_accuracy_score(val_true, val_pred)))
        history["val_macro_f1"].append(float(f1_score(val_true, val_pred, average="macro", zero_division=0)))
        history["lr"].append(opt.param_groups[0]["lr"])
        val_f1 = history["val_macro_f1"][-1]
        sched.step(val_f1)
        if val_f1 > best["val_f1"] + 1e-4:
            best = {"val_f1": val_f1, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= args.stop_patience:
            break

    model.load_state_dict(best["state"])
    model.eval()
    result = {"layer": layer, "best_epoch": best["epoch"], "epochs_run": epoch, "history": history,
              "train_lines_per_class": counts.tolist(), "class_weights": weight.tolist(),
              "scribes": scribes}
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for sp in ("val", "test"):
        with torch.no_grad():
            prob = F.softmax(model(xs[sp]), dim=1).cpu().numpy()
        y = ys[sp].cpu().numpy()
        result[sp] = probe7.metrics(y, prob.argmax(1), n_classes)
        result[sp]["support"] = np.bincount(y, minlength=n_classes).tolist()
        saved.update({f"{sp}_prob": prob.astype(np.float32), f"{sp}_y": y,
                      f"{sp}_page_id": data[sp]["page_id"], f"{sp}_line_id": data[sp]["line_id"]})
    np.savez_compressed(out_dir / f"{layer}_predictions.npz", **saved)
    np.savez(out_dir / f"{layer}_head.npz", W=model.weight.detach().cpu().numpy(),
             b=model.bias.detach().cpu().numpy(), mean=mean.numpy(), std=std.numpy())

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"CHURRO-3B · Balanced Probe training · Whole lines · {layer}")
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].axvline(best["epoch"] - 1, color="grey", ls=":")
    ax[0].set(title=f"{layer}: loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(history["val_balanced_acc"], label="val balanced acc")
    ax[1].plot(history["val_macro_f1"], label="val macro-F1")
    ax[1].set(title=f"{layer}: validation", xlabel="epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{layer}_curves.png", dpi=120)
    plt.close(fig)
    t = result["test"]
    plot_recall(t["recall_per_scribe"], t["support"], scribes, out_dir / f"{layer}_recall.png",
                f"CHURRO-3B · Whole-line recall by writer · {layer} (balanced accuracy {t['balanced_accuracy']:.3f})")
    plot_confusion(t["confusion_matrix"], scribes, out_dir / f"{layer}_confusion.png",
                   f"CHURRO-3B · Whole-line test confusion matrix · {layer}")
    with open(out_dir / f"{layer}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="features/churro_v1", help="cache of the 7 old scribes")
    ap.add_argument("--extra-features", default="features/churro_v2_new", help="cache of S11 and S12")
    ap.add_argument("--aug-features", default=None,
                    help="folder with features of augmented copies (train only), e.g. features/churro_v2_aug")
    ap.add_argument("--split-json", default="nuremberg-9/docs/split_v2_9scribes.json")
    ap.add_argument("--out", default="nuremberg-9/experiments/01_balanced_probe/results/weighted_oversample")
    ap.add_argument("--layers", nargs="+", default=["blk16"])
    ap.add_argument("--weights", default="effective", choices=["none", "inverse", "effective"])
    ap.add_argument("--beta", type=float, default=0.999)
    ap.add_argument("--oversample", default="sqrt", choices=["none", "sqrt", "balanced"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr-patience", type=int, default=3)
    ap.add_argument("--stop-patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    split_info = json.load(open(args.split_json, encoding="utf-8"))
    scribes = sorted(split_info["scribes"])
    data, dropped = {}, {}
    for sp in ("train", "val", "test"):
        data[sp], dropped[sp] = load_all(Path(args.features), Path(args.extra_features), sp)
        have = sorted(set(data[sp]["label"].tolist()))
        assert have == list(range(len(scribes))), f"{sp}: class numbers {have} do not match {len(scribes)} scribes"
        if sp == "train" and args.aug_features:                       # augmented copies go to TRAIN only
            aug, _ = probe7.load_split(Path(args.aug_features), "train")
            data[sp] = {k: np.concatenate([data[sp][k], aug[k]]) for k in data[sp] if k in aug}
            print(f"train: added {len(aug['label'])} augmented copies")
        print(f"{sp}: {len(data[sp]['label'])} lines")

    out_dir = Path(args.out)
    summary = {}
    for layer in args.layers:
        res = run_layer(layer, data, scribes, args, out_dir)
        summary[layer] = {"best_epoch": res["best_epoch"],
                          **{f"{sp}_{m}": res[sp][m] for sp in ("val", "test") for m in ("balanced_accuracy", "macro_f1", "accuracy")}}
        t = res["test"]
        print(f"\n{layer}: best epoch {res['best_epoch']}   test accuracy {t['accuracy']:.3f}   "
              f"balanced {t['balanced_accuracy']:.3f}   macro-F1 {t['macro_f1']:.3f}")
        print("test recall per scribe (n test lines):")
        for s, r, n in zip(scribes, t["recall_per_scribe"], t["support"]):
            print(f"  S{s:<3} {r:.3f}  (n={n}){'   NOISY' if n < THIN else ''}")
    base = split_info.get("band_shortcut_baseline", {})
    if base:
        print(f"band shortcut baseline: val bAcc {base['val']['balanced_accuracy']:.3f}  "
              f"test bAcc {base['test']['balanced_accuracy']:.3f}")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({"scribes": scribes, "summary": summary, "band_shortcut_baseline": base,
                   "dropped_nonfinite_lines": dropped, "run": provenance(vars(args))}, fh, indent=1)
    print(f"\nwrote results to {out_dir}")


if __name__ == "__main__":
    main()

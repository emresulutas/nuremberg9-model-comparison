"""Plan step 5, part 2: linear head on single-word features, compared with the line-based head of step 3.

    python nuremberg-9/experiments/03_word_vs_line/train_words.py --device cpu

Same recipe as step 3 (blk16, effective-number weights computed from expected draws, sqrt oversampling, AdamW,
plateau scheduler, early stopping on validation macro-F1), only the unit is a word instead of a line.
Reported on the test split:
  word level   every word is one sample (this is what the word-based model sees)
  line level   the words of a line vote: mean of their log-probabilities, argmax (on the lines that could be cut)
  line head    the step-3 line-based head on exactly the same lines (its saved test probabilities), for a fair comparison
Outputs: <out>/metrics.json, head.npz, predictions.npz, churro-3b-word-vs-line.png.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np                       # noqa: E402
import torch                             # noqa: E402
import torch.nn.functional as F          # noqa: E402
from sklearn.metrics import f1_score     # noqa: E402

from plot_comparison import plot_comparison

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.common.runinfo import provenance          # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe9 = _load("probe9", "nuremberg-9/experiments/01_balanced_probe/train_probe.py")   # weights, oversampling
probe7 = probe9.probe7                                                                  # metrics
LAYER = "blk16"
KEYS = (LAYER, "label", "page_id", "line_id", "token", "word")


def load_words(folder: Path, split: str) -> tuple[dict, int, int]:
    parts = [np.load(f, allow_pickle=False) for f in sorted(folder.glob(f"{split}_S*.npz"))]
    data = {k: np.concatenate([p[k] for p in parts]) for k in KEYS}
    return data, int(sum(p["lines_seen"] for p in parts)), int(sum(p["lines_cut"] for p in parts))


def line_vote(logprob: np.ndarray, page_id, line_id, y) -> tuple[np.ndarray, np.ndarray, list]:
    """Mean log-probability of the words of each line -> (line scores, line labels, line keys)."""
    groups: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(page_id.tolist(), line_id.tolist())):
        groups.setdefault(key, []).append(i)
    keys = list(groups)
    return (np.stack([logprob[groups[k]].mean(0) for k in keys]), np.array([y[groups[k][0]] for k in keys]), keys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="features/churro_v2_words")
    ap.add_argument("--split-json", default="nuremberg-9/docs/split_v2_9scribes.json")
    ap.add_argument("--line-predictions",
                    default="nuremberg-9/experiments/01_balanced_probe/results/weighted_oversample/blk16_predictions.npz")
    ap.add_argument("--out", default="nuremberg-9/experiments/03_word_vs_line/results/words")
    ap.add_argument("--weights", default="effective", choices=["none", "inverse", "effective"])
    ap.add_argument("--beta", type=float, default=0.999)
    ap.add_argument("--oversample", default="sqrt", choices=["none", "sqrt", "balanced"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr-patience", type=int, default=3)
    ap.add_argument("--stop-patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    scribes = sorted(json.load(open(args.split_json, encoding="utf-8"))["scribes"])
    n_classes, dev = len(scribes), args.device
    data, cut = {}, {}
    for sp in ("train", "val", "test"):
        data[sp], seen, got = load_words(Path(args.features), sp)
        cut[sp] = {"lines_seen": seen, "lines_cut": got, "words": len(data[sp]["label"])}
        print(f"{sp}: {cut[sp]}")
    xs = {sp: torch.tensor(data[sp][LAYER].astype(np.float32)) for sp in data}
    ys = {sp: torch.tensor(data[sp]["label"]).long() for sp in data}
    mean, std = xs["train"].mean(0), xs["train"].std(0).clamp_min(1e-6)
    xs = {sp: ((x - mean) / std).to(dev) for sp, x in xs.items()}
    counts = torch.bincount(ys["train"], minlength=n_classes)
    probs = probe9.draw_probs(ys["train"], counts, args.oversample)
    drawn = counts.double() if probs is None else torch.bincount(ys["train"], weights=probs.double(), minlength=n_classes) * len(probs)
    weight = probe9.class_weights(drawn, args.weights, args.beta).to(dev)
    ys = {sp: y.to(dev) for sp, y in ys.items()}
    probs = probs.to(dev) if probs is not None else None
    print("train words per class:", counts.tolist(), "| weights:", [round(w, 2) for w in weight.tolist()])

    torch.manual_seed(args.seed)
    model = torch.nn.Linear(xs["train"].shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=args.lr_patience)
    best = {"f1": -1.0, "epoch": 0, "state": None}
    n_train = len(ys["train"])
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n_train, device=dev) if probs is None else torch.multinomial(probs, n_train, replacement=True)
        for i in range(0, n_train, args.batch):
            idx = order[i:i + args.batch]
            loss = F.cross_entropy(model(xs["train"][idx]), ys["train"][idx], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xs["val"]).argmax(1).cpu().numpy()
        f1 = float(f1_score(ys["val"].cpu().numpy(), pred, average="macro", zero_division=0))
        sched.step(f1)
        if f1 > best["f1"] + 1e-4:
            best = {"f1": f1, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= args.stop_patience:
            break
    model.load_state_dict(best["state"])
    model.eval()
    print(f"best epoch {best['epoch']} (val macro-F1 {best['f1']:.3f})")

    with torch.no_grad():
        logp = F.log_softmax(model(xs["test"]), dim=1).cpu().numpy()
    y_word = ys["test"].cpu().numpy()
    res = {"best_epoch": best["epoch"], "words_per_split": cut, "scribes": scribes}
    res["word_level"] = probe7.metrics(y_word, logp.argmax(1), n_classes)
    res["word_level"]["support"] = np.bincount(y_word, minlength=n_classes).tolist()
    score, y_line, keys = line_vote(logp, data["test"]["page_id"], data["test"]["line_id"], y_word)
    res["line_vote"] = probe7.metrics(y_line, score.argmax(1), n_classes)
    res["line_vote"]["support"] = np.bincount(y_line, minlength=n_classes).tolist()

    lp = np.load(args.line_predictions, allow_pickle=False)           # step-3 line head on the same lines
    where = {k: i for i, k in enumerate(zip(lp["test_page_id"].tolist(), lp["test_line_id"].tolist()))}
    rows = np.array([where[k] for k in keys])
    res["line_head_same_lines"] = probe7.metrics(lp["test_y"][rows], lp["test_prob"][rows].argmax(1), n_classes)
    res["line_head_same_lines"]["support"] = np.bincount(lp["test_y"][rows], minlength=n_classes).tolist()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "head.npz", W=model.weight.detach().cpu().numpy(), b=model.bias.detach().cpu().numpy(),
             mean=mean.numpy(), std=std.numpy())
    np.savez_compressed(out / "predictions.npz", test_logp=logp.astype(np.float32), test_y=y_word,
                        test_page_id=data["test"]["page_id"], test_line_id=data["test"]["line_id"],
                        test_token=data["test"]["token"], test_word=data["test"]["word"])
    report = {**res, "run": provenance(vars(args))}
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    names = {"word_level": "words (word head)", "line_vote": "lines (words vote)", "line_head_same_lines": "lines (line head)"}
    for k, label in names.items():
        m = res[k]
        print(f"{label:<22} accuracy {m['accuracy']:.3f}  balanced {m['balanced_accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}  n={sum(m['support'])}")
    plot_comparison(report, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Page-disjoint, scribe-stratified train/val/test split for the Nuremberg Letterbooks.

Why by page: all lines of one page go to the same split, so a model cannot memorise a page.
Many pages hold several scribes, so the split balances the *line counts of every scribe*
across train/val/test (greedy, rarest scribes first) instead of balancing pages.

It also prints a "band shortcut" baseline: scribes are strongly tied to the books (Band2..5),
so a model could cheat by recognising the book (paper, scan) instead of the hand. This baseline
predicts, for every test line, the scribe that wrote most lines of that band in train. A real
model must score clearly above it.

Two ways to add scribes (plan step 2, 9 scribes = the 7 of split_v1 + S11 + S12):
  fresh  every page is placed again from zero (v1 pages may move to another split)
  fixed  pages already in a base split (split_v1) stay where they are; only new pages are placed

Chosen for the 9-scribe study (mode lines): S11 and S12 are so rare (3 and 5 pages) that they are split
line by line (each line in exactly one split); the 7 other scribes keep the page split of split_v1:
    python -m src.data.nuremberg_split --mode lines --scribes 1 2 6 7 8 9 10 11 12 --line-scribes 11 12 \
        --base-split nuremberg-7/docs/split_v1.json --out-dir nuremberg-9/docs     # -> split_v2_9scribes.json

Run from the repo root (7 scribes, as before):
    python -m src.data.nuremberg_split --mode fresh --scribes 1 2 6 7 8 9 10 --out-dir nuremberg-7/docs
9 scribes, both ways side by side (writes split_v2a_fresh.json, split_v2b_fixed.json, split_v2_compare.md):
    python -m src.data.nuremberg_split --mode both --scribes 1 2 6 7 8 9 10 11 12 \
        --base-split nuremberg-7/docs/split_v1.json --out-dir nuremberg-9/docs
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .nuremberg import iter_pages

DEFAULT_SCRIBES = (1, 2, 6, 7, 8, 9, 10)     # scribes with at least 150 lines (see docs/dataset_notes.md)
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
THIN_LINES = 30      # a val/test cell with fewer lines than this gives a very noisy estimate


def page_table(root: str, version: str, scribes) -> list[dict]:
    """One row per page that holds at least one line of a selected scribe."""
    keep = set(scribes)
    rows = []
    for page in iter_pages(root, version):
        counts = Counter(l.writer for l in page.lines if l.writer in keep)
        if counts:
            rows.append({"page_id": page.page_id, "band": page.band, "counts": dict(counts)})
    return rows


def make_split(rows: list[dict], scribes, fractions=FRACTIONS, seed: int = 42,
               fixed: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Greedy split. `fixed` maps page_id -> split for pages that must stay where they are."""
    rng = random.Random(seed)
    fixed = fixed or {}
    totals = Counter()
    pages_of = Counter()
    for r in rows:
        for s, c in r["counts"].items():
            totals[s] += c
            pages_of[s] += 1
    target = {sp: {s: fractions[sp] * totals[s] for s in scribes} for sp in fractions}
    current = {sp: Counter() for sp in fractions}
    split: dict[str, list[str]] = {sp: [] for sp in fractions}
    free = []
    for r in rows:
        if r["page_id"] in fixed:
            sp = fixed[r["page_id"]]
            split[sp].append(r["page_id"])
            for s, c in r["counts"].items():
                current[sp][s] += c
        else:
            free.append(r)
    # rarest scribes first: they are the hardest to place
    order = sorted(free, key=lambda r: (min(pages_of[s] for s in r["counts"]), rng.random()))
    for r in order:
        gains = {}
        for sp in fractions:
            gains[sp] = sum(c / totals[s] * (target[sp][s] - current[sp][s]) / max(target[sp][s], 1.0)
                            for s, c in r["counts"].items())
        best = max(gains.values())
        choice = rng.choice([sp for sp, g in gains.items() if abs(g - best) < 1e-12])
        split[choice].append(r["page_id"])
        for s, c in r["counts"].items():
            current[choice][s] += c
    return split


def summarise(rows, split, scribes) -> dict:
    by_id = {r["page_id"]: r for r in rows}
    out = {}
    for sp, ids in split.items():
        lines, pages = Counter(), defaultdict(set)
        for pid in ids:
            for s, c in by_id[pid]["counts"].items():
                lines[s] += c
                pages[s].add(pid)
        out[sp] = {s: {"lines": lines[s], "pages": len(pages[s])} for s in scribes}
    return out


def band_shortcut_baseline(rows, split, scribes) -> dict:
    """Accuracy (line level) of 'predict the scribe that wrote most lines of this band in train'."""
    by_id = {r["page_id"]: r for r in rows}
    band_scribe = defaultdict(Counter)
    for pid in split["train"]:
        for s, c in by_id[pid]["counts"].items():
            band_scribe[by_id[pid]["band"]][s] += c
    majority = {b: cnt.most_common(1)[0][0] for b, cnt in band_scribe.items()}
    res = {}
    for sp in ("val", "test"):
        correct = total = 0
        recall = Counter()
        n = Counter()
        for pid in split[sp]:
            b = by_id[pid]["band"]
            for s, c in by_id[pid]["counts"].items():
                total += c
                n[s] += c
                if majority.get(b) == s:
                    correct += c
                    recall[s] += c
        macro_recall = sum(recall[s] / n[s] for s in scribes if n[s]) / max(1, sum(1 for s in scribes if n[s]))
        res[sp] = {"line_accuracy": round(correct / total, 4), "balanced_accuracy": round(macro_recall, 4)}
    res["band_majority_scribe"] = majority
    return res


def moved_pages(base: dict[str, str], split: dict[str, list[str]]) -> int:
    """How many pages of the base split ended in a different split."""
    where = {pid: sp for sp, ids in split.items() for pid in ids}
    return sum(1 for pid, sp in base.items() if pid in where and where[pid] != sp)


def line_key(page_id: str, line_id: str) -> str:
    return f"{page_id}|{line_id}"


def collect_lines(root: str, version: str, scribes) -> dict[int, list[str]]:
    """{scribe: sorted line keys} for these scribes, over all pages."""
    out = {s: [] for s in scribes}
    for page in iter_pages(root, version):
        for l in page.lines:
            if l.writer in out:
                out[l.writer].append(line_key(page.page_id, l.line_id))
    for s, keys in out.items():
        assert len(keys) == len(set(keys)), f"duplicate line keys for scribe {s}"
        keys.sort()
    return out


def assign_lines(keys: list[str], scribe: int, fractions=FRACTIONS, seed: int = 42) -> dict[str, list[str]]:
    """Every line gets exactly one split (random, 70/15/15, at least one line in val and test)."""
    rng = random.Random(f"{seed}-{scribe}")
    keys = list(keys)
    rng.shuffle(keys)
    n_val = max(1, round(fractions["val"] * len(keys)))
    n_test = max(1, round(fractions["test"] * len(keys)))
    return {"train": keys[n_val + n_test:], "val": keys[:n_val], "test": keys[n_val:n_val + n_test]}


def line_level_rows(root, version, scribes, line_scribes, base_where, overrides):
    """Pseudo pages (split::page) with the real per-split counts, so summarise() and the band
    baseline can be reused: scribes in `line_scribes` follow their line, the others follow the page."""
    lookup = {k: sp for per in overrides.values() for sp, ks in per.items() for k in ks}
    rows, split = [], {sp: [] for sp in FRACTIONS}
    for page in iter_pages(root, version):
        per = {sp: Counter() for sp in FRACTIONS}
        for l in page.lines:
            if l.writer not in scribes:
                continue
            if l.writer in line_scribes:
                sp = lookup.get(line_key(page.page_id, l.line_id))
            else:
                sp = base_where.get(page.page_id)
            if sp:
                per[sp][l.writer] += 1
        for sp, c in per.items():
            if c:
                pid = f"{sp}::{page.page_id}"
                rows.append({"page_id": pid, "band": page.band, "counts": dict(c)})
                split[sp].append(pid)
    return rows, split


def run_lines(args) -> None:
    """Pages of the base split stay as they are; the line_scribes are split line by line."""
    with open(args.base_split, encoding="utf-8") as fh:
        base = json.load(fh)
    base_where = {pid: sp for sp, ids in base["splits"].items() for pid in ids}
    line_scribes = set(args.line_scribes)
    keys = collect_lines(args.root, args.version, sorted(line_scribes))
    overrides = {str(s): assign_lines(keys[s], s, seed=args.seed) for s in sorted(line_scribes)}
    for s in sorted(line_scribes):
        print(f"S{s}: {len(keys[s])} lines -> " + ", ".join(f"{sp} {len(v)}" for sp, v in overrides[str(s)].items()))
    rows, split = line_level_rows(args.root, args.version, args.scribes, line_scribes, base_where, overrides)
    res = describe("lines", rows, split, args.scribes)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.tag}_9scribes.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"version": base.get("version", args.version), "mode": "lines", "scribes": args.scribes,
                   "seed": args.seed, "fractions": FRACTIONS, "base_split": args.base_split,
                   "line_scribes": sorted(line_scribes), "splits": base["splits"],
                   "line_overrides": overrides, "summary": res["summary"],
                   "band_shortcut_baseline": res["band_shortcut_baseline"],
                   "empty_cells": res["empty_cells"], "thin_cells": res["thin_cells"]}, fh, indent=1)
    print(f"wrote {path}")


def describe(name, rows, split, scribes, base_where=None) -> dict:
    summ = summarise(rows, split, scribes)
    base = band_shortcut_baseline(rows, split, scribes)
    empty = [(sp, s) for sp in split for s in scribes if summ[sp][s]["lines"] == 0]
    thin = [(sp, s, summ[sp][s]["lines"]) for sp in ("val", "test") for s in scribes
            if 0 < summ[sp][s]["lines"] < THIN_LINES]
    print(f"\n=== {name} ===")
    print(f"{'scribe':>6} | " + " | ".join(f"{sp:^15}" for sp in split))
    print(f"{'':>6} | " + " | ".join(f"{'lines':>7} {'pages':>7}" for _ in split))
    for s in scribes:
        print(f"{s:>6} | " + " | ".join(f"{summ[sp][s]['lines']:>7} {summ[sp][s]['pages']:>7}" for sp in split))
    print("EMPTY CELLS (scribe missing from a split):", empty or "none")
    print(f"THIN val/test cells (< {THIN_LINES} lines):", thin or "none")
    for sp in ("val", "test"):
        print(f"band shortcut {sp}: line accuracy {base[sp]['line_accuracy']:.3f}   "
              f"balanced accuracy {base[sp]['balanced_accuracy']:.3f}")
    print("main scribe per band in train:", base["band_majority_scribe"])
    moved = moved_pages(base_where, split) if base_where else None
    if moved is not None:
        print(f"pages of the base split now in another split: {moved} of {len(base_where)}")
    return {"summary": summ, "band_shortcut_baseline": base, "empty_cells": empty,
            "thin_cells": thin, "moved_from_base": moved}


def compare_markdown(scribes, results: dict) -> str:
    L = ["# Split v2 (9 scribes): fresh vs fixed", "",
         "Made by `src/data/nuremberg_split.py`. Lines (pages) per scribe and split. "
         f"`thin` = val/test cell below {THIN_LINES} lines, `EMPTY` = scribe missing from a split.", ""]
    for name, res in results.items():
        summ = res["summary"]
        L += [f"## {name}", "", "| scribe | train | val | test |", "|---|---|---|---|"]
        for s in scribes:
            cells = []
            for sp in ("train", "val", "test"):
                c = summ[sp][s]
                tag = " EMPTY" if c["lines"] == 0 else (" thin" if sp != "train" and c["lines"] < THIN_LINES else "")
                cells.append(f"{c['lines']} ({c['pages']}){tag}")
            L.append(f"| S{s} | " + " | ".join(cells) + " |")
        b = res["band_shortcut_baseline"]
        L += ["", f"- band-only baseline, test: accuracy {b['test']['line_accuracy']:.3f}, "
                  f"balanced {b['test']['balanced_accuracy']:.3f} (val balanced {b['val']['balanced_accuracy']:.3f})"]
        if res["moved_from_base"] is not None:
            L.append(f"- pages of split_v1 that changed split: {res['moved_from_base']}")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/nuremberg")
    ap.add_argument("--version", default="basic", choices=["basic", "diplomatic-regularised"])
    ap.add_argument("--scribes", type=int, nargs="+", default=list(DEFAULT_SCRIBES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="both", choices=["fresh", "fixed", "both", "lines"])
    ap.add_argument("--line-scribes", type=int, nargs="+", default=[11, 12],
                    help="mode lines: these scribes are split line by line, the rest follows --base-split")
    ap.add_argument("--base-split", default="nuremberg-7/docs/split_v1.json",
                    help="split whose pages stay fixed in mode fixed/both")
    ap.add_argument("--out-dir", default="nuremberg-9/docs")
    ap.add_argument("--tag", default="split_v2", help="file name prefix")
    args = ap.parse_args()
    if args.mode == "lines":
        run_lines(args)
        return

    rows = page_table(args.root, args.version, args.scribes)
    print(f"scribes: {args.scribes}   pages used: {len(rows)}   seed: {args.seed}")
    base_where = None
    if args.mode in ("fixed", "both"):
        with open(args.base_split, encoding="utf-8") as fh:
            base_where = {pid: sp for sp, ids in json.load(fh)["splits"].items() for pid in ids}
        in_rows = {r["page_id"] for r in rows}
        print(f"base split: {len(base_where)} pages, {len(set(base_where) & in_rows)} of them in the table; "
              f"new pages to place: {len(in_rows - set(base_where))}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variants = {}
    if args.mode in ("fresh", "both"):
        variants["fresh"] = make_split(rows, args.scribes, seed=args.seed)
    if args.mode in ("fixed", "both"):
        variants["fixed"] = make_split(rows, args.scribes, seed=args.seed, fixed=base_where)

    results = {}
    suffix = {"fresh": "a_fresh", "fixed": "b_fixed"}
    for mode, split in variants.items():
        res = describe(mode, rows, split, args.scribes, base_where)
        results[mode] = res
        path = out / (f"{args.tag}{suffix[mode]}.json" if args.mode == "both" else f"{args.tag}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": args.version, "mode": mode, "scribes": args.scribes, "seed": args.seed,
                       "fractions": FRACTIONS, "splits": split, "summary": res["summary"],
                       "band_shortcut_baseline": res["band_shortcut_baseline"],
                       "empty_cells": res["empty_cells"], "thin_cells": res["thin_cells"],
                       "moved_from_base": res["moved_from_base"]}, fh, indent=1)
        print(f"wrote {path}")
    if len(results) > 1:
        md = out / f"{args.tag}_compare.md"
        md.write_text(compare_markdown(args.scribes, results), encoding="utf-8")
        print(f"wrote {md}")


if __name__ == "__main__":
    main()

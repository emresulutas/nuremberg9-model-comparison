"""Print scribe statistics for the Nuremberg labels (lines, pages, split feasibility).

Run from the repo root:
    python -m src.data.nuremberg_stats --root datasets/nuremberg
    python -m src.data.nuremberg_stats --root datasets/nuremberg --version diplomatic-regularised --out docs/stats_basic.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict

from .nuremberg import iter_pages


def collect(root: str, version: str) -> dict:
    lines_per_scribe: Counter = Counter()
    pages_per_scribe: dict[int, set] = defaultdict(set)
    pages_per_scribe_band: dict[int, Counter] = defaultdict(Counter)
    scribes_per_page: Counter = Counter()
    lines_per_page = []
    multi = unlabeled = 0
    n_pages = missing_image = 0
    page_ids = []
    for page in iter_pages(root, version):
        n_pages += 1
        page_ids.append(page.page_id)
        if page.image_path is None:
            missing_image += 1
        lines_per_page.append(len(page.lines))
        present = set()
        for line in page.lines:
            if line.writer is None:
                if line.raw_writer.strip():
                    multi += 1
                else:
                    unlabeled += 1
                continue
            lines_per_scribe[line.writer] += 1
            pages_per_scribe[line.writer].add(page.page_id)
            pages_per_scribe_band[line.writer][page.band] += 1
            present.add(line.writer)
        scribes_per_page[len(present)] += 1
    total = sum(lines_per_scribe.values())
    return {
        "version": version,
        "pages": n_pages,
        "unique_page_ids": len(set(page_ids)),
        "pages_missing_image": missing_image,
        "lines_total_single_scribe": total,
        "lines_multi_scribe": multi,
        "lines_without_label": unlabeled,
        "lines_per_page_median": statistics.median(lines_per_page) if lines_per_page else 0,
        "scribes": {
            str(w): {
                "lines": lines_per_scribe[w],
                "share": round(lines_per_scribe[w] / total, 4) if total else 0,
                "pages": len(pages_per_scribe[w]),
                "lines_by_band": dict(sorted(pages_per_scribe_band[w].items())),
            }
            for w in sorted(lines_per_scribe)
        },
        "pages_by_number_of_scribes": dict(sorted(scribes_per_page.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/nuremberg")
    ap.add_argument("--version", default="basic", choices=["basic", "diplomatic-regularised"])
    ap.add_argument("--out", default=None, help="optional JSON path")
    args = ap.parse_args()

    s = collect(args.root, args.version)
    print(f"version: {s['version']}   pages: {s['pages']} (unique ids {s['unique_page_ids']}, "
          f"missing image: {s['pages_missing_image']})")
    print(f"single-scribe lines: {s['lines_total_single_scribe']}   multi-scribe: {s['lines_multi_scribe']}   "
          f"no label: {s['lines_without_label']}   median lines/page: {s['lines_per_page_median']}")
    print(f"\n{'scribe':>6} {'lines':>7} {'share':>7} {'pages':>6}   lines per band")
    for w, v in s["scribes"].items():
        bands = ", ".join(f"{b}:{n}" for b, n in v["lines_by_band"].items())
        print(f"{w:>6} {v['lines']:>7} {v['share']:>7.1%} {v['pages']:>6}   {bands}")
    print("\npages by number of different scribes on the page:")
    for k, n in s["pages_by_number_of_scribes"].items():
        print(f"  {k} scribe(s): {n} pages")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(s, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

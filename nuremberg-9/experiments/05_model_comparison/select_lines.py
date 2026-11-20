"""Select the same test lines for each model's explanation gallery."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


EXPERIMENT = Path(__file__).resolve().parent
NINE_SCRIBES = EXPERIMENT.parents[1]
SPLIT_JSON = NINE_SCRIBES / "docs/split_v2_9scribes.json"
PREDICTIONS = (
    EXPERIMENT.parent / "01_balanced_probe/results/weighted_oversample/blk16_predictions.npz"
)
OUT = EXPERIMENT / "results/whole_line/comparison/selected_lines.csv"
FIELDS = ("page_id", "line_id", "label", "writer")


def select_lines(predictions: Path, split_json: Path, per_scribe: int, seed: int) -> list[dict]:
    if per_scribe < 1:
        raise ValueError("per_scribe must be at least 1")
    scribes = sorted(json.loads(split_json.read_text(encoding="utf-8"))["scribes"])
    with np.load(predictions, allow_pickle=False) as data:
        page_ids, line_ids, labels = (data[key] for key in ("test_page_id", "test_line_id", "test_y"))
    if any(array.ndim != 1 for array in (page_ids, line_ids, labels)):
        raise ValueError("Expected one page ID, line ID, and label per test row")

    by_label = [[] for _ in scribes]
    seen = set()
    for page_id, line_id, label in zip(page_ids.tolist(), line_ids.tolist(), labels.tolist(), strict=True):
        if not isinstance(label, int) or not 0 <= label < len(scribes):
            raise ValueError(f"Invalid test label {label!r} for {page_id}/{line_id}")
        key = (str(page_id), str(line_id))
        if key in seen:
            raise ValueError(f"Duplicate test line {key[0]}/{key[1]}")
        seen.add(key)
        by_label[label].append(key)

    rng = random.Random(seed)
    selected = []
    for label, (writer, keys) in enumerate(zip(scribes, by_label, strict=True)):
        if not keys:
            raise ValueError(f"No test lines for S{writer}")
        drawn = rng.sample(sorted(keys), min(per_scribe, len(keys)))
        for page_id, line_id in sorted(drawn):
            selected.append(dict(page_id=page_id, line_id=line_id, label=label, writer=writer))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--split-json", type=Path, default=SPLIT_JSON)
    parser.add_argument("--out", type=Path, default=OUT, help="output .csv or .json file")
    parser.add_argument("--per-scribe", type=int, default=3, help="maximum lines per scribe")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    suffix = args.out.suffix.lower()
    if suffix not in (".csv", ".json"):
        parser.error("--out must end in .csv or .json")
    try:
        selected = select_lines(args.predictions, args.split_json, args.per_scribe, args.seed)
    except ValueError as error:
        parser.error(str(error))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        args.out.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    else:
        with args.out.open("w", encoding="utf-8", newline="") as handle:
            output = csv.DictWriter(handle, fieldnames=FIELDS)
            output.writeheader()
            output.writerows(selected)
    print(f"Wrote {len(selected)} test lines to {args.out}")


if __name__ == "__main__":
    main()

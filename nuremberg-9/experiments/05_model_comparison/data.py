"""Prepare grayscale line crops from the existing nine-scribe Nuremberg split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.nuremberg_lines import LineSample, build_samples, iter_crops, load_split  # noqa: E402

SCRIBES = (1, 2, 6, 7, 8, 9, 10, 11, 12)
SPLITS = ("train", "val", "test")
SPLIT_JSON = ROOT / "nuremberg-9/docs/split_v2_9scribes.json"
CHURRO_PREDICTIONS = (
    ROOT / "nuremberg-9/experiments/01_balanced_probe/results/weighted_oversample/blk16_predictions.npz"
)
DEFAULT_CACHE = ROOT / "cache/nuremberg-9-lines"


@dataclass(frozen=True)
class CachedLine:
    image_path: Path
    page_id: str
    line_id: str
    label: int


def load_rows(cache_dir: Path, split: str) -> list[CachedLine]:
    """Read one list of cached line images and their original PAGE XML identifiers."""
    with (cache_dir / f"{split}.csv").open(encoding="utf-8", newline="") as fh:
        return [
            CachedLine(cache_dir / row["image"], row["page_id"], row["line_id"], int(row["label"]))
            for row in csv.DictReader(fh)
        ]


def _check_membership(samples: dict[str, list[LineSample]], split_info: dict) -> None:
    seen: set[tuple[str, str]] = set()
    with np.load(CHURRO_PREDICTIONS, allow_pickle=False) as churro:
        for split, lines in samples.items():
            actual = {(s.page_id, s.line_id): s.label for s in lines}
            if len(actual) != len(lines) or seen.intersection(actual):
                raise ValueError(f"Duplicate line IDs in or before {split}")
            seen.update(actual)
            counts = Counter(s.writer for s in lines)
            expected = {int(k): v["lines"] for k, v in split_info["summary"][split].items()}
            if counts != expected:
                raise ValueError(f"{split} line counts differ from the split file: {dict(counts)}")
            if split in ("val", "test"):
                keys = zip(churro[f"{split}_page_id"].tolist(), churro[f"{split}_line_id"].tolist())
                reference = dict(zip(keys, churro[f"{split}_y"].tolist(), strict=True))
                if actual != reference:
                    raise ValueError(f"{split} IDs or labels differ from the saved CHURRO predictions")


def prepare_cache(data_root: Path, cache_dir: Path) -> dict[str, int]:
    if cache_dir.exists():
        raise FileExistsError(f"Cache already exists: {cache_dir}. Use a new --cache-dir to rebuild.")
    split_info = load_split(SPLIT_JSON)
    if tuple(sorted(split_info["scribes"])) != SCRIBES:
        raise ValueError(f"Unexpected class mapping in {SPLIT_JSON}")
    samples = {split: build_samples(data_root, SPLIT_JSON, split) for split in SPLITS}
    _check_membership(samples, split_info)
    counts = {split: len(lines) for split, lines in samples.items()}

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".nuremberg-lines-", dir=cache_dir.parent) as tmp:
        staging = Path(tmp) / "lines"
        staging.mkdir()
        for split, lines in samples.items():
            (staging / split).mkdir()
            with (staging / f"{split}.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(("image", "page_id", "line_id", "label"))
                for index, (sample, crop) in enumerate(iter_crops(lines)):
                    relative = f"{split}/{index:05d}.png"
                    crop.convert("L").save(staging / relative)
                    writer.writerow((relative, sample.page_id, sample.line_id, sample.label))
        metadata = {
            "scribes": SCRIBES,
            "line_counts": counts,
            "split_json": SPLIT_JSON.relative_to(ROOT).as_posix(),
            "split_sha256": hashlib.sha256(SPLIT_JSON.read_bytes()).hexdigest(),
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        staging.rename(cache_dir)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/nuremberg")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    if not (args.data_root / "images").is_dir() or not (args.data_root / "labels").is_dir():
        parser.error(f"Expected 'images' and 'labels' under --data-root {args.data_root}")
    try:
        counts = prepare_cache(args.data_root, args.cache_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Prepared {counts} in {args.cache_dir}")


if __name__ == "__main__":
    main()

"""Build shared word and augmented-line caches with the CHURRO crop recipes."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from os.path import relpath
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

EXP = Path(__file__).resolve().parent.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "05_model_comparison"))
from data import SCRIBES, SPLITS, SPLIT_JSON, load_rows
from src.data.augment import augment_line, key_rng
from src.data.nuremberg_lines import build_samples, iter_crops
from src.data.nuremberg_words import split_words, tokens

COPIES = {12: 16, 11: 8, 10: 5, 2: 2, 6: 2, 8: 2}
WORD_REFERENCE = EXP / "03_word_vs_line/results/words"
FIELDS = ("image", "page_id", "line_id", "label")


def keys(rows):
    return {(row.page_id, row.line_id): row.label for row in rows}


def word_signature(page, line, y, token):
    return Counter(zip(page, line, y, token, strict=True))


def build_words(samples, stage):
    reference = json.loads((WORD_REFERENCE / "metrics.json").read_text(encoding="utf-8"))["words_per_split"]
    for split, lines in samples.items():
        (stage / split).mkdir()
        counts, manifest = set(), []
        with (stage / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow((*FIELDS, "token"))
            for sample, crop in iter_crops(lines):
                words = tokens(sample.text)
                gray = np.asarray(crop.convert("L"))
                found = split_words(gray, words) if words else None
                if found is None or found[1] > .30:
                    continue
                counts.add((sample.page_id, sample.line_id))
                for token, (left, right) in enumerate(found[0]):
                    name = f"{split}/{len(manifest):06d}.png"
                    Image.fromarray(gray[:, max(left - 3, 0):right + 3]).save(stage / name)
                    row = (sample.page_id, sample.line_id, sample.label, token)
                    writer.writerow((name, *row))
                    manifest.append(row)
        actual = {"lines_seen": len(lines), "lines_cut": len(counts), "words": len(manifest)}
        if actual != reference[split]:
            raise ValueError(f"{split} word selection differs from CHURRO: {actual} vs {reference[split]}")
        if split == "test":
            with np.load(WORD_REFERENCE / "predictions.npz", allow_pickle=False) as saved:
                expected = word_signature(saved["test_page_id"], saved["test_line_id"],
                                          saved["test_y"], saved["test_token"])
            if Counter(manifest) != expected:
                raise ValueError("Word IDs, token indices or labels differ from CHURRO")
        print(f"Word cache {split}: {len(manifest):,} words on {len(counts):,} lines", flush=True)


def build_augmented(rows, samples, stage, dest):
    (stage / "train").mkdir()
    for split, originals in rows.items():
        with (stage / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            for row in originals:
                writer.writerow((relpath(row.image_path.resolve(), dest.resolve()),
                                 row.page_id, row.line_id, row.label))
            if split != "train":
                continue
            extra = Counter()
            chosen = [row for row in samples if row.writer in COPIES]
            for row, crop in iter_crops(chosen):
                for k in range(COPIES[row.writer]):
                    name = f"train/{sum(extra.values()):05d}.png"
                    augment_line(crop, key_rng(row.page_id, row.line_id, k)).convert("L").save(stage / name)
                    writer.writerow((name, row.page_id, f"{row.line_id}#aug{k}", row.label))
                    extra[row.writer] += 1
            expected = Counter({scribe: sum(s.writer == scribe for s in samples) * n
                                for scribe, n in COPIES.items()})
            if extra != expected:
                raise ValueError(f"Augmentation counts differ: {extra} vs {expected}")
            print(f"Augmented cache: {sum(extra.values()):,} extra training lines ({dict(extra)})", flush=True)


def ready(dest, build):
    marker = dest / "metadata.json"
    if marker.is_file() and all((dest / f"{s}.csv").is_file() for s in SPLITS):
        print(f"Reusing {dest}", flush=True)
        return
    if dest.exists():
        raise FileExistsError(f"Incomplete cache {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".details-", dir=dest.parent) as tmp:
        stage = Path(tmp) / "data"
        stage.mkdir()
        build(stage)
        (stage / "metadata.json").write_text(json.dumps({"scribes": SCRIBES}), encoding="utf-8")
        stage.rename(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=ROOT / "datasets/nuremberg")
    ap.add_argument("--line-cache", type=Path, default=ROOT / "cache/nuremberg-9-lines")
    ap.add_argument("--work-cache", type=Path, default=ROOT / "cache/nuremberg-9-model-details")
    args = ap.parse_args()
    rows = {s: load_rows(args.line_cache, s) for s in SPLITS}
    samples = {s: build_samples(args.data_root, SPLIT_JSON, s) for s in SPLITS}
    for split in SPLITS:
        if keys(rows[split]) != keys(samples[split]) or len(rows[split]) != len(samples[split]):
            raise ValueError(f"{split} does not match the original whole-line cache")
    ready(args.work_cache / "words", lambda stage: build_words(samples, stage))
    augmented = args.work_cache / "augmented-lines"
    ready(augmented, lambda stage: build_augmented(rows, samples["train"], stage, augmented))


if __name__ == "__main__":
    main()

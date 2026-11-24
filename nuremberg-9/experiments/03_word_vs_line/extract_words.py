"""Plan step 5, part 1: encoder features of single words (GPU).

    CUDA_VISIBLE_DEVICES=2 python nuremberg-9/experiments/03_word_vs_line/extract_words.py --scribes 1 2 6 8

Every line of the 9-scribe split is cut into words (src/data/nuremberg_words.py, same rule as the word-level study:
lines with cut quality > --max-quality are dropped), each word crop (grey, 3 px margin) goes through the frozen encoder
like a line. Only blk16 is kept. One file per split and scribe: features/churro_v2_words/<split>_S<k>.npz; finished
files are skipped on a restart.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.nuremberg_lines import build_samples, iter_crops   # noqa: E402
from src.data.nuremberg_words import split_words, tokens         # noqa: E402
from src.models.churro_encoder import ChurroEncoder              # noqa: E402

LAYER = "blk16"
MARGIN = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/nuremberg")
    ap.add_argument("--split-json", default="nuremberg-9/docs/split_v2_9scribes.json")
    ap.add_argument("--out", default="features/churro_v2_words")
    ap.add_argument("--scribes", type=int, nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--max-quality", type=float, default=0.30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-pixels", type=int, default=50176)
    ap.add_argument("--max-pixels", type=int, default=1_003_520)
    ap.add_argument("--limit", type=int, default=0, help="only the first N lines per file (smoke test)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = ChurroEncoder(device=args.device, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    for split in args.splits:
        samples_all = build_samples(args.root, args.split_json, split)
        for scribe in args.scribes:
            path = out / f"{split}_S{scribe}.npz"
            if path.exists():
                print(f"{split} S{scribe}: exists, skipping", flush=True)
                continue
            samples = [s for s in samples_all if s.writer == scribe]
            if args.limit:
                samples = samples[: args.limit]
            vecs, meta = [], {"label": [], "page_id": [], "line_id": [], "token": [], "word": []}
            kept, t0 = 0, time.time()
            for i, (s, crop) in enumerate(iter_crops(samples), 1):
                words = tokens(s.text)
                gray = np.asarray(crop.convert("L"))
                got = split_words(gray, words) if words else None
                if got is not None and got[1] <= args.max_quality:
                    kept += 1
                    for k, (x0, x1) in enumerate(got[0]):
                        piece = gray[:, max(x0 - MARGIN, 0): x1 + MARGIN]
                        vecs.append(enc.encode(Image.fromarray(piece).convert("RGB"))[LAYER])
                        meta["label"].append(s.label)
                        meta["page_id"].append(s.page_id)
                        meta["line_id"].append(s.line_id)
                        meta["token"].append(k)
                        meta["word"].append(words[k])
                if i % 200 == 0 or i == len(samples):
                    rate = i / (time.time() - t0)
                    print(f"{split} S{scribe}: {i}/{len(samples)} lines, {kept} cut, {len(vecs)} words, "
                          f"{rate:.1f} lines/s, eta {(len(samples) - i) / rate / 60:.1f} min", flush=True)
            np.savez(path, **{LAYER: np.stack(vecs)}, label=np.array(meta["label"]), page_id=np.array(meta["page_id"]),
                     line_id=np.array(meta["line_id"]), token=np.array(meta["token"]), word=np.array(meta["word"]),
                     lines_seen=len(samples), lines_cut=kept)
            print(f"saved {path} ({len(vecs)} words from {kept}/{len(samples)} lines)", flush=True)


if __name__ == "__main__":
    main()

"""Plan step 4, part 3: encoder features of augmented copies of TRAIN lines of the rare scribes (GPU, about 20 min).

    CUDA_VISIBLE_DEVICES=2 python nuremberg-9/experiments/02_augmentation/extract_aug.py

Copies per line: S12 x16, S11 x8, S10 x5, S2 x2, S6 x2, S8 x2 (see COPIES). Copy k of a line is always the same
(seed from page, line and k). One file per scribe, features/churro_v2_aug/train_S<k>.npz; a finished scribe is skipped
on a restart. Val and test are never augmented. The original cache is not touched.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.augment import augment_line, key_rng             # noqa: E402
from src.data.nuremberg_lines import build_samples, iter_crops  # noqa: E402
from src.models.churro_encoder import FEATURES, ChurroEncoder   # noqa: E402

COPIES = {12: 16, 11: 8, 10: 5, 2: 2, 6: 2, 8: 2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/nuremberg")
    ap.add_argument("--split-json", default="nuremberg-9/docs/split_v2_9scribes.json")
    ap.add_argument("--out", default="features/churro_v2_aug")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-pixels", type=int, default=50176)
    ap.add_argument("--max-pixels", type=int, default=1_003_520)
    ap.add_argument("--limit", type=int, default=0, help="only the first N lines per scribe (smoke test)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = ChurroEncoder(device=args.device, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    train = build_samples(args.root, args.split_json, "train")
    for scribe, copies in COPIES.items():
        path = out / f"train_S{scribe}.npz"
        if path.exists():
            print(f"S{scribe}: {path} exists, skipping", flush=True)
            continue
        samples = [s for s in train if s.writer == scribe]
        if args.limit:
            samples = samples[: args.limit]
        feats = {name: [] for name in FEATURES}
        meta = {"label": [], "writer": [], "page_id": [], "line_id": []}
        total, t0 = len(samples) * copies, time.time()
        for s, crop in iter_crops(samples):
            for k in range(copies):
                img = augment_line(crop, key_rng(s.page_id, s.line_id, k)).convert("L").convert("RGB")
                vec = enc.encode(img)
                for name in FEATURES:
                    feats[name].append(vec[name])
                meta["label"].append(s.label)
                meta["writer"].append(s.writer)
                meta["page_id"].append(s.page_id)
                meta["line_id"].append(f"{s.line_id}#aug{k}")
                n = len(meta["label"])
                if n % 500 == 0 or n == total:
                    rate = n / (time.time() - t0)
                    print(f"S{scribe}: {n}/{total} copies, {rate:.1f}/s, eta {(total - n) / rate / 60:.1f} min", flush=True)
        arrays = {name: np.stack(v) for name, v in feats.items()}
        np.savez(path, **arrays, label=np.array(meta["label"]), writer=np.array(meta["writer"]),
                 page_id=np.array(meta["page_id"]), line_id=np.array(meta["line_id"]))
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()

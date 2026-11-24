"""Plan step 4, part 2: look at the augmented copies before any feature is extracted (CPU, a minute, needs the images).

    python nuremberg-9/experiments/02_augmentation/preview.py

Writes to results/preview/:
  rotation_demo.png     one line of S12 rotated by 0, 5, 10, 20, 30 and -30 degrees: how far may the angle go?
  aug_S<k>.png          per scribe (S2 S6 S8 S10 S11 S12): two train lines, each as the original and 5 random copies,
                        shown grey, as the encoder will see them.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.augment import augment_line, key_rng, rotate_only   # noqa: E402
from src.data.nuremberg_lines import build_samples, iter_crops    # noqa: E402

WIDTH_IN = 11.0        # figure width in inches


def strip_figure(items, path, title):
    """items: list of (label, PIL image). All strips use the same pixels-to-inches factor."""
    max_w = max(im.width for _, im in items)
    ipp = WIDTH_IN / max_w
    label_h = 0.28
    heights = [im.height * ipp for _, im in items]
    total = sum(heights) + label_h * len(items) + 0.5
    fig = plt.figure(figsize=(WIDTH_IN, total))
    fig.suptitle(title, fontsize=12, y=1 - 0.12 / total)
    y = total - 0.5
    for (label, im), h in zip(items, heights):
        y -= label_h
        fig.text(0.01, y / total + 0.002, label, fontsize=9, va="bottom")
        y -= h
        ax = fig.add_axes([0, y / total, im.width * ipp / WIDTH_IN, h / total])
        ax.imshow(im.convert("L"), cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.axis("off")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/nuremberg")
    ap.add_argument("--split-json", default="nuremberg-9/docs/split_v2_9scribes.json")
    ap.add_argument("--out", default="nuremberg-9/experiments/02_augmentation/results/preview")
    ap.add_argument("--scribes", type=int, nargs="+", default=[2, 6, 8, 10, 11, 12])
    ap.add_argument("--lines", type=int, default=2, help="lines per scribe")
    ap.add_argument("--copies", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = build_samples(args.root, args.split_json, "train")
    rng = random.Random(args.seed)
    chosen = {}
    for s in args.scribes:
        pool = [x for x in samples if x.writer == s and len(x.text) >= 15]
        chosen[s] = rng.sample(pool, min(args.lines, len(pool)))
    flat = [x for s in args.scribes for x in chosen[s]]
    crops = {(x.page_id, x.line_id): c for x, c in iter_crops(flat)}

    demo = crops[(chosen[12][0].page_id, chosen[12][0].line_id)]
    strip_figure([(f"rotation {a:+d} deg", rotate_only(demo, a)) for a in (0, 5, 10, 20, 30, -30)],
                 out / "rotation_demo.png", "S12, one train line: how far may the rotation go?")
    for s in args.scribes:
        items = []
        for x in chosen[s]:
            im = crops[(x.page_id, x.line_id)]
            items.append((f"S{s} original   {x.page_id} / {x.line_id}", im))
            for k in range(args.copies):
                items.append((f"copy {k + 1}", augment_line(im, key_rng(x.page_id, x.line_id, 1000 + k))))
        strip_figure(items, out / f"aug_S{s}.png", f"S{s}: original and random copies (as the encoder sees them: grey)")
        print(f"wrote {out / f'aug_S{s}.png'}")
    print(f"wrote {out / 'rotation_demo.png'}")


if __name__ == "__main__":
    main()

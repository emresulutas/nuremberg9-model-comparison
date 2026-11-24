"""Turn a split file + the PAGE XML labels into line crops (one image per labeled text line).

Typical use (feature extraction opens every page image only once):

    samples = build_samples("datasets/nuremberg", "nuremberg-9/docs/split_v2_9scribes.json", "train")
    for sample, crop in iter_crops(samples):          # crop is a PIL image of one line
        ...                                           # sample.label is the class index

The class index comes from the sorted scribe list stored in the split file, so every split of
one experiment uses the same mapping. Lines with several scribe ids or no id are skipped.
No torch needed here; the optional torch Dataset below imports it lazily.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageDraw, ImageStat

from .nuremberg import iter_pages


@dataclass(frozen=True)
class LineSample:
    page_id: str
    image_path: Path
    line_id: str
    writer: int                       # scribe number as in the XML (1..13)
    label: int                        # class index (position of the scribe in the sorted scribe list)
    polygon: tuple[tuple[int, int], ...]
    text: str = ""                    # transcription of the line

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)


def load_split(split_json: str | Path) -> dict:
    with open(split_json, encoding="utf-8") as fh:
        return json.load(fh)


def build_samples(root: str | Path, split_json: str | Path, split: str, version: str | None = None) -> list[LineSample]:
    """All labeled lines that belong to `split` ('train' | 'val' | 'test').

    Normally a line follows its page. A split file may also hold `line_overrides`
    ({scribe: {split: ["<page_id>|<line_id>", ...]}}): the lines of those scribes follow their own
    entry instead of the page (used for the very rare scribes S11 and S12)."""
    info = load_split(split_json)
    scribes = sorted(info["scribes"])
    to_label = {s: i for i, s in enumerate(scribes)}
    wanted = set(info["splits"][split])
    overrides = {int(s): set(per.get(split, [])) for s, per in info.get("line_overrides", {}).items()}
    extra_pages = {k.rsplit("|", 1)[0] for keys in overrides.values() for k in keys}
    version = version or info.get("version", "basic")
    samples: list[LineSample] = []
    for page in iter_pages(root, version):
        if page.page_id not in wanted and page.page_id not in extra_pages:
            continue
        if page.image_path is None:
            raise FileNotFoundError(f"image of {page.page_id} not found ({page.image_name})")
        for line in page.lines:
            if line.writer in overrides:
                if f"{page.page_id}|{line.line_id}" not in overrides[line.writer]:
                    continue
            elif page.page_id not in wanted:
                continue
            if line.writer in to_label:
                samples.append(LineSample(page.page_id, page.image_path, line.line_id, line.writer,
                                          to_label[line.writer], tuple(line.polygon), line.text))
    return samples


def crop_line(page_image: Image.Image, polygon, pad: int = 4, mask_outside: bool = True) -> Image.Image:
    """Crop the line's bounding box; paint everything outside the polygon with the paper colour,
    so neighbouring lines that poke into the box do not leak into the crop."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    left, top = max(min(xs) - pad, 0), max(min(ys) - pad, 0)
    right, bottom = min(max(xs) + pad, page_image.width), min(max(ys) + pad, page_image.height)
    crop = page_image.crop((left, top, right, bottom)).convert("RGB")
    if mask_outside:
        mask = Image.new("L", crop.size, 0)
        shifted = [(x - left, y - top) for x, y in polygon]
        ImageDraw.Draw(mask).polygon(shifted, fill=255)
        paper = tuple(int(v) for v in ImageStat.Stat(crop).median)
        background = Image.new("RGB", crop.size, paper)
        crop = Image.composite(crop, background, mask)
    return crop


def iter_crops(samples: list[LineSample], **crop_kwargs) -> Iterator[tuple[LineSample, Image.Image]]:
    """Yield (sample, crop) page by page, opening each page image once."""
    by_page: dict[Path, list[LineSample]] = defaultdict(list)
    for s in samples:
        by_page[s.image_path].append(s)
    for path in sorted(by_page):
        with Image.open(path) as im:
            im.load()
            for s in by_page[path]:
                yield s, crop_line(im, s.polygon, **crop_kwargs)


class LineDataset:
    """torch-style dataset: item = (transformed crop, label). Slow (reopens the page per item);
    use iter_crops for feature extraction and this only for small experiments."""

    def __init__(self, samples: list[LineSample], transform=None, **crop_kwargs):
        self.samples = samples
        self.transform = transform
        self.crop_kwargs = crop_kwargs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        with Image.open(s.image_path) as im:
            crop = crop_line(im.convert("RGB"), s.polygon, **self.crop_kwargs)
        return (self.transform(crop) if self.transform else crop), s.label

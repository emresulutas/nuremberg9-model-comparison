"""Reader for the Nuremberg Letterbooks PAGE XML labels.

Layout on disk (see docs/dataset_notes.md):
    <root>/images/BandN/<stem>.jpg
    <root>/labels/nuremberg_letterbooks/<version>/BandN/<k>.xml     version = basic | diplomatic-regularised

The scribe label is stored per text line (``<TextLine writerID="6">``); a line with several
ids (``"1,2,6"``) is of uncertain authorship and gets ``writer = None``. The XML names its image
in ``<Page imageFilename="...png">``; the image on disk is a .jpg, so images are matched by stem.

No torch needed. Uses defusedxml if installed (pip install defusedxml), else the stdlib parser.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
try:  # the XML comes from a third-party download: prefer the hardened parser when installed
    from defusedxml import ElementTree as ET
except ImportError:  # pragma: no cover
    from xml.etree import ElementTree as ET

LABEL_SUBDIR = Path("labels") / "nuremberg_letterbooks"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass
class Line:
    line_id: str
    writer: int | None          # single scribe id, or None when the label lists several ids / none
    raw_writer: str             # the writerID attribute as written in the XML ("" if absent)
    polygon: list[tuple[int, int]]
    baseline: list[tuple[int, int]]
    text: str = ""              # transcription of the line (TextEquiv/Unicode), "" if absent

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) of the line polygon."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class Page:
    xml_path: Path
    band: str                   # "Band2" .. "Band5"
    image_name: str             # imageFilename as written in the XML
    image_path: Path | None     # resolved file on disk, None if missing
    width: int
    height: int
    lines: list[Line] = field(default_factory=list)

    @property
    def page_id(self) -> str:
        """Unique id of the page: band + image stem. Use this to keep splits page-disjoint."""
        return f"{self.band}/{Path(self.image_name).stem}"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _points(text: str | None) -> list[tuple[int, int]]:
    pts = []
    for item in (text or "").split():
        x, _, y = item.partition(",")
        pts.append((int(float(x)), int(float(y))))
    return pts


def parse_writer(raw: str) -> int | None:
    """'6' -> 6;  '1,2,6' or '' or non-numeric -> None (not a single known scribe)."""
    raw = raw.strip()
    return int(raw) if raw.isdigit() else None


def resolve_image(images_root: Path, band: str, image_name: str) -> Path | None:
    """Find the page image: same stem, any image suffix, inside images/<band>/."""
    stem = Path(image_name).stem
    for suffix in IMAGE_SUFFIXES:
        cand = images_root / band / (stem + suffix)
        if cand.exists():
            return cand
    return None


def parse_page(xml_path: Path, images_root: Path | None = None) -> Page:
    root = ET.parse(xml_path).getroot()
    page_el = next(e for e in root.iter() if _local(e.tag) == "Page")
    image_name = page_el.get("imageFilename", "")
    band = xml_path.parent.name
    image_path = resolve_image(images_root, band, image_name) if images_root else None
    page = Page(
        xml_path=xml_path,
        band=band,
        image_name=image_name,
        image_path=image_path,
        width=int(page_el.get("imageWidth", 0)),
        height=int(page_el.get("imageHeight", 0)),
    )
    for el in page_el.iter():
        if _local(el.tag) != "TextLine":
            continue
        coords = next((c for c in el if _local(c.tag) == "Coords"), None)
        baseline = next((c for c in el if _local(c.tag) == "Baseline"), None)
        polygon = _points(coords.get("points")) if coords is not None else []
        if not polygon:
            continue
        raw = el.get("writerID", "")
        text = ""
        for te in el:                       # the line's own TextEquiv (not the ones of child elements)
            if _local(te.tag) == "TextEquiv":
                uni = next((c for c in te if _local(c.tag) == "Unicode"), None)
                text = (uni.text or "") if uni is not None else ""
                break
        page.lines.append(
            Line(
                line_id=el.get("id", ""),
                writer=parse_writer(raw),
                raw_writer=raw,
                polygon=polygon,
                baseline=_points(baseline.get("points")) if baseline is not None else [],
                text=text,
            )
        )
    return page


def iter_pages(root: str | Path, version: str = "basic"):
    """Yield every Page of one label version, in a stable order (band, then file number)."""
    root = Path(root)
    label_dir = root / LABEL_SUBDIR / version
    images_root = root / "images"
    for band_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
        xmls = sorted(band_dir.glob("*.xml"), key=lambda p: (len(p.stem), p.stem))
        for xml_path in xmls:
            yield parse_page(xml_path, images_root)

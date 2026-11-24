"""Export one actual test-line crop per writer from the existing explanation selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from shutil import copy2

from compare import ROOT, SPLIT, unseen_page_mask
from data import load_rows

DEFAULT_RESULTS = Path("nuremberg-9/experiments/05_model_comparison/results/whole_line/comparison")
DEFAULT_OUT = Path("nuremberg-9/report/04-writer-lines")


def export(cache, selection, out):
    split = json.loads((ROOT / SPLIT).read_text(encoding="utf-8"))
    scribes = sorted(split["scribes"])
    with selection.open(encoding="utf-8", newline="") as file:
        candidates = list(csv.DictReader(file))
    test = {(r.page_id, r.line_id): r for r in load_rows(cache, "test")}
    keys = {(row["page_id"], row["line_id"]): i for i, row in enumerate(candidates)}
    if len(keys) != len(candidates):
        raise ValueError("Duplicate line in the explanation selection")
    unseen = unseen_page_mask(split, keys)
    picked = {}
    for i, row in enumerate(candidates):
        writer = int(row["writer"])
        key = (row["page_id"], row["line_id"])
        original = test[key]
        if scribes[original.label] != writer:
            raise ValueError(f"Writer label differs for {key}")
        if writer not in picked or (unseen[i] and not picked[writer][0]):
            picked[writer] = (bool(unseen[i]), original)
    if set(picked) != set(scribes):
        raise ValueError("Expected one available crop for every writer")

    out.mkdir(parents=True, exist_ok=True)
    guide = ["# Nuremberg-9 · Yazar başına bir örnek satır",
             "Bunlar model girdisi olarak kullanılan gerçek, maskelenmiş test satırları. Her yazar için açıklama galerisinden bir örnek seçildi. Uygun olduğunda eğitim ve doğrulamada görülmemiş sayfalar tercih edildi.",
             "S11 ve S12 örneklerinin sayfaları veri bölümleri arasında ortak. Tek bir satır, yazarın el yazısındaki bütün çeşitliliği kapsamaz."]
    trace = []
    for writer in scribes:
        held_out, row = picked[writer]
        name = f"S{writer:02d}-line.png"
        copy2(row.image_path, out / name)
        trace.append((f"S{writer}", name, row.page_id, row.line_id, "yes" if held_out else "no"))
        guide += [f"## S{writer}", f"![S{writer} el yazısı örneği]({name})",
                  f"Sayfa: {row.page_id} · Satır: {row.line_id}"]
    with (out / "source-lines.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("writer", "image", "page_id", "line_id", "unseen_page"))
        writer.writerows(trace)
    (out / "README.md").write_text("\n\n".join(guide) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("cache/nuremberg-9-lines"))
    ap.add_argument("--selection", type=Path, default=DEFAULT_RESULTS / "selected_lines.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    export(args.cache, args.selection, args.out)

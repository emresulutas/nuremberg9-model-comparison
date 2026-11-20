"""Draw Grad-CAM and native attention for the same Nuremberg-9 test lines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from data import CHURRO_PREDICTIONS, DEFAULT_CACHE, SCRIBES, CachedLine, load_rows  # noqa: E402
from select_lines import OUT as DEFAULT_SELECTION  # noqa: E402

EXPERIMENT = Path(__file__).resolve().parent
MODELS = ("churro", "resnet18", "resnet50", "swin_tiny")
MODEL_LABELS = {"churro": "CHURRO", "resnet18": "ResNet18", "resnet50": "ResNet50", "swin_tiny": "Swin-Tiny"}
METHODS = {"gradcam": ("Grad-CAM", "inferno"), "attention": ("Self-attention", "viridis")}
PUBLICATION_FILE = "line_explanations.png"


def read_selection(cache_dir: Path, path: Path) -> list[CachedLine]:
    by_key = {(row.page_id, row.line_id): row for row in load_rows(cache_dir, "test")}
    if path.suffix.lower() == ".json":
        selected = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open(encoding="utf-8", newline="") as fh:
            selected = list(csv.DictReader(fh))
    if not selected:
        raise ValueError("The explanation selection has no test lines")
    rows = []
    seen = set()
    for item in selected:
        key = (item["page_id"], item["line_id"])
        row = by_key.get(key)
        if row is None or key in seen or row.label != int(item["label"]) or SCRIBES[row.label] != int(item["writer"]):
            raise ValueError(f"Unknown, repeated or mislabeled test line: {key}")
        rows.append(row)
        seen.add(key)
    return rows


def project_map(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a finite, nonempty two-dimensional explanation map")
    if values.shape == size[::-1]:
        return values.astype(np.float32, copy=False)
    return np.asarray(Image.fromarray(values.astype(np.float32)).resize(size, Image.Resampling.BILINEAR))


def draw_panel(path: Path, examples: list[tuple[CachedLine, Image.Image, dict]], dpi: int) -> str:
    """Draw aligned, aspect-preserving strips for the same source line in each column."""
    panels = [(None, None)]
    for method in METHODS:
        panels.extend((model, method) for model, (_, maps) in examples[0][2].items() if method in maps)

    label_width, crop_width, prediction_width = 1.28, 4.60, 0.56
    largest_aspect = max(image.height / image.width for _, image, _ in examples)
    strip_height = np.clip(crop_width * largest_aspect, 0.31, 0.66 if len(examples) > 1 else 0.91)
    displayed_widths = [min(crop_width, strip_height * image.width / image.height) for _, image, _ in examples]
    width = max(6.40, label_width + sum(displayed_widths) + prediction_width * len(examples))
    row_gap, header, footer = 0.09, 0.78, 0.50
    height = header + len(panels) * (strip_height + row_gap) - row_gap + footer
    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig.text(0.10 / width, 1 - 0.19 / height, "Nuremberg-9 line explanations",
             fontsize=12, weight="bold", va="top")

    left = label_width
    for column, ((row, image, results), displayed_width) in enumerate(zip(examples, displayed_widths, strict=True)):
        x = left / width
        fig.text(x, 1 - 0.59 / height, f"True writer S{SCRIBES[row.label]}", fontsize=9,
                 weight="bold", va="center")
        for index, (model, method) in enumerate(panels):
            y = 1 - (header + index * (strip_height + row_gap) + strip_height) / height
            center_y = y + strip_height / 2 / height
            ax = fig.add_axes((x, y, displayed_width / width, strip_height / height), anchor="W")
            ax.axis("off")
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            label = "Original"
            if model is not None:
                name, cmap = METHODS[method]
                label = f"{MODEL_LABELS[model]}\n{name}"
                prob, maps = results[model]
                values = maps[method]
                if method == "attention":
                    values = values - values.min()
                scale = max(float(values.max()), 1e-12)
                opacity = np.clip(values / scale, 0, 1) * 0.68
                ax.imshow(values, cmap=cmap, vmin=0, vmax=scale, alpha=opacity)
                if method == "gradcam":
                    fig.text(x + (displayed_width + 0.07) / width, center_y, f"S{SCRIBES[int(prob.argmax())]}",
                             fontsize=8, va="center")
            if column == 0:
                fig.text(0.10 / width, center_y, label, fontsize=8, va="center")
        left += displayed_width + prediction_width

    attention_caption = "Self-attention is class-independent. " if any(method == "attention" for _, method in panels) else ""
    caption = ("Grad-CAM targets the predicted writer shown beside each line.\n"
               f"{attention_caption}Color scales are independent for each map.")
    fig.text(0.10 / width, 0.28 / height, caption, fontsize=7.5, color="#454545", va="center")
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return caption


def model_files(model: str, results_dir: Path) -> dict[str, Path]:
    if model == "churro":
        return {"checkpoint": CHURRO_PREDICTIONS.with_name("blk16_head.npz"), "predictions": CHURRO_PREDICTIONS}
    run = results_dir / model / "fit"
    if not run.exists():
        run = run.parent
    return {"checkpoint": run / "best.pt", "predictions": run / "predictions.npz"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_identity(index: int, row: CachedLine) -> dict:
    writer = SCRIBES[row.label]
    return {"file": f"{index:02d}_S{writer}", "page_id": row.page_id, "line_id": row.line_id, "writer": writer}


def load_model_gallery(out: Path, model: str, inputs: dict, rows: list[CachedLine], images: list[Image.Image]):
    """Reuse an atomically published model gallery after an interrupted combined run."""
    catalog = json.loads((out / "catalog.json").read_text(encoding="utf-8"))
    if (catalog.get("model") != model or catalog.get("inputs") != inputs or catalog.get("scribes") != list(SCRIBES)
            or len(catalog["lines"]) != len(rows)):
        raise ValueError(f"Existing model gallery does not match the selected lines or fitted model: {out}")

    expected_maps = ("gradcam", "attention") if model in ("churro", "swin_tiny") else ("gradcam",)
    results = []
    for i, (saved, row, image) in enumerate(zip(catalog["lines"], rows, images, strict=True), 1):
        identity = line_identity(i, row)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Existing model gallery does not match the selected lines: {out}")
        stem = out / identity["file"]
        figure = stem.with_suffix(".png")
        if not figure.is_file() or not figure.stat().st_size:
            raise ValueError(f"Missing or empty line figure: {figure}")
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
            if not np.array_equal(archive["image"], np.asarray(image)):
                raise ValueError(f"Existing model gallery does not match the selected crop: {stem}")
            maps = {method: project_map(archive[f"{model}_{method}"], image.size) for method in expected_maps}
        prob = np.asarray(saved["probabilities"][model], dtype=np.float32)
        if prob.shape != (len(SCRIBES),) or not np.isfinite(prob).all():
            raise ValueError(f"Invalid probabilities in {out / 'catalog.json'}")
        results.append((prob, maps))
    return results


def generate_model_gallery(model: str, checkpoint: Path, rows: list[CachedLine], images: list[Image.Image], device: str):
    print(f"Explaining {len(rows)} test lines with {MODEL_LABELS[model]}", flush=True)
    if model == "churro":
        from explain_churro import ChurroExplainer

        explainer = ChurroExplainer(device)
    else:
        from explain_baseline import BaselineExplainer

        explainer = BaselineExplainer(checkpoint, device)
        if explainer.model_name != model:
            raise ValueError(f"Expected a {model} checkpoint, got {explainer.model_name}")
    results = []
    for row, image in zip(rows, images, strict=True):
        prob, maps = explainer.explain(image, row)
        results.append((prob, {key: project_map(value, image.size) for key, value in maps.items()}))
    return results


def publication_examples(rows: list[CachedLine], images: list[Image.Image]) -> list[int]:
    indices = [0]
    if len(rows) > 1:
        candidates = [i for i, row in enumerate(rows[1:], 1) if row.label != rows[0].label] or range(1, len(rows))
        aspects = [np.log(image.width / image.height) for image in images]
        indices.append(min(candidates, key=lambda i: abs(aspects[i] - aspects[0])))
    return indices


def publish_gallery(out: Path, rows: list[CachedLine], images: list[Image.Image], per_line: list[dict],
                    metadata: dict | None = None) -> None:
    """Write a complete gallery on the same filesystem, then publish with one rename."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".gallery-", dir=out.parent) as tmp:
        staging = Path(tmp) / "gallery"
        staging.mkdir()
        lines = []
        for i, (row, image, results) in enumerate(zip(rows, images, per_line, strict=True), 1):
            identity = line_identity(i, row)
            stem = staging / identity["file"]
            draw_panel(stem.with_suffix(".png"), [(row, image, results)], dpi=180)
            arrays = {f"{model}_{key}": values for model, (_, maps) in results.items() for key, values in maps.items()}
            np.savez_compressed(stem.with_suffix(".npz"), image=np.asarray(image), **arrays)
            lines.append({**identity, "probabilities": {model: prob.tolist() for model, (prob, _) in results.items()}})

        catalog = {"scribes": SCRIBES, "lines": lines}
        if metadata is None:
            indices = publication_examples(rows, images)
            caption = draw_panel(staging / PUBLICATION_FILE, [(rows[i], images[i], per_line[i]) for i in indices], dpi=300)
            catalog["publication_panel"] = {
                "file": PUBLICATION_FILE,
                "caption": caption,
                "selection_rule": "First gallery line and closest aspect ratio from another writer when available.",
                "examples": [
                    {"page_id": item["page_id"], "line_id": item["line_id"], "writer": item["writer"],
                     "gallery_file": f"{item['file']}.png", "maps_file": f"{item['file']}.npz"}
                    for item in (lines[i] for i in indices)
                ],
            }
        else:
            catalog.update(metadata)
        (staging / "catalog.json").write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(out)


def explain_lines(rows: list[CachedLine], models: list[str], files: dict[str, dict[str, Path]],
                  device: str, model_results_dir: Path | None):
    images = []
    for row in rows:
        with Image.open(row.image_path) as image:
            images.append(image.convert("L"))
    per_line = [{} for _ in rows]
    for model in models:
        out = model_results_dir / model / "explanations" if model_results_dir is not None else None
        inputs = {key: file_hash(path) for key, path in files[model].items()} if out is not None else {}
        if out is not None and out.exists():
            results = load_model_gallery(out, model, inputs, rows, images)
            print(f"Reusing {MODEL_LABELS[model]} line explanations at {out}", flush=True)
        else:
            results = generate_model_gallery(model, files[model]["checkpoint"], rows, images, device)
            if out is not None:
                publish_gallery(out, rows, images, [{model: result} for result in results],
                                metadata={"model": model, "inputs": inputs})
                print(f"Wrote {len(rows)} line explanations to {out}", flush=True)
        for combined, result in zip(per_line, results, strict=True):
            combined[model] = result
    return images, per_line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT / "results/whole_line")
    parser.add_argument("--model-results-dir", type=Path, help="also save each model's gallery in <model>/explanations")
    parser.add_argument("--out", type=Path, required=True, help="new directory for the explanation gallery")
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("Choose distinct models")
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"--out already exists: {args.out}")
    if args.model_results_dir is not None:
        combined_out = args.out.resolve()
        for model in args.models:
            model_out = (args.model_results_dir / model / "explanations").resolve()
            if (combined_out == model_out or combined_out in model_out.parents or model_out in combined_out.parents):
                parser.error("--out must be separate from the per-model explanation directories")
    files = {model: model_files(model, args.results_dir) for model in args.models}
    for paths in files.values():
        for path in paths.values():
            if not path.is_file():
                parser.error(f"Missing fitted-model artifact: {path}")

    rows = read_selection(args.cache_dir, args.selection)
    images, per_line = explain_lines(rows, args.models, files, args.device, args.model_results_dir)
    publish_gallery(args.out, rows, images, per_line)
    print(f"Wrote {len(rows)} line comparisons to {args.out}")


if __name__ == "__main__":
    main()

"""Compare saved line predictions on the full Nuremberg-9 test split.

Run from the repository root. CHURRO's block 16 predictions are included by default.
Each --model NAME=PATH archive must contain test_page_id and test_line_id (strings),
test_y (zero-based class indices), and test_prob (one probability column per scribe
in ascending scribe order). Rows are matched by (page_id, line_id).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[3]
SPLIT = Path("nuremberg-9/docs/split_v2_9scribes.json")
REFERENCE = (
    Path("nuremberg-9/experiments/01_balanced_probe/results/weighted_oversample")
    / "blk16_predictions.npz"
)


def load_predictions(path: Path, n_classes: int) -> tuple[dict[tuple[str, str], int], np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"{path}: expected an NPZ archive")
    with archive:
        fields = ("test_page_id", "test_line_id", "test_y", "test_prob")
        missing = set(fields) - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields {', '.join(sorted(missing))}")
        page_ids, line_ids, y, prob = (archive[field] for field in fields)

    if y.ndim != 1 or not len(y) or y.dtype.kind not in "iu":
        raise ValueError(f"{path}: test_y must be a nonempty vector of integer labels")
    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError(f"{path}: test_y must contain class indices from 0 to {n_classes - 1}")
    for name, values in (("test_page_id", page_ids), ("test_line_id", line_ids)):
        if values.shape != y.shape or values.dtype.kind != "U" or np.any(values == ""):
            raise ValueError(f"{path}: {name} must contain one nonempty Unicode string per label")
    if prob.shape != (len(y), n_classes) or prob.dtype.kind not in "fiu":
        raise ValueError(f"{path}: test_prob must be a numeric matrix of shape ({len(y)}, {n_classes})")
    if not np.isfinite(prob).all() or np.any(prob < 0) or np.any(prob > 1):
        raise ValueError(f"{path}: test_prob must contain finite values between 0 and 1")
    if not np.allclose(prob.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=1e-5):
        raise ValueError(f"{path}: each test_prob row must sum to 1")

    rows = {key: i for i, key in enumerate(zip(page_ids.tolist(), line_ids.tolist()))}
    if len(rows) != len(y):
        raise ValueError(f"{path}: duplicate (page_id, line_id) keys")
    return rows, y, prob


def align_predictions(
    path: Path, reference_rows: dict[tuple[str, str], int], reference_y: np.ndarray, n_classes: int
) -> np.ndarray:
    rows, y, prob = load_predictions(path, n_classes)
    missing, extra = reference_rows.keys() - rows.keys(), rows.keys() - reference_rows.keys()
    if missing or extra:
        raise ValueError(f"{path}: test lines differ from CHURRO ({len(missing)} missing, {len(extra)} extra)")
    order = [rows[key] for key in reference_rows]
    if np.any(y[order] != reference_y):
        raise ValueError(f"{path}: test labels differ from CHURRO for matching lines")
    return prob[order]


def scores(y: np.ndarray, prob: np.ndarray, scribes: list[int]) -> dict:
    predicted = prob.argmax(axis=1)
    labels = np.arange(len(scribes))
    matrix = confusion_matrix(y, predicted, labels=labels)
    support = matrix.sum(axis=1)
    present = support > 0
    recall = matrix.diagonal()[present] / support[present]
    return {
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1_score(y, predicted, labels=labels[present], average="macro", zero_division=0)),
        "support_per_scribe": support.tolist(),
        "recall_per_scribe": [float(matrix[i, i] / count) if count else None for i, count in enumerate(support)],
        "confusion_matrix": matrix.tolist(),
    }


def unseen_page_mask(split: dict, rows: dict[tuple[str, str], int]) -> np.ndarray:
    """Keep test lines whose page contributed no training or validation lines."""
    seen = set(split["splits"]["train"]) | set(split["splits"]["val"])
    for override in split.get("line_overrides", {}).values():
        for fold in ("train", "val"):
            seen.update(key.rsplit("|", 1)[0] for key in override.get(fold, []))
    return np.asarray([page not in seen for page, _ in rows], dtype=bool)


METRICS = ("macro_f1", "balanced_accuracy", "accuracy")


def differences(models: dict) -> dict:
    """Report percentage-point differences from CHURRO on exactly the same rows."""
    return {
        name: {metric: 100 * (model[metric] - models["churro"][metric]) for metric in METRICS}
        for name, model in models.items() if name != "churro"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                        help="compare an additional model's test predictions (repeatable)")
    parser.add_argument("--out", type=Path, help="write a copy of the JSON scores to this file")
    parser.add_argument("--figures", type=Path, metavar="DIR",
                        help="save test score, recall and confusion figures to this directory")
    args = parser.parse_args()

    model_paths = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        name = name.strip()
        if not name or not path.strip() or name == "churro" or name in model_paths:
            parser.error(f"--model requires a unique NAME=PATH entry, got {spec!r}")
        model_paths[name] = Path(path)

    try:
        split = json.loads((ROOT / SPLIT).read_text(encoding="utf-8"))
        scribes = sorted(split["scribes"])
        rows, y, prob = load_predictions(ROOT / REFERENCE, len(scribes))
        expected = [split["summary"]["test"][str(scribe)]["lines"] for scribe in scribes]
        if np.bincount(y.astype(np.int64), minlength=len(scribes)).tolist() != expected:
            raise ValueError("CHURRO test support differs from the Nuremberg-9 split")

        probabilities = {"churro": prob}
        for name, path in model_paths.items():
            probabilities[name] = align_predictions(ROOT / path, rows, y, len(scribes))
        models = {name: scores(y, prob, scribes) for name, prob in probabilities.items()}
        unseen = unseen_page_mask(split, rows)
        if not unseen.any():
            raise ValueError("The test split contains no pages unseen in training and validation")
        y_unseen = y[unseen]
        unseen_models = {name: scores(y_unseen, prob[unseen], scribes) for name, prob in probabilities.items()}

        report = {
            "split": SPLIT.as_posix(),
            "reference_predictions": REFERENCE.as_posix(),
            "scribes": scribes,
            "n_test_lines": len(y),
            "line_split_scribes": split["line_scribes"],
            "split_note": "S11 and S12 have 14 and 7 test lines. Their pages overlap across splits.",
            "models": models,
            "differences_from_churro_pp": differences(models),
            "unseen_pages": {
                "n_test_lines": len(y_unseen),
                "scribes_with_test_lines": [scribes[i] for i in np.unique(y_unseen)],
                "models": unseen_models,
                "differences_from_churro_pp": differences(unseen_models),
            },
        }
        result = json.dumps(report, indent=2, allow_nan=False)
        if args.figures is not None:
            from report_figures import render_figures

            render_figures(report, args.figures)
        if args.out is not None:
            args.out.write_text(result + "\n", encoding="utf-8")
    except (BadZipFile, EOFError, OSError, ValueError) as error:
        parser.error(str(error))
    print(result)


if __name__ == "__main__":
    main()

"""Fine-tune an image baseline on whole Nuremberg-9 lines."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

from compare import scores
from data import SCRIBES, load_rows
from whole_line import build_transform

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.common.runinfo import provenance  # noqa: E402

MODEL_NAMES = ("resnet18", "resnet50", "swin_tiny")
INPUT_VIEW = "whole_line"
MIN_F1_GAIN = 1e-4


def build_model(model_name: str, pretrained: bool = False) -> nn.Module:
    """Create the selected backbone with a fresh head in Nuremberg scribe order."""
    if model_name == "swin_tiny":
        import timm

        return timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained,
                                 num_classes=len(SCRIBES), strict_img_size=False)
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
    model.fc = nn.Linear(model.fc.in_features, len(SCRIBES))
    return model


class WholeLineDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row.image_path) as image:
            return self.transform(image), row.label


def group_by_shape(lines):
    """Collate full lines of equal size and retain their positions in the logical batch."""
    by_shape = defaultdict(list)
    for position, (pixels, label) in enumerate(lines):
        by_shape[tuple(pixels.shape[-2:])].append((position, pixels, label))
    groups = []
    for group in by_shape.values():
        positions, pixels, labels = zip(*group)
        groups.append((torch.tensor(positions), torch.stack(pixels), torch.tensor(labels)))
    return groups


def pixel_batches(groups, max_batch_pixels):
    """Limit the memory used by one forward without changing the sampled line set."""
    for positions, images, labels in groups:
        height, width = images.shape[-2:]
        size = max(1, max_batch_pixels // (height * width))
        yield from zip(positions.split(size), images.split(size), labels.split(size), strict=True)


def sampling_and_weights(rows, oversample, beta):
    """Compute effective-number weights from expected draws, as in the CHURRO probe."""
    labels = torch.tensor([row.label for row in rows])
    counts = torch.bincount(labels, minlength=len(SCRIBES))
    if not counts.all():
        raise ValueError("The training cache must include all nine scribes")
    expected = counts.double()
    sampler = None
    if oversample == "sqrt":
        probabilities = counts.double()[labels].rsqrt()
        probabilities /= probabilities.sum()
        expected = torch.bincount(labels, weights=probabilities, minlength=len(SCRIBES)) * len(rows)
        sampler = WeightedRandomSampler(probabilities, len(rows), replacement=True)
    weights = (1 - beta) / (1 - beta ** expected.clamp_min(1))
    return sampler, (weights * expected.sum() / (weights * expected).sum()).float(), counts


@torch.inference_mode()
def predict_lines(model, loader, device, max_batch_pixels):
    """Score each full line once and restore the original manifest order."""
    model.eval()
    predictions = []
    for groups in loader:
        batch_size = sum(len(labels) for _, _, labels in groups)
        batch = torch.empty((batch_size, len(SCRIBES)), dtype=torch.float32)
        for positions, images, _ in pixel_batches(groups, max_batch_pixels):
            logits = model(images.to(device, non_blocking=True))
            batch[positions] = logits.float().softmax(dim=1).cpu()
        predictions.append(batch)
    return torch.cat(predictions).numpy()


def macro_f1(labels, predictions):
    return float(f1_score(labels, predictions.argmax(axis=1), labels=range(len(SCRIBES)),
                          average="macro", zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", required=True, help="New output directory for this run")
    parser.add_argument("--batch-size", type=int, default=32, help="Lines per optimizer step")
    parser.add_argument("--max-batch-pixels", type=int, default=1_003_520,
                        help="Maximum pixels per forward, unless a single line is larger")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--oversample", choices=("none", "sqrt"), default="sqrt")
    parser.add_argument("--beta", type=float, default=0.999, help="Effective-number loss parameter")
    parser.add_argument("--lr", type=float, help="Defaults to 3e-4 for ResNet and 1e-4 for Swin")
    parser.add_argument("--weight-decay", type=float, help="Defaults to 1e-4 for ResNet and 0.05 for Swin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip downloads for smoke tests")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        parser.error(f"--out already exists: {out}")
    if min(args.batch_size, args.max_batch_pixels, args.epochs, args.patience) < 1:
        parser.error("Batch and pixel budgets, epochs and stopping patience must be positive")
    if min(args.workers, args.lr_patience) < 0 or not 0 <= args.beta < 1:
        parser.error("Workers and LR patience must be nonnegative, and beta in [0, 1)")
    swin = args.model == "swin_tiny"
    args.lr = args.lr if args.lr is not None else (1e-4 if swin else 3e-4)
    args.weight_decay = args.weight_decay if args.weight_decay is not None else (0.05 if swin else 1e-4)
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    transform = build_transform()
    splits = ("train", "val", "test")
    rows = {split: load_rows(Path(args.cache_dir), split) for split in splits}
    if not all(rows.values()):
        raise ValueError("The line cache must have train, val and test examples")
    labels = {split: np.asarray([row.label for row in lines], dtype=np.int64) for split, lines in rows.items()}
    datasets = {split: WholeLineDataset(rows[split], transform) for split in splits}
    sampler, weights, counts = sampling_and_weights(rows["train"], args.oversample, args.beta)
    options = {"num_workers": args.workers, "pin_memory": amp, "persistent_workers": args.workers > 0,
               "collate_fn": group_by_shape}
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, sampler=sampler,
                              shuffle=sampler is None, **options)
    eval_loaders = {split: DataLoader(datasets[split], batch_size=args.batch_size, **options)
                    for split in ("val", "test")}

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience
    )
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), reduction="sum")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    best_state, best_epoch, best_score = None, 0, -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for groups in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch_size = sum(len(batch_labels) for _, _, batch_labels in groups)
            weight_sum = sum(float(weights[batch_labels].sum()) for _, _, batch_labels in groups)
            for _, images, batch_labels in pixel_batches(groups, args.max_batch_pixels):
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(images.to(device, non_blocking=True))
                    loss = criterion(logits, batch_labels.to(device, non_blocking=True)) / weight_sum
                scaler.scale(loss).backward()
                total_loss += loss.detach() * batch_size
            scaler.step(optimizer)
            scaler.update()
        val_prob = predict_lines(model, eval_loaders["val"], device, args.max_batch_pixels)
        score = macro_f1(labels["val"], val_prob)
        avg_loss = total_loss.item() / len(rows["train"])
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_macro_f1": score,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch}: train loss {avg_loss:.4f}, validation macro-F1 {score:.4f}", flush=True)
        schedule.step(score)
        if score > best_score + MIN_F1_GAIN:
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch, best_score = epoch, score
        elif epoch - best_epoch >= args.patience:
            break

    model.load_state_dict(best_state)
    saved = {"scribes": np.asarray(SCRIBES, dtype=np.int64)}
    evaluation = {}
    for split in ("val", "test"):
        probabilities = predict_lines(model, eval_loaders[split], device, args.max_batch_pixels)
        saved.update({f"{split}_prob": probabilities, f"{split}_y": labels[split],
                      f"{split}_page_id": np.asarray([str(row.page_id) for row in rows[split]], dtype=str),
                      f"{split}_line_id": np.asarray([str(row.line_id) for row in rows[split]], dtype=str)})
        evaluation[split] = {"lines": len(rows[split]), **scores(labels[split], probabilities, list(SCRIBES))}

    checkpoint = {"model": best_state, "model_name": args.model, "scribes": list(SCRIBES),
                  "input_view": INPUT_VIEW,
                  "weights": weights.tolist(), "best_epoch": best_epoch}
    summary = {"model": args.model, "scribes": list(SCRIBES), "best_epoch": best_epoch,
               "input_view": INPUT_VIEW,
               "train_lines_per_scribe": counts.tolist(), "weights": weights.tolist(),
               "history": history, **evaluation, "run": provenance(vars(args), ROOT)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".baseline-", dir=out.parent) as tmp:
        staging = Path(tmp) / "run"
        staging.mkdir()
        torch.save(checkpoint, staging / "best.pt")
        np.savez_compressed(staging / "predictions.npz", **saved)
        (staging / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if out.exists():
            raise FileExistsError(f"Output already exists: {out}")
        staging.rename(out)
    print(f"Best validation macro-F1 {best_score:.4f} at epoch {best_epoch}. Results in {out}")


if __name__ == "__main__":
    main()

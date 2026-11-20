"""CHURRO Grad-CAM and received self-attention on Nuremberg-9 test lines."""

from __future__ import annotations

import warnings

import numpy as np
from PIL import Image

from compare import load_predictions
from data import CHURRO_PREDICTIONS, SCRIBES, CachedLine


class ChurroExplainer:
    """Use the stored nine-scribe head and the existing encoder's token grid."""

    def __init__(self, device: str):
        import torch

        from src.models.churro_encoder import ChurroEncoder
        from src.models.churro_explain import load_head

        self.rows, self.labels, self.probabilities = load_predictions(CHURRO_PREDICTIONS, len(SCRIBES))
        self.head = load_head(CHURRO_PREDICTIONS.with_name("blk16_head.npz"), device)
        self.encoder = ChurroEncoder(device=device, dtype=torch.float32)
        self.encoder.visual.requires_grad_(False)

    def explain(self, image: Image.Image, row: CachedLine) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return stored probabilities and two maps in the original patch-grid order."""
        import torch

        from src.models.churro_explain import attention_received, to_grid, trace

        key = (row.page_id, row.line_id)
        index = self.rows.get(key)
        if index is None or self.labels[index] != row.label:
            raise ValueError(f"Line {key!r} does not match the canonical CHURRO test predictions")
        probability = self.probabilities[index]
        target = int(probability.argmax())

        encoder = self.encoder
        batch = encoder.processor(
            images=[image.convert("L").convert("RGB")], size=encoder.size, return_tensors="pt"
        )
        pixels = batch["pixel_values"].to(encoder.device, encoder.dtype)
        grid = batch["image_grid_thw"].to(encoder.device)
        with torch.enable_grad():
            state = trace(encoder.visual, pixels, grid)
            features = (state["T"].mean(0) - self.head["mean"]) / self.head["std"]
            logits = self.head["W"] @ features + self.head["b"]
            activations = state["A"]
            gradient, = torch.autograd.grad(logits[target], activations)
            cam = torch.relu((activations.detach() * gradient.mean(0)).sum(1))

        rerun = int(logits.detach().argmax())
        if rerun != target:
            warnings.warn(
                f"CHURRO float32 tracing predicted S{SCRIBES[rerun]} for {key!r}, "
                f"compared with stored S{SCRIBES[target]}. Grad-CAM targets S{SCRIBES[target]}.",
                RuntimeWarning,
                stacklevel=2,
            )

        attention = attention_received(state["qk"])
        if attention is None:
            raise RuntimeError(f"CHURRO block 16 self-attention could not be captured for {key!r}")
        order, height, width = state["window_index"], state["h"], state["w"]
        return probability, {
            "gradcam": to_grid(cam, order, height, width),
            "attention": to_grid(attention, order, height, width),
        }

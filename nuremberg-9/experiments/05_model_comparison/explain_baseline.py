"""Replay baseline checkpoints to explain the saved line prediction."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from compare import load_predictions
from data import SCRIBES, CachedLine
from train_baseline import build_model
from whole_line import build_transform, swin_attention_map


class BaselineExplainer:
    def __init__(self, checkpoint: Path, device: str):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if tuple(saved["scribes"]) != SCRIBES:
            raise ValueError(f"{checkpoint}: checkpoint class order differs from the nine-scribe split")
        if saved.get("input_view") != "whole_line":
            raise ValueError(f"{checkpoint}: expected a whole-line checkpoint")

        self.model_name = saved["model_name"]
        self.device = torch.device(device)
        self.transform = build_transform()
        self.rows, self.labels, self.probabilities = load_predictions(
            checkpoint.with_name("predictions.npz"), len(SCRIBES)
        )

        self.model = build_model(self.model_name, pretrained=False)
        self.model.load_state_dict(saved["model"])
        self.model.to(self.device).eval().requires_grad_(False)
        self.swin_block = None
        if self.model_name == "swin_tiny":
            self.swin_block = self.model.layers[-1].blocks[-1]
            self.target_layer = self.swin_block.norm1
            if hasattr(self.swin_block.attn, "fused_attn"):
                self.swin_block.attn.fused_attn = False
        else:
            self.target_layer = self.model.layer4[-1]

    @torch.enable_grad()
    def explain(self, image: Image.Image, row: CachedLine) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Explain the saved class using the whole-line score and the model's native token grid."""
        key = (row.page_id, row.line_id)
        if key not in self.rows:
            raise ValueError(f"{self.model_name}: no prediction for {row.page_id}/{row.line_id}")
        index = self.rows[key]
        if self.labels[index] != row.label:
            raise ValueError(f"{self.model_name}: wrong label for {row.page_id}/{row.line_id}")
        probability = self.probabilities[index]
        target = int(probability.argmax())

        activation = None
        attention = None

        def capture_activation(_module, _args, output):
            nonlocal activation
            # The class score needs gradients from this layer forward.
            activation = output.detach().requires_grad_(True)
            return activation

        def capture_attention(_module, _args, output):
            nonlocal attention
            attention = output.detach()

        with ExitStack() as hooks:
            hooks.enter_context(self.target_layer.register_forward_hook(capture_activation))
            if self.swin_block is not None:
                hooks.enter_context(self.swin_block.attn.softmax.register_forward_hook(capture_attention))
            inputs = self.transform(image).unsqueeze(0).to(self.device)
            logits = self.model(inputs)

        if activation is None:
            raise RuntimeError(f"{self.model_name}: Grad-CAM target layer was not reached")
        gradient, = torch.autograd.grad(logits[0, target], activation)
        features = activation.detach()
        maps = {}
        if self.swin_block is not None:
            if attention is None or any(self.swin_block.shift_size):
                raise RuntimeError("Expected attention weights from the unshifted last Swin block")
            height, width = features.shape[1:3]
            maps["attention"] = swin_attention_map(attention, height, width, self.swin_block.window_size)
            features = features.permute(0, 3, 1, 2)
            gradient = gradient.permute(0, 3, 1, 2)

        weights = gradient.mean(dim=(-2, -1), keepdim=True)
        maps["gradcam"] = (features * weights).sum(dim=1).relu()
        return probability, {name: values[0].float().cpu().numpy() for name, values in maps.items()}

"""Shared full-line image geometry and Swin attention projection."""

from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


def line_view(image: Image.Image) -> Image.Image:
    """Resize the grayscale crop to the geometry of the CHURRO vision processor."""
    height, width = smart_resize(
        image.height, image.width, factor=28, min_pixels=50_176, max_pixels=1_003_520,
    )
    return image.convert("L").convert("RGB").resize((width, height), Image.Resampling.BICUBIC)


def build_transform() -> transforms.Compose:
    """Use whole-line geometry and the normalization of the ImageNet backbones."""
    return transforms.Compose([
        line_view,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def swin_attention_map(
    attention: torch.Tensor, height: int, width: int, window_size: tuple[int, int],
) -> torch.Tensor:
    """Average attention received from real queries on the unshifted Swin token grid."""
    window_h, window_w = window_size
    rows = (height + window_h - 1) // window_h
    cols = (width + window_w - 1) // window_w
    windows = rows * cols
    tokens = window_h * window_w
    if attention.ndim != 4 or attention.shape[0] % windows or attention.shape[-2:] != (tokens, tokens):
        raise ValueError(f"Swin attention shape {tuple(attention.shape)} does not match the token grid")

    # Reproduce the spatial grouping of Swin's window partition. Exclude
    # synthetic queries introduced by Swin when the grid does not fill a window.
    query_rows = torch.arange(rows * window_h, device=attention.device) < height
    query_cols = torch.arange(cols * window_w, device=attention.device) < width
    queries = (query_rows[:, None] & query_cols).reshape(rows, window_h, cols, window_w)
    queries = queries.permute(0, 2, 1, 3).reshape(windows, tokens)

    window_attention = attention.float().mean(dim=1).reshape(-1, windows, tokens, tokens)
    received = (window_attention * queries[None, :, :, None]).sum(dim=-2)
    received /= queries.sum(dim=-1)[None, :, None]
    grid = received.reshape(-1, rows, cols, window_h, window_w).permute(0, 1, 3, 2, 4)
    return grid.reshape(-1, rows * window_h, cols * window_w)[:, :height, :width]

"""Trace the CHURRO vision blocks used by the writer-classification explanations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

CAM_BLOCK = 7        # Grad-CAM taken at the output of block 8 (index 7)
FEAT_BLOCK = 15      # the blk16 feature (index 15), a full-attention block


class _Stop(Exception):
    """Raised by a hook to stop the forward pass once block 16 is done."""


def to_grid(values: torch.Tensor, window_index: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Per-token values in the encoder's window order -> (h, w) map over the 14-px patch grid.

    Tokens come in groups of 2x2 patches (merge units) that the encoder shuffles by `window_index`."""
    nu = values.shape[0] // 4
    unit = values.reshape(nu, 4)
    orig = torch.empty_like(unit)
    orig[window_index.to(unit.device)] = unit
    return orig.reshape(h // 2, w // 2, 2, 2).permute(0, 2, 1, 3).reshape(h, w).cpu().numpy()


def get_window_index(visual, grid: torch.Tensor, modeling) -> torch.Tensor:
    """The encoder's shuffle of 2x2 merge units (a method in transformers 4.x, a helper function in 5.x)."""
    if hasattr(visual, "get_window_index"):
        return visual.get_window_index(grid)[0]
    return modeling.get_vision_window_index(grid, spatial_merge_size=visual.spatial_merge_size,
                                            window_size=visual.window_size, patch_size=visual.patch_size)[0]


def trace(visual, pixels: torch.Tensor, grid: torch.Tensor):
    """Forward up to block 16 and keep what the maps need.

    Returns dict: A (output of block 8, leaf that requires grad), T (output of block 16, in the graph),
    window_index, q and k of the block-16 attention (after rotary embedding), (h, w) patch grid."""
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as modeling
    state: dict = {}
    rec: dict = {"on": False, "out": []}
    orig_rot = modeling.apply_rotary_pos_emb_vision

    def wrapped(*a, **k):
        out = orig_rot(*a, **k)
        if rec["on"]:
            rec["out"].extend([x.detach() for x in out] if isinstance(out, tuple) else [out.detach()])
        return out

    def cam_hook(module, args, output):
        t = output[0] if isinstance(output, tuple) else output
        leaf = t.detach().requires_grad_(True)
        state["A"] = leaf
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    def feat_pre(module, args, kwargs):
        rec["on"] = True

    def feat_hook(module, args, output):
        rec["on"] = False
        state["T"] = output[0] if isinstance(output, tuple) else output
        raise _Stop

    h1 = visual.blocks[CAM_BLOCK].register_forward_hook(cam_hook)
    h2 = visual.blocks[FEAT_BLOCK].register_forward_pre_hook(feat_pre, with_kwargs=True)
    h3 = visual.blocks[FEAT_BLOCK].register_forward_hook(feat_hook)
    modeling.apply_rotary_pos_emb_vision = wrapped
    try:
        with torch.enable_grad():
            try:
                visual(pixels, grid_thw=grid)
            except _Stop:
                pass
    finally:
        modeling.apply_rotary_pos_emb_vision = orig_rot
        for h in (h1, h2, h3):
            h.remove()
    if "T" not in state:
        raise RuntimeError("block 16 was not reached")
    _, gh, gw = grid[0].tolist()
    window_index = get_window_index(visual, grid, modeling)
    state.update(window_index=window_index, h=gh, w=gw, qk=rec["out"])
    return state


def attention_received(qk: list) -> torch.Tensor | None:
    """(n,) share of attention each token receives, averaged over query tokens and heads."""
    if len(qk) < 2:
        return None
    q, k = qk[0], qk[1]
    if q.dim() == 4:
        q, k = q.squeeze(0), k.squeeze(0)               # (n, heads, head_dim)
    q, k = q.float().permute(1, 0, 2), k.float().permute(1, 0, 2)
    scale = q.shape[-1] ** -0.5
    total = torch.zeros(q.shape[1], device=q.device)
    for head in range(q.shape[0]):
        total += torch.softmax(q[head] @ k[head].T * scale, dim=-1).mean(0)
    return total / q.shape[0]


def load_head(path: Path, device) -> dict:
    with np.load(path, allow_pickle=False) as arrays:
        return {key: torch.tensor(arrays[key], dtype=torch.float32, device=device) for key in ("W", "b", "mean", "std")}

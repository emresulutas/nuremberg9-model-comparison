"""CHURRO vision encoder only (the language model is never loaded).

CHURRO-3B is a fine-tune of Qwen2.5-VL-3B. Its safetensors hold the vision tower under the
prefix ``visual.``; we build just that tower from the config and load only those weights.

For one line image the encoder returns mean-pooled token vectors from several depths:
    blk8, blk16, blk24, blk32   output of transformer blocks 7/15/23/31 (1280-d)
    out                         output of the 2x2 patch merger (2048-d), what the LM would see
Mean pooling over all tokens turns a variable-size line into one fixed vector.

Turing GPUs (RTX 2080 Ti) have no bf16, so the weights are cast to fp16 and attention is sdpa.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID = "stanford-oval/churro-3B"
BLOCKS = {"blk8": 7, "blk16": 15, "blk24": 23, "blk32": 31}     # name -> block index
FEATURES = (*BLOCKS, "out")
DIMS = {**{name: 1280 for name in BLOCKS}, "out": 2048}


def snapshot_path(model_id: str = MODEL_ID) -> Path:
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(model_id, local_files_only=True))
    except Exception:                      # not cached yet: download once
        return Path(snapshot_download(model_id))


class ChurroEncoder:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", min_pixels: int = 50176,
                 max_pixels: int = 1_003_520, dtype: torch.dtype = torch.float16):
        from safetensors import safe_open
        from transformers import AutoConfig, AutoImageProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

        path = snapshot_path(model_id)
        vcfg = AutoConfig.from_pretrained(path).vision_config
        try:
            vcfg._attn_implementation = "sdpa"
        except Exception:
            pass
        self.visual = Qwen2_5_VisionTransformerPretrainedModel(vcfg)

        weight_map = json.load(open(path / "model.safetensors.index.json", encoding="utf-8"))["weight_map"]
        keys_by_file: dict[str, list[str]] = defaultdict(list)
        for key, fname in weight_map.items():
            if key.startswith("visual."):
                keys_by_file[fname].append(key)
        state = {}
        for fname, keys in keys_by_file.items():
            with safe_open(str(path / fname), framework="pt") as sf:
                for key in keys:
                    state[key[len("visual."):]] = sf.get_tensor(key)
        self.visual.load_state_dict(state, strict=True)       # fails loudly if any weight is missing
        self.visual = self.visual.to(dtype).to(device).eval()      # fp16 by default; fp32 repairs overflowed lines

        self.device = device
        self.dtype = dtype
        self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}   # = min/max pixels
        self.processor = AutoImageProcessor.from_pretrained(path)
        self._pooled: dict[str, torch.Tensor] = {}
        for name, idx in BLOCKS.items():
            self.visual.blocks[idx].register_forward_hook(self._hook(name))

    def _hook(self, name: str):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            self._pooled[name] = out.reshape(-1, out.shape[-1]).float().mean(dim=0)
        return hook

    def _forward(self, image: Image.Image):
        batch = self.processor(images=[image.convert("RGB")], size=self.size, return_tensors="pt")
        pixels = batch["pixel_values"].to(self.device, self.dtype)
        grid = batch["image_grid_thw"].to(self.device)
        merged = self.visual(pixels, grid_thw=grid)
        if not torch.is_tensor(merged):                      # newer transformers wrap the result
            merged = getattr(merged, "pooler_output", None) if getattr(merged, "pooler_output", None) is not None \
                else merged.last_hidden_state
        merged = merged.reshape(-1, merged.shape[-1])
        self._pooled["out"] = merged.float().mean(dim=0)
        return merged, batch["image_grid_thw"][0].tolist()   # grid = (t, h, w) in 14-px patches

    @torch.inference_mode()
    def encode(self, image: Image.Image) -> dict[str, np.ndarray]:
        """One PIL line image -> {feature name: float16 vector}."""
        self._forward(image)
        return {k: v.cpu().numpy().astype(np.float16) for k, v in self._pooled.items()}

    @torch.inference_mode()
    def encode_tokens(self, image: Image.Image):
        """Merged tokens of one line, for the evidence maps of explain.py.

        Returns (tokens (n, 2048) float32, (rows, cols) of the merged token grid, (width, height) in pixels
        of the resized image the encoder saw). The merger output is in raster order of 2x2 patch blocks."""
        merged, (_, h, w) = self._forward(image)
        merge = self.visual.config.spatial_merge_size
        patch = self.visual.config.patch_size
        return merged.float().cpu().numpy(), (h // merge, w // merge), (w * patch, h * patch)

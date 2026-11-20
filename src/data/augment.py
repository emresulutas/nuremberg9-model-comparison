"""Augmentation of line crops for plan step 4 (rare scribes only, train lines only).

A line crop (PIL image from `nuremberg_lines.crop_line`) goes in, a slightly changed copy comes out. The copy
then goes through the frozen encoder like any line, so cached features of the original stay valid.

What is changed (all copies, every transform keeps the shape of the letters):
  rotation  angle ~ Normal(0, max_rotate / 3) clipped to +-max_rotate (default 10 deg, so sigma 3.3 like the +-3 deg usual for documents): most copies are within
            +-3 deg, an angle near 10 deg is rare (the first preview used 30 deg: letters stay intact, but the ink shrinks in a big empty canvas). The canvas grows so no ink is cut off and the new corners are
            filled with the paper colour of the line (the same colour `crop_line` uses outside the polygon).
  scale     0.92 .. 1.08, the same factor on both axes (no stretching of letters).
  brightness / contrast +-20 %, light Gaussian blur, light Gaussian noise, JPEG compression: scan differences.
Deliberately NOT used: shear (slant is a writer cue), elastic or grid distortion (changes stroke shape),
perspective, flips. The grey conversion happens later, in the feature extraction (`--gray`).

Needs albumentations 2.x (pip install "albumentations>=2.0,<3").
"""
from __future__ import annotations

import zlib

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageStat

MAX_ROTATE = 10.0


def paper_colour(img: Image.Image) -> tuple[int, int, int]:
    return tuple(int(v) for v in ImageStat.Stat(img.convert("RGB")).median)


def key_rng(page_id: str, line_id: str, copy: int, seed: int = 42) -> np.random.Generator:
    """Same line + same copy number -> same copy, in every run."""
    return np.random.default_rng([seed, copy, zlib.crc32(f"{page_id}|{line_id}".encode())])


def _affine(arr: np.ndarray, angle: float, scale: float, fill) -> np.ndarray:
    t = A.Affine(rotate=(angle, angle), scale=(scale, scale), keep_ratio=True, fit_output=True,
                 border_mode=cv2.BORDER_CONSTANT, fill=fill, interpolation=cv2.INTER_LINEAR, p=1.0)
    return t(image=arr)["image"]


def rotate_only(img: Image.Image, angle: float) -> Image.Image:
    """Only the rotation (for the preview of how far the angle may go)."""
    return Image.fromarray(_affine(np.array(img.convert("RGB")), angle, 1.0, paper_colour(img)))


def augment_line(img: Image.Image, rng: np.random.Generator, max_rotate: float = MAX_ROTATE) -> Image.Image:
    angle = float(np.clip(rng.normal(0.0, max_rotate / 3.0), -max_rotate, max_rotate))
    scale = float(rng.uniform(0.92, 1.08))
    arr = _affine(np.array(img.convert("RGB")), angle, scale, paper_colour(img))
    photo = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.3, 1.0), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.3),
        A.ImageCompression(quality_range=(50, 90), p=0.3),
    ], seed=int(rng.integers(0, 2**31 - 1)))
    return Image.fromarray(photo(image=arr)["image"])

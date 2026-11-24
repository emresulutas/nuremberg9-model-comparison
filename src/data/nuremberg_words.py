"""Word boxes inside a text line, without word-level labels.

The PAGE XML has no <Word> elements, only the transcription of each line. So the number of words in a line
is known; the image only has to say where the word breaks are. Method (no torch, only numpy):

1. ink = pixels darker than the Otsu threshold of the line crop (grey).
2. column profile = ink pixels per column, smoothed; low = (almost) no ink. In cursive writing the gaps are
   rarely empty (slanted strokes, long s, tails), so a gap is a valley, not a hole.
3. n words need n-1 cuts. They are placed by dynamic programming: every cut sits in a low valley and the
   cuts keep roughly the order and proportions of the words' letter counts (a long word is wide).
4. `quality` = how deep the chosen valleys are (0 = empty column, 1 = as busy as a typical column).
   Callers keep the lines whose quality is below a threshold; nothing is guessed silently.
5. word i of the transcription is the i-th box from the left.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

SMOOTH_REL = 0.12      # profile smoothing width, as a share of the ink height of the line
PRIOR = 0.15           # cost per squared (average word width) of distance from the length-proportional position
MIN_WORD_REL = 0.30    # a word is at least this share of the ink height wide


def tokens(text: str) -> list[str]:
    """Words of a transcription: NFKC (long s -> s), lower case, punctuation removed, empty tokens dropped."""
    out = []
    for raw in text.split():
        norm = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", raw).lower())
        if norm:
            out.append(norm)
    return out


def otsu(gray: np.ndarray) -> int:
    """Otsu threshold of a uint8 image (the grey level that best separates dark ink from light paper)."""
    p = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p /= p.sum()
    omega, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    return int(np.nanargmax(between))


def split_words(gray: np.ndarray, words: list[str]) -> tuple[list[tuple[int, int]], float] | None:
    """(boxes, quality): column ranges (x0, x1) of the words, left to right, and how deep the cut valleys are."""
    n = len(words)
    if n < 1:
        return None
    ink = gray <= otsu(gray)
    rows = np.where(ink.any(1))[0]
    if len(rows) == 0:
        return None
    height = int(rows[-1] - rows[0] + 1)
    raw = ink.sum(0).astype(np.float64)
    width = max(3, int(SMOOTH_REL * height))
    prof = np.convolve(raw, np.ones(width) / width, mode="same")
    typical = float(np.percentile(prof[prof > 0], 90)) if (prof > 0).any() else 0.0
    if typical <= 0:
        return None
    v = prof / typical
    cols = np.where(raw > 0)[0]
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    if n == 1:
        return [(x0, x1)], 0.0
    if x1 - x0 < n * 3:
        return None
    avg_w = (x1 - x0) / n
    min_w = max(2, int(MIN_WORD_REL * height))
    weights = np.array([len(w) + 1 for w in words], dtype=np.float64)       # +1: the space after the word
    expect = x0 + (x1 - x0) * np.cumsum(weights)[:-1] / weights.sum()       # expected cut positions
    xs = np.arange(x0, x1)
    inf = np.inf
    dp = np.full((n - 1, len(xs)), inf)
    back = np.zeros((n - 1, len(xs)), dtype=np.int64)
    for k in range(n - 1):
        cost = v[x0:x1] + PRIOR * ((xs - expect[k]) / avg_w) ** 2
        if k == 0:
            ok = xs - x0 >= min_w
            dp[0] = np.where(ok, cost, inf)
            continue
        prev = dp[k - 1]
        best = np.minimum.accumulate(prev)                                    # best previous cost at <= x
        arg = np.maximum.accumulate(np.where(prev <= best, np.arange(len(xs)), 0))   # where that best sits
        j = np.arange(len(xs)) - min_w                                        # previous cut at least min_w left
        jj = np.where(j >= 0, j, 0)
        ok = (j >= 0) & np.isfinite(best[jj])
        dp[k] = np.where(ok, cost + best[jj], inf)
        back[k] = np.where(ok, arg[jj], 0)
    last_ok = xs[-1] - xs >= min_w
    final = np.where(last_ok, dp[n - 2], inf)
    if not np.isfinite(final).any():
        return None
    i = int(np.argmin(final))
    cuts = [0] * (n - 1)
    for k in range(n - 2, -1, -1):
        cuts[k] = i
        i = int(back[k, i])
    cut_x = [int(xs[c]) for c in cuts]
    quality = float(np.mean([v[c] for c in cut_x]))
    edges = [x0] + cut_x + [x1]
    boxes = []
    for a, b in zip(edges[:-1], edges[1:]):                                   # trim every word to its own ink
        sub = np.where(raw[a:b] > 0)[0]
        if len(sub) == 0:
            return None
        boxes.append((a + int(sub[0]), a + int(sub[-1]) + 1))
    return boxes, quality

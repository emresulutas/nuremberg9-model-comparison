"""Count and plot exact four-model error intersections on aligned test lines."""
from __future__ import annotations

import numpy as np
from matplotlib import patheffects
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import distance_transform_edt

CIRCLES = ((.35, .45), (.50, .60), (.65, .45), (.50, .30))
RADIUS = .26
COLORS = ("#464e9a", "#08786d", "#9a4e2b", "#a34076")


def exact_counts(y, predictions):
    if len(predictions) != 4:
        raise ValueError("The diagram requires four aligned models")
    wrong_pattern = np.zeros(len(y), dtype=np.uint8)
    for i, prob in enumerate(predictions):
        wrong_pattern |= (prob.argmax(axis=1) != y).astype(np.uint8) << i
    return np.bincount(wrong_pattern, minlength=16)


def label_positions():
    """Locate labels for the four single-model regions and the common center."""
    grid = np.linspace(0, 1, 650)
    xx, yy = np.meshgrid(grid, grid)
    masks = np.zeros(xx.shape, dtype=np.uint8)
    for i, (x, y) in enumerate(CIRCLES):
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= RADIUS ** 2
        masks |= inside.astype(np.uint8) << i
    positions = {}
    for mask in (1, 2, 4, 8, 15):
        zone = masks == mask
        if not zone.any():
            raise ValueError(f"No region for intersection {mask:04b}")
        distance = distance_transform_edt(zone)
        row, col = np.unravel_index(distance.argmax(), zone.shape)
        positions[mask] = (grid[col], grid[row])
    return positions


def draw(counts, names):
    fig = plt.figure(figsize=(11.8, 6.15), facecolor="white")
    ink, subtle = "#172c40", "#4b6173"
    fig.text(.045, .92, "Which errors did the four models share?", weight="bold", fontsize=24, color=ink)
    fig.text(.045, .865, f"{counts.sum():,} whole-line test predictions from unseen pages · Seven writers",
             fontsize=11, color=subtle)

    venn = fig.add_axes([.035, .16, .545, .62])
    venn.set(xlim=(.04, .96), ylim=(.01, .89), aspect="equal")
    venn.axis("off")
    for center, color in zip(CIRCLES, COLORS, strict=True):
        venn.add_patch(Circle(center, RADIUS, facecolor=color, alpha=.07, linewidth=0))
        venn.add_patch(Circle(center, RADIUS, fill=False, edgecolor=color, linewidth=1.6, alpha=.88))

    positions = label_positions()
    for i, color in enumerate(COLORS):
        mask = 1 << i
        venn.text(*positions[mask], f"{counts[mask]:,}", ha="center", va="center", size=17,
                  weight="bold", color=color,
                  path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])
    x, y = positions[15]
    venn.text(x, y + .018, f"{counts[15]:,}", ha="center", va="center", size=26, weight="bold", color=ink)
    venn.text(x, y - .046, "ALL FOUR\nWRONG", ha="center", va="center", size=8, weight="bold", color=ink)

    table = fig.add_axes([.592, .235, .368, .525])
    table.set(xlim=(0, 1), ylim=(0, 1))
    table.axis("off")
    table.text(0, .98, "When one model stood apart", fontsize=14, weight="bold", color=ink)
    table.text(.59, .80, "Only this model\nwas wrong", ha="center", va="center", fontsize=10, color=subtle)
    table.text(.88, .80, "Only this model\nwas correct", ha="center", va="center", fontsize=10, color=subtle)
    for i, (color, name) in enumerate(zip(COLORS, names, strict=True)):
        y = .65 - .16 * i
        table.axhline(y + .076, color="#dbe3e9", linewidth=.8)
        table.text(0, y, name, va="center", fontsize=11, weight="bold", color=color)
        table.text(.59, y, f"{counts[1 << i]:,}", ha="center", va="center", fontsize=14, color=ink)
        table.text(.88, y, f"{counts[15 ^ (1 << i)]:,}", ha="center", va="center", fontsize=14, color=ink)

    two_or_three = sum(counts[m] for m in range(1, 15) if m.bit_count() in (2, 3))
    fig.text(.045, .125, "Each circle represents one model’s errors. Only selected regions are labeled.",
             fontsize=10, color=subtle)
    fig.text(.045, .061,
             f"{counts[0]:,} lines were correct for all four. The CSV lists all overlaps, including {two_or_three:,} lines missed by two or three models.",
             fontsize=9.5, color=subtle)
    return fig

def draw_intersections(counts, names):
    from itertools import combinations

    fig, ax = plt.subplots(figsize=(11, 10))
    fig.subplots_adjust(left=.07, right=.95, top=.84, bottom=.07)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.text(.07, .94, "Which models missed the same lines?", fontsize=24, weight="bold", color="#172c40")
    fig.text(.07, .895, f"{sum(counts):,} unseen-page test lines  ·  Each line is counted once. All unlisted models were correct.", fontsize=12)
    y = .98
    for size, label in [(2, "Exactly two models wrong"), (3, "Exactly three models wrong"), (4, "All four models wrong")]:
        groups = [(g, counts[sum(1 << i for i in g)]) for g in combinations(range(4), size)]
        ax.text(0, y, label, weight="bold", fontsize=16, color="#172c40")
        y -= .053
        for members, count in sorted(groups, key=lambda item: -item[1]):
            ax.text(.015, y, " + ".join(names[i] for i in members), va="center", fontsize=12)
            ax.barh(y, .29 * count / max(counts[15], max(n for _, n in groups)), left=.61, height=.018, color="#367a82")
            ax.text(.97, y, str(count), va="center", ha="right", weight="bold", fontsize=13)
            y -= .054
        y -= .044
    singles = sum(counts[1 << i] for i in range(4))
    fig.text(.07, .045, f"Other lines: exactly one model wrong on {singles:,}  ·  All four correct on {counts[0]:,}.", fontsize=12)
    return fig

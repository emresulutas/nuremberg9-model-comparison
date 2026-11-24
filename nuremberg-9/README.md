# nuremberg-9

Scribe identification on the Nuremberg Letterbooks with 9 scribes: 1, 2, 6, 7, 8, 9, 10 plus S11 and S12
(S13 has only 2 lines and is dropped). Same frozen CHURRO-3B encoder and linear head as `../nuremberg-7/`.

Planned content (plan steps 2-5, see `../docs/project_state.md`):
- `docs/split_v2_9scribes.json` keeps the original seven writers' page assignments. S11 and S12 are split by line, so their pages can occur in multiple splits.
- later: imbalance handling, augmentation, line-based vs word-based final training.

The [four-model comparison](experiments/05_model_comparison/README.md) uses this split for CHURRO,
ResNet18, ResNet50 and Swin-Tiny. It includes shared scoring and an explanation gallery on the same test lines.

The [paper report](report/README.md) organizes the results by model, the four-model benchmark, and the ResNet18 vs CHURRO-3B follow-up.

Shared code stays in `../src/`; run everything from the repo root.

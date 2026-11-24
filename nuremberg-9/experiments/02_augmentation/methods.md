# Step 4: augmentation of rare scribes

Setup: same head as step 3 (blk16, class-balanced weights from expected draws, sqrt oversampling). Only difference: augmented copies of train lines added (S12 x16, S11 x8, S10 x5, S2/S6/S8 x2). Val and test untouched. 5 seeds (42-46), mean +- std.

Augmentation (`src/data/augment.py`): rotation Normal(0, 10/3) clipped to +-10 deg, scale 0.92-1.08, brightness/contrast, blur, noise, JPEG. No shear, elastic, perspective or flip (slant and stroke shape are writer cues). Rotation was limited to 10 deg after the 30 deg preview showed ink shrinking in a large empty canvas.

## Result (test, mean +- std over 5 seeds)

| metric | without | with augmentation |
|---|---|---|
| balanced accuracy | 0.905 +- 0.003 | 0.900 +- 0.001 |
| macro-F1 | 0.882 +- 0.003 | 0.892 +- 0.003 |
| accuracy | 0.956 +- 0.001 | 0.958 +- 0.001 |

Per-scribe recall: S1, S7, S8 up by 0.3-0.8 points; S2 -1.0, S10 -1.9 (one line), S11 -2.9 (one line), S12 unchanged (6 of 7 lines). Figure: `results/compare.png`, numbers: `results/compare.json`.

## Reading

No clear gain. The differences are about 1 point or less and the per-scribe changes of S10/S11/S12 are single test lines. Macro-F1 rises by 1 point while balanced accuracy drops 0.5 points, so the copies mostly change precision (fewer lines wrongly assigned to rare scribes), not recall of the rare scribes. The rare scribes are already handled by weights + oversampling; copies of the same lines add little new information to a frozen encoder with a linear head.

Caveats: S10/S11/S12 test sets are tiny (32/14/7 lines), S11/S12 lines share pages across splits.
Possible follow-up (not done): 30 deg ablation, stronger photometric only.

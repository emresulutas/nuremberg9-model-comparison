# Balanced Probe · CHURRO-3B whole-line classifier

Every reported writer-identification score uses a linear classifier trained on whole-line features from block 16 of the frozen CHURRO-3B visual encoder. The book-based shortcut score comes from assigning the book's main writer. See the [results and figures](results/weighted_oversample/README.md).

Made 2026-10-30 / 2026-11-02. Code: `train_probe.py`, run with `nuremberg-9/scripts/run_probe9.sh`
(features first: `run_extract_new_scribes.sh`). Results: `results/weighted_oversample/`.

## Setup
| | |
|---|---|
| Encoder | frozen CHURRO-3B vision tower, features of block 16 (winner of the 7-scribe study) |
| Head | one linear layer, AdamW lr 1e-3, weight decay 1e-4, batch 512, seed 42 |
| Scribes | 1, 2, 6, 7, 8, 9, 10, 11, 12 (class numbers 0..8; S11 = 7, S12 = 8) |
| Split | `nuremberg-9/docs/split_v2_9scribes.json`: the 7 old scribes keep the page split of split_v1, S11 and S12 are split line by line 70/15/15 |
| Features | old 7 scribes reused from `features/churro_v1`, S11 and S12 extracted alone (`features/churro_v2_new`, 137 lines) |
| Selection | validation macro-F1, plateau scheduler and early stopping (patience 10), best epoch kept; test only reported |

## Choices and why
1. **Why the split changed.** S11 has 92 lines on 5 pages and S12 only 45 lines on 3 pages: a page split leaves S12 without a test
   page (candidates `split_v2a_fresh.json` and `split_v2b_fixed.json` stay as record). Line split gives every scribe val and test lines.
   Price: lines of one page sit in train and test, so S11 and S12 results are optimistic. The 7 old scribes stay comparable with split_v1.
2. **Loss weighting (what the supervisor confirmed: "frequency adjustment" = more weight on errors of rare scribes during training).**
   Plain inverse frequency gives S12 about 400 x the weight of S1 and lets the few lines be memorised. We use the class-balanced loss of
   Cui et al. 2019: weight ~ 1 / E(n), E(n) = (1 - beta^n) / (1 - beta), beta = 0.999. This alone is a ratio of about 33.
3. **Oversampling, train only.** Per epoch the lines are drawn with replacement; the share of a scribe is proportional to sqrt(n)
   (not fully balanced, to limit repeats of the 31 S12 lines). Validation and test are untouched.
4. **Weights are computed from the counts after oversampling** (expected draws per epoch), so a rare scribe is not corrected twice.
   Train lines per class: 12600, 1400, 1062, 6591, 1273, 12140, 96, 64, 31. Expected draws: 9147, 3049, 2656, 6616, 2907, 8978, 798, 652, 454.
   Class weights: 0.93, 0.98, 1.00, 0.93, 0.98, 0.93, 1.69, 1.94, 2.55 (no explosion). A line of S12 still counts about 55 x a line of S1
   (drawn about 15 times per epoch, weight 2.7 x): memorisation risk, to be reduced by augmentation (plan step 4).
5. **Not used.** Logit adjustment (Menon et al. 2021): the supervisor meant loss weighting. Tversky loss: made for segmentation, a batch holds
   0-2 lines of S12 so its overlap counts would be very noisy; not tried. No comparison grid of loss variants on purpose (decision of 2026-10-30).

## Result (test, 7636 lines)
Accuracy 0.958, balanced accuracy 0.910, macro-F1 0.887 (validation: balanced 0.884, macro-F1 0.898, best epoch 35).
Band-only shortcut baseline: balanced 0.214 on test, 0.235 on validation.

| scribe | S1 | S2 | S6 | S7 | S8 | S9 | S10 | S11 | S12 |
|---|---|---|---|---|---|---|---|---|---|
| test lines | 2711 | 306 | 235 | 1440 | 289 | 2602 | 32 | 14 | 7 |
| recall | .949 | .905 | .864 | .966 | .844 | .992 | .812 | 1.000 | .857 |

Figures: `blk16_recall.png` (red bars = fewer than 30 test lines), `blk16_confusion.png`, `blk16_curves.png`.
Main errors: S8 -> S1 (29 lines), S6 <-> S2, S1 -> S8 (43 lines): the big scribes still attract the small ones a little.

## Caveats for the report
- S11 (14 lines) and S12 (7 lines) test recalls are very noisy: one line moves S12 by 14 points.
- S11 and S12 share pages across train/val/test (line split): optimistic. S10 (32 test lines, 3 pages) is also thin.
- The validation macro-F1 used for early stopping is itself noisy for S11 and S12 (14 and 7 validation lines).
- Seed 42 only, one run: differences of a point or two between runs are not interpretable.

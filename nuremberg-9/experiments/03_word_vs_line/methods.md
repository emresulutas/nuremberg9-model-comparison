# Word vs Line · CHURRO-3B

All three rows use block 16 of the frozen CHURRO-3B visual encoder. The word modes share one linear classifier trained on word crops. The whole-line mode reuses the linear classifier from [Balanced Probe](../01_balanced_probe/methods.md).

Word crops come from `src/data/nuremberg_words.py` (quality ≤ 0.30) with a 3 px grayscale margin. The splitter processed 32,632 of 35,257 training lines, 7,083 of 7,622 validation lines and 7,094 of 7,636 test lines. These yielded 272,595 training, 59,331 validation and 59,740 test words. The word head uses seed 42, effective-number weights from expected draws, square-root oversampling, AdamW, a plateau scheduler and early stopping on validation Macro-F1. Both line-level modes use the same 7,094 test lines.

## Result (test)

| CHURRO-3B configuration | accuracy | balanced acc | macro-F1 |
|---|---|---|---|
| single words (word head) | 0.898 | 0.807 | 0.757 |
| lines, words vote (mean log-prob) | 0.966 | 0.918 | 0.907 |
| lines, line head (step 3) | 0.961 | 0.927 | 0.900 |

[Comparison figure](results/words/churro-3b-word-vs-line.png) · [Metrics](results/words/metrics.json)

## Reading
- A single word is much harder than a line (balanced 0.81 vs 0.93), as expected from step 1.
- Voting over the words of a line brings the word head to the level of the line head: accuracy and macro-F1 slightly higher, balanced accuracy slightly lower. Differences are about 1 point, one seed, so no clear winner.
- Unit does not matter much once the whole line is used; the line head needs one pass per line, the word head one pass per word.

## Caveats
- Class weights came out about 1.0 for every scribe: with word counts in the thousands the effective number saturates at beta 0.999, so only oversampling was active (in step 3 the weights ranged 0.93-2.55).
- About 7 % of the lines could not be cut and are missing from the comparison (both sides use the same lines).
- S10/S11/S12: 32/14/6 test lines, very noisy. One seed only.

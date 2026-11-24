# Split v2 (9 scribes): fresh vs fixed

Made by `src/data/nuremberg_split.py`. Lines (pages) per scribe and split. `thin` = val/test cell below 30 lines, `EMPTY` = scribe missing from a split.

## fresh

| scribe | train | val | test |
|---|---|---|---|
| S1 | 12589 (568) | 2729 (109) | 2702 (111) |
| S2 | 1381 (86) | 325 (15) | 302 (15) |
| S6 | 1055 (77) | 244 (18) | 234 (17) |
| S7 | 6589 (294) | 1442 (64) | 1422 (65) |
| S8 | 1272 (66) | 290 (17) | 286 (14) |
| S9 | 12126 (499) | 2609 (110) | 2617 (108) |
| S10 | 115 (8) | 25 (2) thin | 24 (2) thin |
| S11 | 54 (2) | 19 (2) thin | 19 (1) thin |
| S12 | 6 (1) | 25 (1) thin | 14 (1) thin |

- band-only baseline, test: accuracy 0.606, balanced 0.239 (val balanced 0.222)
- pages of split_v1 that changed split: 745

## fixed

| scribe | train | val | test |
|---|---|---|---|
| S1 | 12600 (558) | 2709 (114) | 2711 (116) |
| S2 | 1400 (81) | 302 (16) | 306 (19) |
| S6 | 1062 (78) | 236 (19) | 235 (15) |
| S7 | 6591 (294) | 1422 (66) | 1440 (63) |
| S8 | 1273 (67) | 286 (15) | 289 (15) |
| S9 | 12140 (503) | 2610 (106) | 2602 (108) |
| S10 | 96 (7) | 36 (2) | 32 (3) |
| S11 | 36 (2) | 30 (1) | 26 (2) thin |
| S12 | 20 (2) | 25 (1) thin | 0 (0) EMPTY |

- band-only baseline, test: accuracy 0.545, balanced 0.240 (val balanced 0.235)
- pages of split_v1 that changed split: 0

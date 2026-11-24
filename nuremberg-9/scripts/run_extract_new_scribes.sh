#!/usr/bin/env bash
# Plan step 3, part A: encoder features of the new scribes S11 and S12 only (about 137 lines, under a minute).
# The 7 old scribes keep their cached features in features/churro_v1 (same pages, same lines, same class numbers 0..6;
# S11 -> 7, S12 -> 8). Run on BoXX after `git pull`, one GPU, no tmux needed:
#   bash nuremberg-9/scripts/run_extract_new_scribes.sh          # GPU 2 by default
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU="${GPU:-2}"
SPLIT=nuremberg-9/docs/split_v2_9scribes.json
OUT=features/churro_v2_new
mkdir -p logs
CUDA_VISIBLE_DEVICES="$GPU" python nuremberg-7/experiments/01_churro_probe/extract_features.py \
    --split-json "$SPLIT" --out "$OUT" --only-scribes 11 12 2>&1 | tee logs/extract_new_scribes.log
python - <<'PYEOF'
import numpy as np
for sp in ("train", "val", "test"):
    d = np.load(f"features/churro_v2_new/{sp}.npz")
    w = d["writer"]
    print(sp, "S11:", int((w == 11).sum()), "S12:", int((w == 12).sum()), "labels:", sorted(set(d["label"].tolist())))
print("expected: train 64/31, val 14/7, test 14/7, labels [7, 8]")
PYEOF
echo "DONE: new-scribe features in $OUT"

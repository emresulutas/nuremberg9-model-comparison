#!/usr/bin/env bash
# Plan step 3, part B: train the 9-scribe head (weighted loss + oversampling, train only) on the cached features.
# Needs features/churro_v1 (old scribes) and features/churro_v2_new (run_extract_new_scribes.sh first).
# A linear head on cached features: seconds to minutes on CPU, no tmux needed. Run on BoXX after `git pull`:
#   bash nuremberg-9/scripts/run_probe9.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RES=nuremberg-9/experiments/01_balanced_probe/results/weighted_oversample
mkdir -p logs
python nuremberg-9/experiments/01_balanced_probe/train_probe.py --out "$RES" --device cpu 2>&1 | tee logs/probe9.log
cp logs/probe9.log "$RES/log.txt"
git add "$RES"
git diff --cached --quiet || git commit -m "Add 9-scribe head with weighted loss and oversampling"
git pull --rebase
git push
echo "DONE: 9-scribe probe results pushed"

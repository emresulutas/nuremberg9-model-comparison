#!/usr/bin/env bash
# Plan step 4, parts 3 and 4 in one go: augmented copies -> features -> train without and with them on 5 seeds -> compare -> push.
# Run in the tmux session `churro`:   bash nuremberg-9/scripts/run_aug_all.sh     # GPU 2 by default, about 30 min
# Same settings as step 3 (weights + sqrt oversampling); the only difference is the augmented copies added to train.
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU="${GPU:-2}"
RES=nuremberg-9/experiments/02_augmentation/results
TRAIN=nuremberg-9/experiments/01_balanced_probe/train_probe.py
mkdir -p logs "$RES/seeds"
CUDA_VISIBLE_DEVICES="$GPU" python nuremberg-9/experiments/02_augmentation/extract_aug.py 2>&1 | tee logs/extract_aug.log
for SEED in 42 43 44 45 46; do
    python "$TRAIN" --seed "$SEED" --out "$RES/seeds/base_s$SEED" --device cpu > "logs/probe9_base_s$SEED.log" 2>&1
    python "$TRAIN" --seed "$SEED" --aug-features features/churro_v2_aug --out "$RES/seeds/aug_s$SEED" --device cpu > "logs/probe9_aug_s$SEED.log" 2>&1
    echo "seed $SEED done"
done
python nuremberg-9/experiments/02_augmentation/compare.py 2>&1 | tee "$RES/compare_log.txt"
git add "$RES"
git diff --cached --quiet || git commit -m "Add augmentation comparison: 9-scribe head without and with augmented copies, 5 seeds"
git pull --rebase
git push
echo "DONE: augmentation comparison pushed"

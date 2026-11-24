#!/usr/bin/env bash
# Plan step 4, part 2: preview pictures of the augmented copies (CPU, about a minute, needs the images on BoXX).
# Run after `git pull`:   bash nuremberg-9/scripts/run_aug_preview.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
python -c "import albumentations" 2>/dev/null || { echo 'albumentations is missing: pip install -r requirements.txt'; exit 1; }
OUT=nuremberg-9/experiments/02_augmentation/results/preview
mkdir -p logs
python nuremberg-9/experiments/02_augmentation/preview.py --out "$OUT" 2>&1 | tee logs/aug_preview.log
git add "$OUT"
git diff --cached --quiet || git commit -m "Add augmentation preview pictures for the rare scribes"
git pull --rebase
git push
echo "DONE: augmentation preview pushed"

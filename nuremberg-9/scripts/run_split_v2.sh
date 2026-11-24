#!/usr/bin/env bash
# Plan step 2: 9-scribe split. The 7 old scribes keep the page split of split_v1; S11 and S12 are split line by line.
# CPU only, about a minute, no training. Run on BoXX after `git pull`:   bash nuremberg-9/scripts/run_split_v2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=nuremberg-9/docs
mkdir -p logs "$OUT"
python -m src.data.nuremberg_split --mode lines --scribes 1 2 6 7 8 9 10 11 12 --line-scribes 11 12 \
    --base-split nuremberg-7/docs/split_v1.json --out-dir "$OUT" 2>&1 | tee logs/split_v2_lines.log
cp logs/split_v2_lines.log "$OUT/split_v2_9scribes_log.txt"
git add "$OUT"
git diff --cached --quiet || git commit -m "Add 9-scribe split: S11 and S12 split line by line"
git pull --rebase
git push
echo "DONE: 9-scribe split pushed"

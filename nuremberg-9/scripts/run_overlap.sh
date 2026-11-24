#!/usr/bin/env bash
# Next goal in one go: two more word-head seeds -> error overlap of word-vote and line models -> push.
# Run in the tmux session `churro`:   bash nuremberg-9/scripts/run_overlap.sh      # GPU 2 by default
# Nothing is re-extracted: the word features and the line predictions of the earlier steps are reused.
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU="${GPU:-2}"
RES=nuremberg-9/experiments/03_word_vs_line/results
mkdir -p logs
for SEED in 43 44; do
    echo "word head, seed $SEED"
    CUDA_VISIBLE_DEVICES="$GPU" python nuremberg-9/experiments/03_word_vs_line/train_words.py --seed "$SEED" \
        --out "$RES/words_s$SEED" 2>&1 | tee "logs/words_s$SEED.log" | tail -4
done
python nuremberg-9/experiments/04_error_overlap/analyze.py 2>&1 | tee nuremberg-9/experiments/04_error_overlap/results_log.txt
git add nuremberg-9/experiments/04_error_overlap "$RES/words_s43" "$RES/words_s44"
git diff --cached --quiet || git commit -m "Add error overlap analysis of the word-vote and line models"
git pull --rebase
git push
echo "DONE: error overlap analysis pushed"

#!/usr/bin/env bash
# Plan step 5 in one go: smoke test -> word features on 2 GPUs -> word head vs line head -> push.
# Run in the tmux session `churro`:   bash nuremberg-9/scripts/run_words_all.sh     # GPUs 2 and 3 by default
# The line-based side is the step-3 head (nothing to rerun); the word-based side is trained here with the same recipe.
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
EXT=nuremberg-9/experiments/03_word_vs_line/extract_words.py
RES=nuremberg-9/experiments/03_word_vs_line/results
mkdir -p logs "$RES"
echo "smoke test (20 lines of S12)"
CUDA_VISIBLE_DEVICES="$GPU_A" python "$EXT" --scribes 12 --splits train --limit 20 --out features/_words_smoke 2>&1 | tail -3
rm -rf features/_words_smoke
CUDA_VISIBLE_DEVICES="$GPU_A" python "$EXT" --scribes 1 2 6 8 > logs/words_extract_a.log 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES="$GPU_B" python "$EXT" --scribes 9 7 10 11 12 > logs/words_extract_b.log 2>&1 &
PID_B=$!
wait "$PID_A"
wait "$PID_B"
python nuremberg-9/experiments/03_word_vs_line/train_words.py --device cpu 2>&1 | tee "$RES/train_log.txt"
git add "$RES"
git diff --cached --quiet || git commit -m "Add word-based head and comparison with the line-based head (step 5)"
git pull --rebase
git push
echo "DONE: word vs line comparison pushed"

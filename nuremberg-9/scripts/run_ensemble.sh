#!/usr/bin/env bash
# Error overlap (refreshed figures) + ensemble with weight search on 15 seed pairs -> push.
# Run in the tmux session `churro`:   bash nuremberg-9/scripts/run_ensemble.sh
# No training: it reuses the saved heads and predictions (line seeds 42-46, word seeds 42-44).
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP=nuremberg-9/experiments/04_error_overlap
python "$EXP/analyze.py" 2>&1 | tee "$EXP/results_log.txt"
python "$EXP/ensemble.py" 2>&1 | tee "$EXP/ensemble_log.txt"
git add "$EXP"
git diff --cached --quiet || git commit -m "Ensemble of word-vote and line models: weight search over 15 seed pairs, controls, figures"
git pull --rebase
git push
echo "DONE: ensemble analysis pushed"

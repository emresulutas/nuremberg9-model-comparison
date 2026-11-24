#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

experiment=nuremberg-9/experiments/06_resnet18_vs_churro
baseline=nuremberg-9/experiments/05_model_comparison
churro_repeats=nuremberg-9/experiments/02_augmentation/results/seeds
results="$experiment/results"
session=resnet18-vs-churro

if (( $# == 0 )); then
    runner="$PWD/$experiment/run_repeats.sh"
    tmux new-session -d -s "$session" -n gpu1 bash "$runner" 1 43 45
    tmux new-window -t "$session:" -n gpu2 bash "$runner" 2 44 46
    if [[ -n ${TMUX:-} ]]; then
        tmux switch-client -t "$session"
    else
        tmux attach-session -t "$session"
    fi
    exit
fi

gpu=$1
shift
python_bin=${PYTHON_BIN:-/HDD2/esulutas/envs/churro-comparison/bin/python}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
mkdir -p "$results/logs"
for seed in "$@"; do
    out="$results/resnet18/seed_$seed"
    if [[ ! -s $out/predictions.npz ]]; then
        "$python_bin" "$baseline/train_baseline.py" --model resnet18 --device cuda \
            --cache-dir cache/nuremberg-9-lines --seed "$seed" --out "$out" \
            2>&1 | tee "$results/logs/resnet18_seed_$seed.log"
    fi
done

(
    flock 9
    for seed in 43 44 45 46; do
        [[ -s $results/resnet18/seed_$seed/predictions.npz ]] || exit 0
    done

    models=(--model "resnet18_s42=$baseline/results/whole_line/resnet18/fit/predictions.npz")
    for seed in 43 44 45 46; do
        models+=(--model "resnet18_s$seed=$results/resnet18/seed_$seed/predictions.npz")
        models+=(--model "churro_s$seed=$churro_repeats/base_s$seed/blk16_predictions.npz")
    done
    "$python_bin" "$baseline/compare.py" "${models[@]}" --out "$results/scores.json" >/dev/null
    printf 'Comparison saved to %s/scores.json\n' "$results"
) 9>"$results/.comparison.lock"

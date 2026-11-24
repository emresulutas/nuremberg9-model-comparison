#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

exp=nuremberg-9/experiments/07_model_details
saved=nuremberg-9/experiments/05_model_comparison
repeat=nuremberg-9/experiments/06_resnet18_vs_churro/results/resnet18
results="$exp/results"
cache=cache/nuremberg-9-model-details
line_cache=cache/nuremberg-9-lines
session=nuremberg-model-details
python_bin="${PYTHON_BIN:-/HDD2/esulutas/envs/churro-comparison/bin/python}"

if (( $# == 0 )) || [[ $1 == --foreground ]]; then
    test -x "$python_bin"
    for name in resnet18 resnet50 swin_tiny; do
        test -s "$saved/results/whole_line/$name/fit/predictions.npz"
    done
    runner="$PWD/$exp/run.sh"
    if [[ ${1:-} == --foreground ]]; then
        if tmux has-session -t "=$session" 2>/dev/null; then
            printf 'The existing job is already queued in tmux (%s).\n' "$session" >&2
            exit 1
        fi
        mkdir -p "$results"
        exec 9>"$results/.foreground.lock"
        if ! flock -n 9; then
            printf 'The foreground job is already running.\n' >&2
            exit 1
        fi
        set -m
        bash "$runner" 1 resnet18 resnet50 &
        first=$!
        bash "$runner" 2 swin_tiny &
        second=$!
        trap 'kill -TERM -- "-$first" "-$second" 2>/dev/null || true; wait || true; exit 130' INT TERM
        status=0
        wait "$first" || status=$?
        wait "$second" || status=$?
        exit "$status"
    fi
    if ! tmux has-session -t "=$session" 2>/dev/null; then
        mkdir -p "$results"
        exec 9>"$results/.foreground.lock"
        if ! flock -n 9; then
            printf 'The foreground job is already running.\n' >&2
            exit 1
        fi
        tmux new-session -d -s "$session" -n gpu1 bash "$runner" 1 resnet18 resnet50
        tmux new-window -t "$session:" -n gpu2 bash "$runner" 2 swin_tiny
    fi
    if [[ -n ${TMUX:-} ]]; then
        tmux switch-client -t "$session"
    else
        tmux attach-session -t "$session"
    fi
    exit
fi

gpu=$1
shift
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
mkdir -p "$results/logs" "$cache"
exec > >(tee -a "$results/logs/gpu$gpu.log") 2>&1

echo "GPU $gpu lane: $*"
flock "$cache/.prepare.lock" "$python_bin" "$exp/prepare.py" --line-cache "$line_cache" --work-cache "$cache"
while [[ " $* " == *" resnet18 "* ]] && tmux has-session -t '=resnet18-vs-churro' 2>/dev/null; do
    echo "Waiting for the earlier ResNet18 repeat session to finish."
    sleep 60
done
while :; do
    read -r memory < <(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( memory < 256 )); then
        break
    fi
    echo "Waiting for GPU $gpu ($memory MiB in use)."
    sleep 60
done

train() {
    local model=$1 seed=$2 data=$3 out=$4 view=$5
    if [[ -s $out/predictions.npz && -s $out/metrics.json ]]; then
        echo "Reusing $out"
        return
    fi
    echo "Training $model · $view · seed $seed · $out"
    "$python_bin" "$saved/train_baseline.py" --model "$model" --seed "$seed" \
        --cache-dir "$data" --out "$out" --input-view "$view" --device cuda
}

for model in "$@"; do (
    model_out="$results/$model"
    mkdir -p "$model_out"
    exec 8>"$model_out/.run.lock"
    if ! flock -n 8; then
        echo "$model is already running elsewhere."
        exit 0
    fi
    for seed in 42 43 44; do
        train "$model" "$seed" "$cache/words" "$model_out/word_seed_$seed" word
    done
    for seed in 42 43 44 45 46; do
        if (( seed != 42 )); then
            if [[ $model == resnet18 && -s $repeat/seed_$seed/predictions.npz && ! -e $model_out/line_seed_$seed ]]; then
                ln -s "$PWD/$repeat/seed_$seed" "$model_out/line_seed_$seed"
            fi
            train "$model" "$seed" "$line_cache" "$model_out/line_seed_$seed" whole_line
        fi
        train "$model" "$seed" "$cache/augmented-lines" "$model_out/aug_seed_$seed" whole_line
    done
    "$python_bin" "$exp/analyze.py" --model "$model" \
        --baseline "$saved/results/whole_line/$model/fit" --out "$model_out"
    flock "$results/.report.lock" "$python_bin" nuremberg-9/scripts/build_report9.py
    echo "Completed $model"
); done

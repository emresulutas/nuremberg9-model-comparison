#!/usr/bin/env bash
# From the active Python environment: bash nuremberg-9/scripts/run_model_comparison.sh
# Add --prepare-layout to create result directories and link the existing CHURRO fit only.
# NUREMBERG_ROOT can point to an absolute dataset path. The default is datasets/nuremberg.
# Defaults to GPU 1 for ResNet50 then ResNet18, and GPU 2 for Swin-Tiny.
# CUDA_VISIBLE_DEVICES can select one or two GPUs. DEVICE=cpu runs serially.
# PYTHON_BIN defaults to python.
# TRAIN_BATCH_SIZE, TRAIN_MAX_BATCH_PIXELS and TRAIN_WORKERS override trainer defaults.
# RESULTS_DIR defaults to the experiment's results/whole_line. Use a new directory for changed training settings.
# Stage logs are appended under logs/model_comparison. Run the same command again after fixing a failed stage.
set -euo pipefail
cd "$(dirname "$0")/../.."

exp=nuremberg-9/experiments/05_model_comparison
results=${RESULTS_DIR:-"$exp/results/whole_line"}
comparison="$results/comparison"
metrics="$comparison/metrics"
gallery="$comparison/explanations"
selection="$comparison/selected_lines.csv"
churro_reference=nuremberg-9/experiments/01_balanced_probe/results/weighted_oversample
cache_dir=cache/nuremberg-9-lines
logs=logs/model_comparison
data_root=${NUREMBERG_ROOT:-datasets/nuremberg}
device=${DEVICE:-cuda}
models=(resnet18 resnet50 swin_tiny)
explanation_models=(churro "${models[@]}")
fit_files=(best.pt predictions.npz metrics.json)
churro_files=(blk16_head.npz blk16_predictions.npz blk16_metrics.json summary.json
              blk16_confusion.png blk16_curves.png blk16_recall.png log.txt)
python_bin=${PYTHON_BIN:-python}
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

layout_only=0
if (( $# )); then
    [[ $# == 1 && $1 == --prepare-layout ]] || die 'Usage: run_model_comparison.sh [--prepare-layout]'
    layout_only=1
fi

mkdir -p -- "$logs"
if command -v flock >/dev/null; then
    exec 9>"$logs/.run.lock"
    flock -n 9 || die 'This comparison is already running in this checkout.'
fi

run_stage() {
    local stage=$1
    shift
    printf '\n[%s] %s\n' "$stage" "$(date -Is)" | tee -a "$logs/$stage.log"
    if ! "$@" 2>&1 | tee -a "$logs/$stage.log"; then
        die "$stage failed. See $logs/$stage.log."
    fi
}

complete_directory() {
    local directory=$1 file
    shift
    if [[ ! -e $directory && ! -L $directory ]]; then
        return 1
    fi
    [[ -d $directory ]] || die "Expected an output directory at $directory."
    for file in "$@"; do
        [[ -f $directory/$file && -s $directory/$file ]] || die "Incomplete output at $directory. Missing or empty: $file."
    done
    printf 'Reusing %s\n' "$directory"
}

mkdir -p -- "$comparison"
for model in "${explanation_models[@]}"; do
    mkdir -p -- "$results/$model"
done
churro_fit="$results/churro/fit"
if ! complete_directory "$churro_fit" "${churro_files[@]}"; then
    for file in "${churro_files[@]}"; do
        [[ -s $churro_reference/$file ]] || die "Missing canonical CHURRO artifact: $churro_reference/$file."
    done
    churro_staging=$(mktemp -d "$results/churro/.fit.XXXXXX")
    for file in "${churro_files[@]}"; do
        ln -sr -- "$churro_reference/$file" "$churro_staging/$file"
    done
    mv -T -- "$churro_staging" "$churro_fit"
fi
if (( layout_only )); then
    printf 'Prepared the model comparison directories in %s\n' "$results"
    exit 0
fi

if [[ -n ${NUREMBERG_ROOT:-} && $data_root != /* ]]; then
    die 'NUREMBERG_ROOT must be an absolute path.'
fi
command -v "$python_bin" >/dev/null || die "Cannot find $python_bin. Activate the experiment environment first."

gpu_ids=()
case $device in
    cuda|cuda:0)
        gpu_list=${CUDA_VISIBLE_DEVICES-1,2}
        if [[ -z $gpu_list || $gpu_list == ,* || $gpu_list == *, || $gpu_list == *,,* ]]; then
            die 'Set CUDA_VISIBLE_DEVICES to one or two CUDA GPU IDs, e.g. 1,2.'
        fi
        IFS=, read -r -a gpu_ids <<< "$gpu_list"
        if (( ${#gpu_ids[@]} > 2 )) || [[ ${gpu_ids[0]} == "${gpu_ids[1]:-}" ]]; then
            die 'Set CUDA_VISIBLE_DEVICES to one GPU ID or two distinct GPU IDs.'
        fi
        ;;
    cpu) ;;
    *) die 'Set DEVICE to cuda or cpu, and select physical GPUs with CUDA_VISIBLE_DEVICES.' ;;
esac

explanations_pending=0
if ! complete_directory "$gallery" catalog.json line_explanations.png; then
    explanations_pending=1
fi
for model in "${explanation_models[@]}"; do
    if ! complete_directory "$results/$model/explanations" catalog.json; then
        explanations_pending=1
    fi
done
if (( explanations_pending )) ||
    [[ ! -s $results/resnet50/fit/best.pt || ! -s $results/resnet18/fit/best.pt ||
       ! -s $results/swin_tiny/fit/best.pt ]]; then
    run_stage preflight env CUDA_VISIBLE_DEVICES="${gpu_list:-}" "$python_bin" -c '
import sys
import timm
import torch
from packaging.version import Version

if not Version("1.0.31") <= Version(timm.__version__) < Version("2"):
    raise SystemExit(f"Expected timm>=1.0.31,<2, found {timm.__version__}")
requested = int(sys.argv[1])
if requested and torch.cuda.device_count() != requested:
    raise SystemExit(f"Expected {requested} CUDA GPUs, found {torch.cuda.device_count()}")
print(f"PyTorch {torch.__version__}, timm {timm.__version__}, visible CUDA GPUs: {requested}")
' "${#gpu_ids[@]}"
fi

if ! complete_directory "$cache_dir" metadata.json train.csv val.csv test.csv; then
    run_stage prepare "$python_bin" "$exp/data.py" --data-root "$data_root" --cache-dir "$cache_dir"
fi

training_args=()
if [[ -n ${TRAIN_BATCH_SIZE:-} ]]; then
    training_args+=(--batch-size "$TRAIN_BATCH_SIZE")
fi
if [[ -n ${TRAIN_MAX_BATCH_PIXELS:-} ]]; then
    training_args+=(--max-batch-pixels "$TRAIN_MAX_BATCH_PIXELS")
fi
if [[ -n ${TRAIN_WORKERS:-} ]]; then
    training_args+=(--workers "$TRAIN_WORKERS")
fi

train_lane() {
    local gpu=$1 model model_dir
    shift
    set +m
    for model in "$@"; do
        model_dir="$results/$model/fit"
        if ! complete_directory "$model_dir" "${fit_files[@]}"; then
            printf 'Training %s on %s\n' "$model" "${gpu:+GPU }${gpu:-CPU}"
            run_stage "$model" env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$exp/train_baseline.py" \
                --model "$model" --cache-dir "$cache_dir" --out "$model_dir" --device "$device" \
                "${training_args[@]}"
        fi
    done
}

training_pids=()
cancel_training() {
    local status=$1 pid
    trap - INT TERM
    for pid in "${training_pids[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    wait "${training_pids[@]}" 2>/dev/null || true
    exit "$status"
}

if (( ${#gpu_ids[@]} == 2 )); then
    trap 'cancel_training 130' INT
    trap 'cancel_training 143' TERM
    # Give each lane its own process group so a cancellation also stops Python and its workers.
    set -m
    train_lane "${gpu_ids[0]}" resnet50 resnet18 &
    training_pids+=("$!")
    train_lane "${gpu_ids[1]}" swin_tiny &
    training_pids+=("$!")
    set +m
    training_failed=0
    for pid in "${training_pids[@]}"; do
        if ! wait "$pid"; then
            training_failed=1
        fi
    done
    trap - INT TERM
    (( training_failed == 0 )) || die "Training failed. See the stage logs in $logs. Completed models will be reused."
else
    train_lane "${gpu_ids[0]:-}" resnet50 swin_tiny resnet18
fi

compare_args=()
for model in "${models[@]}"; do
    compare_args+=(--model "$model=$results/$model/fit/predictions.npz")
done

if ! complete_directory "$metrics" scores.json test_scores.png test_scores.svg \
        test_scores_unseen_pages.png test_scores_unseen_pages.svg recall_per_scribe.png recall_per_scribe.svg \
        confusion.png confusion.svg; then
    comparison_staging=$(mktemp -d "$comparison/.metrics.XXXXXX")
    run_stage compare env CUDA_VISIBLE_DEVICES= "$python_bin" "$exp/compare.py" "${compare_args[@]}" \
        --figures "$comparison_staging" --out "$comparison_staging/scores.json"
    mv -T -- "$comparison_staging" "$metrics"
fi
printf 'Comparison scores: %s/scores.json\n' "$metrics"

if [[ -e $selection || -L $selection ]]; then
    [[ -f $selection && -s $selection ]] || die "Invalid line selection at $selection."
else
    selection_staging=$(mktemp -d "$comparison/.selection.XXXXXX")
    run_stage select "$python_bin" "$exp/select_lines.py" --out "$selection_staging/selected_lines.csv"
    mv -T -- "$selection_staging/selected_lines.csv" "$selection"
    rmdir -- "$selection_staging"
fi

if (( explanations_pending )); then
    run_stage explain env CUDA_VISIBLE_DEVICES="${gpu_ids[0]:-}" "$python_bin" "$exp/explain.py" --cache-dir "$cache_dir" \
        --results-dir "$results" --model-results-dir "$results" --selection "$selection" --out "$gallery" --device "$device"
    for model in "${explanation_models[@]}"; do
        complete_directory "$results/$model/explanations" catalog.json ||
            die "Missing explanation artifacts for $model."
    done
    complete_directory "$gallery" catalog.json line_explanations.png || die 'Missing comparison explanations.'
fi
printf 'Explanation gallery: %s\n' "$gallery"

#!/usr/bin/env bash
# Supervisor requests 6-7 (06-11-2026) on the 14 clean species, one job queue.
#
#   bash run_clean14.sh [cv] [degraded] [final]      (default: all three)
#
#   cv        request 6: 5 models x 5 folds          -> Task-B-clean14-cv/
#   degraded  request 7: ConvNeXt x 5 folds, grayscale and 16px
#                                                     -> Task-B-clean14-cv-{grayscale,px16}/
#   final     request 6: Final-B 5 models x 3 seeds  -> Task-B-clean14-final/
#
# Every run: backbone and head LR 1e-4, 5 epochs warmup + cosine, max 50 epochs,
# best.pth = lowest val loss after warmup, patience 15, 224 cache.
# A CV job is one model (its folds run in order: train_kfold.py locks the model
# folder). Jobs are started longest first, at most MAX_PARALLEL at a time.
# Re-running the script is safe: finished folds/runs are kept, unfinished ones
# restart from epoch 1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PYTHON="${PYTHON:-python}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/task-B-group-split-224}"
EXP_ROOT="${EXP_ROOT:-$ROOT/data/experiments}"
CHECKSUMS="$ROOT/docs/timeline/01-11-2026-task-B-group-split-224.sha256"
read -r -a MODELS <<< "${MODELS:-efficientnet_b3 resnet50 mobilenetv3_large swin_tiny convnext_tiny}"
read -r -a SEEDS <<< "${SEEDS:-42 123 456}"
PHASES=("$@")
(( ${#PHASES[@]} )) || PHASES=(cv degraded final)
for phase in "${PHASES[@]}"; do
    [[ "$phase" =~ ^(cv|degraded|final)$ ]] || { echo "Unknown phase: $phase" >&2; exit 2; }
done
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_PARALLEL must be a positive integer' >&2; exit 2; }
export PYTHONUNBUFFERED=1

# The data check is cheap (400 MB); CV does not check by itself.
if [[ "${SKIP_DATA_CHECK:-0}" != 1 ]]; then
    [[ "$(basename "$DATA_ROOT")" == task-B-group-split-224 ]] || {
        echo "DATA_ROOT must be a folder named task-B-group-split-224 (checksum paths)" >&2; exit 2; }
    (cd "$(dirname "$DATA_ROOT")" && sha256sum -c --quiet "$CHECKSUMS") || {
        echo "Data does not match $CHECKSUMS" >&2; exit 1; }
    echo "Data OK: $DATA_ROOT"
fi

LOG_DIR="$EXP_ROOT/clean14-logs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
{
    echo "phases: ${PHASES[*]}"; echo "max_parallel: $MAX_PARALLEL"; echo "host: $(hostname)"
    git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --short --untracked-files=no
} > "$LOG_DIR/launch.txt"
"$PYTHON" -m pip freeze > "$LOG_DIR/pip_freeze.txt" 2>/dev/null || true
nvidia-smi > "$LOG_DIR/nvidia-smi.txt" 2>/dev/null || true
echo "Logs: $LOG_DIR"

RULE=(--backbone_lr 1e-4 --head_lr 1e-4 --warmup_epochs 5 --epochs 50
      --selection_metric val_loss --select_after_warmup --early_stopping_patience 15)
CV=("$PYTHON" "$ROOT/notebooks/phase 2/supervisor-feedback/task_01_fivefold_cv_training/train_kfold.py"
    --species_set clean14 --data_root "$DATA_ROOT" --num_workers "$NUM_WORKERS"
    --wandb_project woodbench-clean14 --resume "${RULE[@]}")
FINAL=("$PYTHON" "$HERE/train_final_b.py" --species_set clean14 --data_root "$DATA_ROOT"
       --num_workers "$NUM_WORKERS" --output_root "$EXP_ROOT/Task-B-clean14-final")

names=(); commands=()
add() { names+=("$1"); shift; commands+=("$(printf '%q ' "$@")"); }
for phase in "${PHASES[@]}"; do
    case "$phase" in
        cv) for m in "${MODELS[@]}"; do
                add "cv_$m" "${CV[@]}" --backbone "$m" --output_root "$EXP_ROOT/Task-B-clean14-cv"
            done ;;
        degraded) for v in grayscale px16; do
                add "cv_${v}_convnext_tiny" "${CV[@]}" --backbone convnext_tiny --input_variant "$v" \
                    --output_root "$EXP_ROOT/Task-B-clean14-cv-$v"
            done ;;
        final) for s in "${SEEDS[@]}"; do for m in "${MODELS[@]}"; do
                add "final_s${s}_$m" "${FINAL[@]}" --backbone "$m" --seed "$s"
            done; done ;;
    esac
done
printf '%s\n' "${names[@]}" > "$LOG_DIR/jobs.txt"
echo "${#names[@]} jobs, at most $MAX_PARALLEL at a time"

pids=()
stop_children() {
    trap - INT TERM
    kill "${pids[@]}" 2>/dev/null || true
    wait || true
    exit 130
}
trap stop_children INT TERM
for i in "${!names[@]}"; do
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n || true; done
    name="${names[$i]}"
    (
        status=0
        eval "${commands[$i]}" > "$LOG_DIR/$name.log" 2>&1 || status=$?
        echo "$status" > "$LOG_DIR/$name.exit"
        echo "$(date -u +%H:%M:%S) $name finished (exit $status)"
    ) &
    pids+=("$!")
    echo "$(date -u +%H:%M:%S) $name started"
done
wait || true
failed=$( (grep -lv '^0$' "$LOG_DIR"/*.exit 2>/dev/null || true) | wc -l)
echo "All jobs done; $failed failed (see *.exit and logs in $LOG_DIR)"
(( failed == 0 ))

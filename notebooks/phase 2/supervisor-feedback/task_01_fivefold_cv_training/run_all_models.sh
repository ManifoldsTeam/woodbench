#!/usr/bin/env bash
# Run five models concurrently; each model trains its folds in order.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo 'MAX_PARALLEL must be a positive integer' >&2
    exit 2
fi
read -r -a models <<< "${MODELS:-convnext_tiny mobilenetv3_large resnet50 efficientnet_b3 swin_tiny}"
mkdir -p "$LOG_ROOT"
# Serialize launches sharing a log root. Python also locks each model output.
exec 9> "$LOG_ROOT/.launcher.lock"
flock -n 9 || { echo 'Another launcher owns this LOG_ROOT' >&2; exit 1; }
SESSION_DIR="$(mktemp -d "$LOG_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
echo "Logs: $SESSION_DIR; maximum concurrent models: $MAX_PARALLEL"
pids=()
names=()
failed=0
stop_children() {
    trap - INT TERM
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}
trap stop_children INT TERM
wait_batch() {
    local index status
    for index in "${!pids[@]}"; do
        status=0
        wait "${pids[$index]}" || status=$?
        echo "$status" > "$SESSION_DIR/${names[$index]}.exit"
        rm -f "$SESSION_DIR/${names[$index]}.pid"
        if (( status != 0 )); then
            echo "${names[$index]} failed (exit $status); see its log" >&2
            failed=1
        else
            echo "${names[$index]} completed"
        fi
    done
    pids=()
    names=()
}
for model in "${models[@]}"; do
    # Omitted overrides use the CLI defaults, including each model's batch size.
    args=(--backbone "$model")
    for setting in N_FOLDS:n_folds EPOCHS:epochs BATCH_SIZE:batch_size \
                   BACKBONE_LR:backbone_lr HEAD_LR:head_lr WEIGHT_DECAY:weight_decay \
                   LABEL_SMOOTHING:label_smoothing SEED:seed NUM_WORKERS:num_workers \
                   WANDB_PROJECT:wandb_project OUTPUT_ROOT:output_root \
                   DATA_ROOT:data_root MANIFEST_PATH:manifest_path; do
        env_name="${setting%%:*}"
        flag="${setting#*:}"
        value="${!env_name:-}"
        if [[ -n "$value" ]]; then args+=("--$flag" "$value"); fi
    done
    if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
    if [[ "${USE_90ROTATION:-0}" == 1 ]]; then args+=(--use_90rotation); fi
    "$PYTHON" "$SCRIPT_DIR/train_kfold.py" "${args[@]}" > "$SESSION_DIR/$model.log" 2>&1 &
    pids+=("$!")
    names+=("$model")
    echo "$!" > "$SESSION_DIR/$model.pid"
    if (( ${#pids[@]} >= MAX_PARALLEL )); then wait_batch; fi
done
wait_batch
exit "$failed"

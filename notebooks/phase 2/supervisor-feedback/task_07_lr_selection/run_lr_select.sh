#!/usr/bin/env bash
# Step 1: every MODELS x LRS run, at most MAX_PARALLEL at a time; one log per run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/../../../../artifacts/phase_2/supervisor-feedback/task_07_lr_selection/logs}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo 'MAX_PARALLEL must be a positive integer' >&2
    exit 2
fi
read -r -a models <<< "${MODELS:-convnext_tiny swin_tiny mobilenetv3_large efficientnet_b3 resnet50}"
read -r -a lrs <<< "${LRS:-1e-5 1e-4}"
mkdir -p "$LOG_ROOT"
SESSION_DIR="$(mktemp -d "$LOG_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
echo "Logs: $SESSION_DIR; maximum concurrent runs: $MAX_PARALLEL"
pids=()
names=()
failed=0
wait_batch() {
    local index status
    for index in "${!pids[@]}"; do
        status=0
        wait "${pids[$index]}" || status=$?
        echo "$status" > "$SESSION_DIR/${names[$index]}.exit"
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
trap 'kill "${pids[@]}" 2>/dev/null || true; exit 130' INT TERM
for model in "${models[@]}"; do
    for lr in "${lrs[@]}"; do
        name="${model}_bblr${lr}"
        args=(--backbone "$model" --backbone_lr "$lr")
        for setting in SEED:seed EPOCHS:epochs NUM_WORKERS:num_workers \
                       OUTPUT_ROOT:output_root DATA_ROOT:data_root; do
            env_name="${setting%%:*}"
            flag="${setting#*:}"
            value="${!env_name:-}"
            if [[ -n "$value" ]]; then args+=("--$flag" "$value"); fi
        done
        "$PYTHON" "$SCRIPT_DIR/train_lr_select.py" "${args[@]}" > "$SESSION_DIR/$name.log" 2>&1 &
        pids+=("$!")
        names+=("$name")
        if (( ${#pids[@]} >= MAX_PARALLEL )); then wait_batch; fi
    done
done
wait_batch
exit "$failed"

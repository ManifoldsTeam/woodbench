#!/bin/bash
# Run all 5 models with 5-fold CV in parallel
# Usage: bash run_all_models.sh

set -e

# Configuration
N_FOLDS=5
EPOCHS=30
BACKBONE_LR=1e-5
HEAD_LR=1e-4
WEIGHT_DECAY=1e-4
LABEL_SMOOTHING=0.1
SEED=42
BATCH_SIZE=32

# Models to train
MODELS=("convnext_tiny" "mobilenetv3_large" "resnet50" "efficientnet_b3" "swin_tiny")

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "5-Fold CV by Origin Key - All Models"
echo "=============================================="
echo "Date: $(date)"
echo "Log dir: $LOG_DIR"
echo "Models: ${MODELS[*]}"
echo "Folds: $N_FOLDS, Epochs: $EPOCHS"
echo "=============================================="

# Function to run model
run_model() {
    local model=$1
    local log_file="$LOG_DIR/${model}.log"
    local pid_file="$LOG_DIR/${model}.pid"

    echo "Starting $model..."
    echo "Log: $log_file"

    # Run in background with nohup
    nohup python train_kfold.py \
        --backbone "$model" \
        --n_folds $N_FOLDS \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --backbone_lr $BACKBONE_LR \
        --head_lr $HEAD_LR \
        --weight_decay $WEIGHT_DECAY \
        --label_smoothing $LABEL_SMOOTHING \
        --seed $SEED \
        --wandb_project "woodbench-origin-cv" \
        > "$log_file" 2>&1 &

    local pid=$!
    echo $pid > "$pid_file"
    echo "$model started with PID $pid"
}

# Kill existing processes
kill_existing() {
    echo "Checking for existing processes..."
    for model in "${MODELS[@]}"; do
        local pid_file="$LOG_DIR/${model}.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 $pid 2>/dev/null; then
                echo "Killing existing $model (PID $pid)..."
                kill $pid 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
    # Kill any remaining python processes for this script
    pkill -f "train_kfold.py" 2>/dev/null || true
    sleep 2
}

# Start all models in parallel
kill_existing

echo ""
echo "Starting all models in parallel..."
echo ""

for model in "${MODELS[@]}"; do
    run_model "$model"
done

echo ""
echo "=============================================="
echo "All models started!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  tail -f logs/<model>.log"
echo ""
echo "Check status:"
echo "  ps aux | grep train_kfold"
echo ""
echo "Kill all:"
echo "  pkill -f train_kfold.py"
echo ""
echo "=============================================="

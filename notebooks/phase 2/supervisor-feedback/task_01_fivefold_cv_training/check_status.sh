#!/bin/bash
# Check running processes and their logs
# Usage: bash check_status.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

echo "=========================================="
echo "5-FOLD CV STATUS"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Check running processes
echo "--- Running Processes ---"
RUNNING=$(ps aux | grep -E "train_kfold|train_single_fold" | grep -v grep | wc -l)
echo "Training processes: $RUNNING"

if [ $RUNNING -gt 0 ]; then
    echo ""
    ps aux | grep -E "train_kfold|train_single_fold" | grep -v grep | awk '{print "  PID: "$2, "CMD: "$11, $12, $13, $14}'
fi

echo ""
echo "--- GPU Status ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
else
    echo "  nvidia-smi not available"
fi

echo ""
echo "--- Disk Space (data dir) ---"
du -sh "$SCRIPT_DIR/../../../../../data/experiments/Task-B-origin-cv" 2>/dev/null || echo "  Not found yet"

echo ""
echo "--- Recent Log Output (last 20 lines per model) ---"
for model in convnext_tiny mobilenetv3_large resnet50 efficientnet_b3 swin_tiny; do
    log_file="$LOG_DIR/${model}.log"
    if [ -f "$log_file" ]; then
        echo ""
        echo "=== $model ==="
        tail -n 10 "$log_file"
    fi
done

echo ""
echo "--- Progress Summary ---"
python "$SCRIPT_DIR/check_progress.py" 2>/dev/null || echo "  check_progress.py not available"

echo ""
echo "=========================================="

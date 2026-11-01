#!/usr/bin/env bash
# Launch step 1 (LR selection) with full logging: env snapshot, GPU monitor, launcher log.
set -uo pipefail
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate wood
REPO=/root/woodbench
TASK="$REPO/notebooks/phase 2/supervisor-feedback/task_07_lr_selection"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LAUNCH_DIR="$REPO/artifacts/phase_2/supervisor-feedback/task_07_lr_selection/launch-$STAMP"
mkdir -p "$LAUNCH_DIR"
cp "$1" "$LAUNCH_DIR/dryrun.log" 2>/dev/null || true
cp "$REPO/scripts/build_resized_cache.py" "$LAUNCH_DIR/" ; cp "$REPO/data/task-B-group-split-224.sha256" "$LAUNCH_DIR/" 2>/dev/null || true

{
  echo "date_utc: $STAMP"
  echo "host: $(hostname)"
  echo "git_branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "git_commit: $(git -C "$REPO" rev-parse HEAD)"
  echo "git_status:"; git -C "$REPO" status --short
  echo "python: $(python --version 2>&1) ($(which python))"
  python -c "import torch,timm,torchvision;print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version());print('torchvision',torchvision.__version__);print('timm',timm.__version__)"
  echo "MAX_PARALLEL=${MAX_PARALLEL:-5} MODELS=${MODELS:-default} LRS=${LRS:-default}"
  echo "DATA_ROOT=${DATA_ROOT:-default}"
  echo "data_note: DATA_ROOT is a lossless 224x224 cache of data/task-B-group-split made by scripts/build_resized_cache.py (same Resize((224,224)) as the pipeline; 300 random images verified identical tensors). Reason: CPU quota ~24 cores could not decode 1280x1024 PNGs fast enough (GPU ~6%)."
  echo "nproc: $(nproc)"; echo "cpu quota: $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)"; free -g; df -h /
} > "$LAUNCH_DIR/env.txt" 2>&1
pip freeze > "$LAUNCH_DIR/pip_freeze.txt" 2>&1
conda env export -n wood > "$LAUNCH_DIR/conda_env.yaml" 2>&1
nvidia-smi > "$LAUNCH_DIR/nvidia-smi.txt" 2>&1

# GPU monitor every 60 s until the launcher exits.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv -l 60 > "$LAUNCH_DIR/gpu_monitor.csv" 2>&1 &
MON=$!

cd "$TASK"
MAX_PARALLEL="${MAX_PARALLEL:-5}" DATA_ROOT="${DATA_ROOT}" bash run_lr_select.sh > "$LAUNCH_DIR/launch.log" 2>&1
STATUS=$?
kill "$MON" 2>/dev/null
echo "$STATUS" > "$LAUNCH_DIR/launcher.exit"
echo "finished_utc: $(date -u +%Y%m%dT%H%M%SZ) exit=$STATUS" >> "$LAUNCH_DIR/env.txt"
df -h / >> "$LAUNCH_DIR/env.txt"
exit "$STATUS"

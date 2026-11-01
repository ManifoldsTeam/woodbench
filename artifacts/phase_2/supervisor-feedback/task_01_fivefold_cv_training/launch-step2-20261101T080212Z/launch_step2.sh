#!/usr/bin/env bash
# Step 2: origin CV v2, 5 models x 5 folds, with full logging.
set -uo pipefail
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate wood
REPO=/root/woodbench
TASK="$REPO/notebooks/phase 2/supervisor-feedback/task_01_fivefold_cv_training"
ART="$REPO/artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LAUNCH_DIR="$ART/launch-step2-$STAMP"
mkdir -p "$LAUNCH_DIR"
export BACKBONE_LR=1e-4 DATA_ROOT="$REPO/data/task-B-group-split-224" \
       N_FOLDS=5 MAX_PARALLEL=5 USE_90ROTATION=0 RESUME=0
unset MODELS BATCH_SIZE INPUT_VARIANT BACKBONE_LRS
cp "$REPO/scripts/build_resized_cache.py" "$REPO/data/task-B-group-split-224.sha256" "$LAUNCH_DIR/"
cp -r "$1" "$LAUNCH_DIR/smoke_logs" 2>/dev/null || true
cp "$REPO/data/experiments/Task-B-lr-select/reports/"lr-select-*/decisions.csv "$LAUNCH_DIR/step1_decisions.csv" 2>/dev/null || true
{
  echo "date_utc: $STAMP"
  echo "step: 2 (origin CV v2, best-val-origin-cv-v2)"
  echo "git_branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "git_commit: $(git -C "$REPO" rev-parse HEAD)"
  echo "git_status:"; git -C "$REPO" status --short
  echo "python: $(python --version 2>&1) ($(which python))"
  python -c "import torch,timm,torchvision;print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version());print('torchvision',torchvision.__version__);print('timm',timm.__version__)"
  echo "BACKBONE_LR=$BACKBONE_LR (step 1: all 5 models chose 1e-4, see step1_decisions.csv)"
  echo "DATA_ROOT=$DATA_ROOT N_FOLDS=$N_FOLDS MAX_PARALLEL=$MAX_PARALLEL USE_90ROTATION=$USE_90ROTATION RESUME=$RESUME"
  echo "data_note: DATA_ROOT is a lossless 224x224 cache of data/task-B-group-split (scripts/build_resized_cache.py, same Resize((224,224)) as the pipeline; 300 random images verified identical tensors)."
  echo "nproc: $(nproc)"; echo "cpu quota: $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)"; free -g; df -h /
} > "$LAUNCH_DIR/env.txt" 2>&1
pip freeze > "$LAUNCH_DIR/pip_freeze.txt" 2>&1
conda env export -n wood > "$LAUNCH_DIR/conda_env.yaml" 2>&1
nvidia-smi > "$LAUNCH_DIR/nvidia-smi.txt" 2>&1
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv -l 60 > "$LAUNCH_DIR/gpu_monitor.csv" 2>&1 &
MON=$!
cd "$TASK"
bash run_all_models.sh > "$LAUNCH_DIR/launch.log" 2>&1
STATUS=$?
python check_progress.py --verbose > "$LAUNCH_DIR/final_progress.txt" 2>&1
kill "$MON" 2>/dev/null
echo "$STATUS" > "$LAUNCH_DIR/launcher.exit"
echo "finished_utc: $(date -u +%Y%m%dT%H%M%SZ) exit=$STATUS" >> "$LAUNCH_DIR/env.txt"
df -h / >> "$LAUNCH_DIR/env.txt"
exit "$STATUS"

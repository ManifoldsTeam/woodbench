#!/usr/bin/env bash
# Steps 3-4: ConvNeXt fold 1, grayscale and px16, in parallel, with full logging.
set -uo pipefail
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate wood
REPO=/root/woodbench
TASK="$REPO/notebooks/phase 2/supervisor-feedback/task_01_fivefold_cv_training"
ART="$REPO/artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LAUNCH_DIR="$ART/launch-step34-$STAMP"
mkdir -p "$LAUNCH_DIR"
DATA_ROOT="$REPO/data/task-B-group-split-224"
cp "$REPO/scripts/build_resized_cache.py" "$REPO/data/task-B-group-split-224.sha256" "$LAUNCH_DIR/"
{
  echo "date_utc: $STAMP"
  echo "steps: 3 (grayscale) and 4 (px16), convnext_tiny, fold 1, protocol best-val-origin-cv-v2"
  echo "git_branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "git_commit: $(git -C "$REPO" rev-parse HEAD)"
  echo "git_status:"; git -C "$REPO" status --short
  echo "python: $(python --version 2>&1) ($(which python))"
  python -c "import torch,timm,torchvision;print('torch',torch.__version__,'cuda',torch.version.cuda);print('torchvision',torchvision.__version__);print('timm',timm.__version__)"
  echo "backbone_lr=1e-4 (step 1) DATA_ROOT=$DATA_ROOT"
  echo "reference: convnext_tiny RGB fold 1 of step 2 (data/experiments/Task-B-origin-cv-v2/convnext_tiny/fold_1)"
  echo "data_note: DATA_ROOT is a lossless 224x224 cache (scripts/build_resized_cache.py, same Resize((224,224)) as the pipeline)."
  echo "  px16: degrade starts with Resize(224) -> identity on the cache, so inputs are pixel-identical to the original pipeline."
  echo "  grayscale: pipeline applies Grayscale(3) before Resize(224); on the cache Resize happens first, so pixels can differ by rounding (about +-1 level)."
  free -g; df -h /
} > "$LAUNCH_DIR/env.txt" 2>&1
pip freeze > "$LAUNCH_DIR/pip_freeze.txt" 2>&1
conda env export -n wood > "$LAUNCH_DIR/conda_env.yaml" 2>&1
nvidia-smi > "$LAUNCH_DIR/nvidia-smi.txt" 2>&1
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv -l 60 > "$LAUNCH_DIR/gpu_monitor.csv" 2>&1 &
MON=$!
cd "$TASK"
export PYTHONUNBUFFERED=1
pids=()
for variant in grayscale px16; do
  python train_kfold.py --backbone convnext_tiny --backbone_lr 1e-4 --fold 1 \
    --input_variant "$variant" --data_root "$DATA_ROOT" \
    --output_root "$REPO/data/experiments/Task-B-origin-cv-v2-$variant" \
    > "$LAUNCH_DIR/convnext_tiny_$variant.log" 2>&1 &
  pids+=("$!")
done
STATUS=0
for i in 0 1; do
  s=0; wait "${pids[$i]}" || s=$?
  echo "$s" > "$LAUNCH_DIR/$( [ $i = 0 ] && echo grayscale || echo px16 ).exit"
  (( s != 0 )) && STATUS=1
done
for variant in grayscale px16; do
  python check_progress.py --model convnext_tiny --verbose \
    --output "$REPO/data/experiments/Task-B-origin-cv-v2-$variant" > "$LAUNCH_DIR/final_progress_$variant.txt" 2>&1
done
kill "$MON" 2>/dev/null
echo "$STATUS" > "$LAUNCH_DIR/launcher.exit"
echo "finished_utc: $(date -u +%Y%m%dT%H%M%SZ) exit=$STATUS" >> "$LAUNCH_DIR/env.txt"
exit "$STATUS"

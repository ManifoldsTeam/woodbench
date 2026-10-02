# 5-Fold Cross-Validation by Origin Key

## Quick Start (Local Dry Run)

### Test trên local trước khi chạy trên vast.ai:

```bash
cd notebooks/phase\ 2/supervisor-feedback/task_01_fivefold_cv_training/

# Dry run để verify setup
python train_kfold.py --backbone convnext_tiny --dryrun

# Dry run với verbose
python train_kfold.py --backbone convnext_tiny --dryrun -v
```

Dry run sẽ kiểm tra:
- ✅ Dataset tồn tại
- ✅ Origin key extraction
- ✅ Split distribution
- ✅ Model khả dụng
- ✅ GPU available
- ✅ Output path writable

---

## Quy trình chạy trên vast.ai

### Bước 1: SSH vào vast.ai

```bash
ssh user@your-vast-instance
```

### Bước 2: Setup environment (lần đầu)

```bash
# Clone repo
git clone https://github.com/your-repo/woodbench.git
cd woodbench

# Cài conda env
conda env create -f environment.yml
conda activate wood

# Tải dataset từ rclone
rclone sync remote:dataset ./data/task-B-group-split
```

### Bước 3: Setup WandB

```bash
# Login
wandb login

# Hoặc set API key
export WANDB_API_KEY=your_api_key
```

### Bước 4: Dry run để verify

```bash
cd notebooks/phase\ 2/supervisor-feedback/task_01_fivefold_cv_training/

# Verify với 1 model
python train_kfold.py --backbone convnext_tiny --dryrun
```

### Bước 5: Chạy tất cả models

**Shell 1: Chạy training (dùng `nohup`)**

```bash
cd notebooks/phase\ 2/supervisor-feedback/task_01_fivefold_cv_training/

# Tạo logs dir
mkdir -p logs

# Chạy tất cả models trong background
nohup bash run_all_models.sh > kfold.log 2>&1 &

# Ghi PID để track
echo $! > training.pid
cat training.pid
```

**Shell 2: Monitor progress (mở SSH tab mới)**

```bash
# Check nhanh
python check_progress.py

# Chi tiết hơn
python check_progress.py -v

# Watch mode (auto refresh 30s)
bash monitor_progress.sh
```

**Shell 3: Check GPU status**

```bash
# GPU usage
nvidia-smi

# Processes
ps aux | grep train_kfold
```

### Bước 6: SSH disconnect

**Q: Tắt kết nối SSH thì process có bị kill không?**

**A: KHÔNG** - Vì dùng `nohup`:
- `nohup` chạy process độc lập với terminal
- SIGHUP signal không ảnh hưởng
- Process tiếp tục chạy ngay cả khi logout

**Để verify:**

```bash
# Sau khi chạy nohup
echo $!  # Lưu PID

# Thoát SSH
exit

# SSH lại sau vài phút
ssh user@vast

# Check process vẫn chạy
ps aux | grep train_kfold
```

### Bước 7: Check lại progress

```bash
cd notebooks/phase\ 2/supervisor-feedback/task_01_fivefold_cv_training/

# Check progress
python check_progress.py

# Check logs
tail -f logs/convnext_tiny.log
```

---

## ZSH vs TERMUX

### Vấn đề với Termux (Android)

| Vấn đề | Mô tả |
|---------|--------|
| Process killed on disconnect | Termux giữ process trong app, nhưng SSH session không persist |
| Limited shell | Không có `tmux` hay `screen` tốt |
| Resource limits | Android resource limits |

### Giải pháp: Cài zsh trên Linux host

**Trên vast.ai instance (Linux):**

```bash
# Cài zsh
sudo apt update && sudo apt install -y zsh

# Cài oh-my-zsh (optional)
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Set zsh as default shell
chsh -s /bin/zsh
```

### So sánh

| Feature | Termux | Zsh (Linux) |
|---------|--------|-------------|
| Background processes | ⚠️ Unreliable | ✅ Reliable |
| nohup | ⚠️ May not work | ✅ Works |
| tmux/screen | ❌ Not available | ✅ Available |
| Multiple sessions | ❌ Limited | ✅ Unlimited |
| Resource limits | ⚠️ Android limits | ✅ Full Linux |

### Recommendation

**Nếu dùng vast.ai (Linux):**
- Cài zsh và dùng `nohup` 
- Hoặc dùng `tmux` để persist sessions

**Nếu dùng Termux (Android):**
- Dùng Termux session giữ mở
- Hoặc dùng `nohup` nhưng cần test kỹ

---

## tmux (Alternative to nohup)

Nếu muốn interactive session persistence:

```bash
# Tạo tmux session
tmux new -s training

# Chạy training
python train_kfold.py --backbone convnext_tiny

# Detach: Ctrl+B, D

# Thoát SSH

# SSH lại và attach
tmux attach -s training
```

### Commands tmux

| Command | Description |
|---------|-------------|
| `tmux new -s name` | Tạo session mới |
| `Ctrl+B, D` | Detach (thoát nhưng giữ chạy) |
| `tmux attach -s name` | Attach lại session |
| `tmux ls` | List all sessions |
| `tmux kill-session -t name` | Kill session |

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL MACHINE                                 │
│  1. Dry run test                                               │
│     python train_kfold.py --backbone convnext_tiny --dryrun    │
│                                                                 │
│  2. Sync to vast                                               │
│     rsync -avz notebooks/ user@vast:/workspace/                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VAST.AI INSTANCE                             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Shell 1   │  │  Shell 2    │  │  Shell 3    │            │
│  │             │  │             │  │             │            │
│  │ nohup      │  │ check_      │  │ nvidia-smi  │            │
│  │ run_all    │  │ progress.py │  │             │            │
│  │ _models.sh │  │             │  │             │            │
│  │             │  │ tail -f    │  │             │            │
│  │ (RUNNING)  │  │ logs/*.log  │  │             │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  SSH DISCONNECT ─────────────────────────────────────────────► │
│  (process vẫn chạy vì nohup)                                 │
│                                                                 │
│  SSH RECONNECT ──────────────────────────────────────────────► │
│  python check_progress.py                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dry Run Output Example

```
============================================================
🔍 DRYRUN MODE - Verifying setup
============================================================

✅ Dataset: /path/to/data/task-B-group-split
   Samples: 3365
   Classes: 17
   Files: 3365

✅ Origin distribution:
   Unique origins: 89
   Min samples/origin: 1
   Max samples/origin: 120
   Mean samples/origin: 37.8

✅ 5-Fold Split:
   Fold 1: 71 train origins, 18 val origins
   Fold 2: 71 train origins, 18 val origins
   Fold 3: 71 train origins, 18 val origins
   Fold 4: 71 train origins, 18 val origins
   Fold 5: 71 train origins, 18 val origins

✅ Model: convnext_tiny
   Parameters: 27,967,312

✅ GPU: cuda
   Name: NVIDIA GeForce RTX 3060
   Memory: 12.5 GB

✅ Output: /path/to/experiments/Task-B-origin-cv/convnext_tiny

============================================================
✅ DRYRUN PASSED - Setup is valid
============================================================

To start training, run without --dryrun:
  python train_kfold.py --backbone convnext_tiny --n_folds 5
```

---

## Commands Reference

### Training

```bash
# Dry run
python train_kfold.py --backbone convnext_tiny --dryrun

# Train 1 model (all 5 folds)
python train_kfold.py --backbone convnext_tiny --n_folds 5 --epochs 30

# Train 1 fold only (debug)
python train_single_fold.py --backbone convnext_tiny --fold 1

# Resume from checkpoint
python train_kfold.py --backbone convnext_tiny --resume

# All models
bash run_all_models.sh
```

### Monitoring

```bash
# Quick progress
python check_progress.py

# Verbose
python check_progress.py -v

# Watch mode
bash monitor_progress.sh

# Check status + GPU
bash check_status.sh

# Check specific model
python check_progress.py --model convnext_tiny --verbose
```

### Process Management

```bash
# Kill all training
pkill -f train_kfold.py

# Kill specific model
pkill -f convnext_tiny

# Check running processes
ps aux | grep train_kfold

# Check GPU
nvidia-smi
```

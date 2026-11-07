# Handoff 07-11-2026: chạy yêu cầu 6–7 của thầy (14 loài) trên Vast

Tài liệu này đủ để chạy trên một instance Vast mới, chỉ cần repo woodbench và cache ảnh trên Drive. Code đã viết và
kiểm trên máy local (RTX 3060) ngày 07-11-2026, nhánh `feat/supervisor-2026-11-06`. Máy local không train vì còn dùng
cho việc khác; mọi run thật chạy trên Vast.

## 0. Tóm tắt

| Mục | Nội dung |
|---|---|
| Việc | 50 run full training, xếp thành 22 job trong một hàng đợi, 5 job chạy song song trên một RTX 4090 |
| Lệnh chính | `bash run_clean14.sh` (mục 4) |
| Dữ liệu | `drive:wood/2026-11-07-task-B-group-split-224-cache/task-B-group-split-224.zip` (410 MB) |
| Thời gian ước lượng | khoảng 30 phút chuẩn bị + 2,5–3,5 giờ chạy (mục 7) |
| Mang về | `artifacts/.../task_15_clean14_runs/results/` (git) và checkpoint (zip lên Drive, mục 6) |

## 1. Thầy yêu cầu gì

Nguyên văn (06-11-2026): "Mấy cái này chạy trước trên 14 loài sạch (4.115 ảnh, 140 mẫu)

6. CV 5 model × 5 fold (25 run) và Final-B 5 model × 3 seed (15 run), dùng code chọn checkpoint mới. Nếu sau này 3 loài
   sửa xong, kết quả này thành phân tích phụ. Nếu không sửa được, đây là kết quả chính của bài.
7. Grayscale và 16px: ConvNeXt × 5 fold trên 14 loài (10 run).
8. Kiểm tra tái lập kết quả tháng 8 [...]"

Yêu cầu 8 đã xong ở máy local (`artifacts/phase_2/supervisor-feedback/task_14_august_reproduction/`), không cần Vast.

**14 loài** = 17 loài trừ Callitris columellaris, Cunninghamia lanceolata, Fokienia hodginsii (3 loài đang kiểm ảnh trùng,
task 10). Mỗi loài còn lại có đúng 10 origin.

## 2. Quy tắc chung cho cả 50 run

| Thành phần | Giá trị |
|---|---|
| LR | backbone 1e-4, head 1e-4 (bước chọn LR 01-11 chọn 1e-4 cho cả 5 model); head = `model.get_classifier()` |
| Lịch LR | 5 epoch warmup tuyến tính + cosine, tối đa 50 epoch |
| Chọn checkpoint | `best.pth` = val loss thấp nhất **từ epoch 6** (bỏ warmup); early stopping patience 15 theo val loss |
| Chấm | fold held-out (CV) hoặc val + test (Final-B) chấm **một lần** bằng `best.pth` |
| Giữ như Final-B main | AdamW wd 1e-4, label smoothing 0,1, AMP, batch theo model (ResNet/MobileNet 64, còn lại 32), transform `src/data/transforms.py` |
| Dữ liệu | cache 224×224 `data/task-B-group-split-224`, tensor đầu vào giống hệt ảnh gốc qua `Resize((224,224))` |

### Fold CV giữ nguyên như CV v2 (01-11)

`make_origin_folds` xáo origin từng loài nối tiếp nhau trên một luồng RNG. Nếu bỏ 3 loài trước khi chia, mọi loài đứng sau
sẽ nhận fold khác. Vì vậy `--species_set clean14` vẫn chia fold trên 17 loài như CV v2, sau đó mới bỏ 3 loài
(`origin_cv.drop_species`). Đã kiểm trên dữ liệu thật: với cả 5 fold, origin train / inner-val / held-out của 14 loài
trùng khớp với `metrics.json` của CV v2. Nhờ vậy có thể so từng fold giữa model 17 loài cũ và model 14 loài mới
(`cv_vs_17species.csv`).

| Fold | Train | Val | Held-out |
|---:|---:|---:|---:|
| 1 | 2466 | 833 | 816 |
| 2 | 2448 | 843 | 824 |
| 3 | 2453 | 815 | 847 |
| 4 | 2475 | 827 | 813 |
| 5 | 2500 | 800 | 815 |

Final-B dùng split cố định: train 2.880 / val 803 / test 432.

### Ma trận run

| Yêu cầu | Run | Output (`data/experiments/`) |
|---|---:|---|
| 6, CV | 5 model × 5 fold = 25 | `Task-B-clean14-cv/<model>/fold_<k>/` |
| 6, Final-B | 5 model × seed 42/123/456 = 15 | `Task-B-clean14-final/seed_<s>/<model>/` |
| 7 | ConvNeXt × 5 fold × {grayscale, px16} = 10 | `Task-B-clean14-cv-grayscale/`, `Task-B-clean14-cv-px16/` |

Grayscale là `Grayscale(3)`, px16 là resize 224→16→224 bilinear. Cả hai áp dụng cho train, val và held-out, giống bước
3–4 ngày 01-11.

## 3. Chuẩn bị

### 3.1 Ở máy local, trước khi thuê Vast

Nhánh `feat/supervisor-2026-11-06` mới chỉ có ở local. Đẩy lên GitHub để Vast clone được:

```zsh
cd ~/Code/ManifoldsTeam/woodbench
git push -u origin feat/supervisor-2026-11-06
```

Nếu muốn tải dữ liệu bằng rclone trên Vast, chép config rclone sang instance (file này chứa token Drive, **xóa trước khi
Destroy instance**):

```zsh
ssh <vast> 'mkdir -p ~/.config/rclone'
scp ~/.config/rclone/rclone.conf <vast>:~/.config/rclone/
```

Cách khác, không cần rclone: zip lại ở local rồi `scp` sang, sau đó kiểm như mục 3.3 (bỏ bước kiểm sha256 của file zip).

```zsh
cd ~/Code/ManifoldsTeam/woodbench/data && zip -0 -r -q /tmp/task-B-group-split-224.zip task-B-group-split-224
scp /tmp/task-B-group-split-224.zip <vast>:/root/woodbench/data/
```

### 3.2 Code và môi trường (Vast)

```bash
cd /root
git clone -b feat/supervisor-2026-11-06 git@github.com:ManifoldsTeam/woodbench.git
cd woodbench
conda create -y -n wood python=3.12 && conda activate wood
pip install torch==2.15.0 torchvision==0.29.2 timm==1.0.30     # bản đã dùng ngày 01-11 trên Vast
pip install -r requirements.txt
python -c "import torch, timm; assert torch.cuda.is_available(); print(torch.__version__, timm.__version__, torch.cuda.get_device_name(0))"
```

`scripts/setup_ssh_server.sh` mà `CLAUDE.md` nhắc tới không có trong repo; cài như trên. Nếu pip không có đúng bản
torch 2.15.0 thì dùng bản có sẵn và ghi lại: kết quả GPU vốn không tái lập bit-exact (cuDNN/AMP), và `pip_freeze.txt` được
launcher lưu cùng log.

### 3.3 Dữ liệu

Folder trên Drive: `drive:wood/2026-11-07-task-B-group-split-224-cache/`

| File | Nội dung |
|---|---|
| `task-B-group-split-224.zip` | 4.800 ảnh 224×224 (17 loài; code tự bỏ 3 loài) + `task-B-group-split-224.sha256` bên trong |
| `task-B-group-split-224.zip.sha256` | sha256 của file zip: `165551b2…789e` |
| `README.txt` | mô tả và cách kiểm |

```bash
cd /root/woodbench/data
rclone copy -P drive:wood/2026-11-07-task-B-group-split-224-cache/ .
sha256sum -c task-B-group-split-224.zip.sha256
unzip -q task-B-group-split-224.zip
sha256sum -c --quiet ../docs/timeline/01-11-2026-task-B-group-split-224.sha256 && echo DATA OK
for s in train val test; do echo "$s $(find task-B-group-split-224/$s -name '*.png' | wc -l)"; done
# phải ra: train 3365, val 903, test 532
rm task-B-group-split-224.zip
```

Launcher và `train_final_b.py` cũng tự kiểm lại sha256 trước khi chạy. Không cần ảnh full-res `data/task-B-group-split`:
đọc ảnh gốc 2560×2048 khiến CPU thành nút cổ chai (Final-B tháng 8 mất khoảng 7 giờ vì lý do này).

### 3.4 Test, dry-run, tải trọng số

```bash
cd /root/woodbench
PYTHONPATH="$PWD" pytest -q tests/test_clean14_runs.py tests/test_origin_cv.py tests/test_cv_training_runs.py
# local 07-11: 69 passed

cd "notebooks/phase 2/supervisor-feedback/task_15_clean14_runs"
python train_final_b.py --backbone convnext_tiny --seed 42 --dryrun
python ../task_01_fivefold_cv_training/train_kfold.py --backbone convnext_tiny --backbone_lr 1e-4 \
  --species_set clean14 --data_root ../../../../data/task-B-group-split-224 --dryrun
# phải thấy: 14 species, train 2880 / val 803 / test 432; CV dataset 4115, 5 fold như bảng mục 2

# Tải trước trọng số ImageNet, để 5 job không tải cùng lúc
python -c "
import timm, sys; sys.path.insert(0, '../../../..')
from src.models.factory import BACKBONE_NAMES
for name in BACKBONE_NAMES.values(): timm.create_model(name, pretrained=True)
print('weights cached')"
```

W&B là tùy chọn: chỉ log khi đã `export WANDB_API_KEY=...` (project `woodbench-clean14`). Không có key thì vẫn chạy bình
thường.

## 4. Chạy

```bash
cd "/root/woodbench/notebooks/phase 2/supervisor-feedback/task_15_clean14_runs"
MAX_PARALLEL=5 nohup bash run_clean14.sh > "launch-$(date -u +%Y%m%dT%H%M%SZ).out" 2>&1 < /dev/null &
disown
```

- Không đối số = cả ba phần, theo thứ tự `cv degraded final`. Chạy riêng phần nào thì truyền tên phần đó, ví dụ
  `bash run_clean14.sh final`.
- Thứ tự hàng đợi: 5 job CV (mỗi job = 1 model × 5 fold chạy nối tiếp, vì `train_kfold.py` khóa thư mục model), rồi 2 job
  ConvNeXt grayscale/px16, rồi 15 run Final-B. Hết một job thì job tiếp theo vào.
- Ngày 01-11, 5 job song song chạy ổn trên 4090 24 GB với quota khoảng 24 core CPU. Nếu `nvidia-smi` cho thấy GPU chưa đầy
  và VRAM còn trên 8 GB, có thể dùng `MAX_PARALLEL=6` cho lần sau. Đừng đổi giữa chừng.
- **Chạy lại an toàn:** nếu instance hoặc job bị ngắt, chạy lại đúng lệnh trên. Fold/run đã xong (có `metrics.json`) được
  giữ, còn fold/run dở thì train lại từ epoch 1. Không có resume giữa chừng một run.

Log: `data/experiments/clean14-logs/<UTC>/` chứa:

- một `<job>.log` và một `<job>.exit` cho mỗi job;
- `launch.txt` (commit, git status);
- `pip_freeze.txt` và `nvidia-smi.txt`.

## 5. Theo dõi

```bash
cd /root/woodbench
L=$(ls -d data/experiments/clean14-logs/* | tail -1)
cat "notebooks/phase 2/supervisor-feedback/task_15_clean14_runs"/launch-*.out | tail      # job nào bắt đầu/xong
ls $L/*.exit 2>/dev/null | wc -l                                                          # số job xong / 22
python "notebooks/phase 2/supervisor-feedback/task_01_fivefold_cv_training/check_progress.py" \
  --output data/experiments/Task-B-clean14-cv                                             # tiến độ CV theo fold
ls data/experiments/Task-B-clean14-final/seed_*/*/metrics.json 2>/dev/null | wc -l        # Final-B xong / 15
grep -h "best epoch" $L/final_*.log                                                       # kết quả Final-B từng run
tr '\r' '\n' < $L/cv_efficientnet_b3.log | grep epoch | tail -3                           # epoch gần nhất của 1 job
nvidia-smi
```

Một job lỗi: xem `$L/<job>.log`. Sửa xong thì chạy lại launcher; các phần đã xong không bị chạy lại.

## 6. Sau khi xong

### 6.1 Tổng hợp

```bash
cd "/root/woodbench/notebooks/phase 2/supervisor-feedback/task_15_clean14_runs"
python summarize_clean14.py
cat ../../../../artifacts/phase_2/supervisor-feedback/task_15_clean14_runs/results/summary.md
```

`summarize_clean14.py` gọi `cv_reports.export_reports` cho từng thư mục CV (registry, OOF, per-species, giống CV v2) và
dừng nếu thiếu fold nào. Script ghi `results/` gồm:

- `summary.md` (mọi bảng);
- `cv_summary.csv`, `cv_folds.csv`;
- `cv_vs_17species.csv` (cùng fold, cùng 14 loài: model 17 loài CV v2 vs model 14 loài);
- `degraded_summary.csv`;
- `final_b_runs.csv`, `final_b_summary.csv`, `final_b_per_species_f1.csv`;
- `checkpoints.sha256`;
- `runs/` (mọi json/csv của từng run, không có `.pth`).

### 6.2 Mang kết quả về (git)

```bash
cd /root/woodbench
cp -r data/experiments/clean14-logs artifacts/phase_2/supervisor-feedback/task_15_clean14_runs/logs
git add artifacts/phase_2/supervisor-feedback/task_15_clean14_runs
git commit -m "Add 14-species CV, Final-B and grayscale/16px results"
git push
```

### 6.3 Backup checkpoint lên Drive (bắt buộc trước khi Destroy)

Khoảng 4 GB `best.pth` chỉ có trên instance. Nén từng thư mục (zip `-0` vì `.pth` gần như không nén được), lên Drive, rồi
kiểm:

```bash
cd /root/woodbench/data/experiments
D=/root/clean14-upload; mkdir -p $D
for r in Task-B-clean14-cv Task-B-clean14-cv-grayscale Task-B-clean14-cv-px16 Task-B-clean14-final clean14-logs; do
  zip -0 -r -q $D/$r.zip $r
done
cp ../../artifacts/phase_2/supervisor-feedback/task_15_clean14_runs/results/checkpoints.sha256 $D/
(cd $D && md5sum *.zip > zip.md5)
rclone copy -P $D drive:wood/2026-11-07-clean14-runs/
rclone check $D drive:wood/2026-11-07-clean14-runs/     # phải ra 0 differences
```

Nếu chạy vào ngày khác thì đổi ngày trong tên folder. Thêm một `README.txt` ngắn vào folder Drive (thầy chỉ xem được
Drive), như `drive:wood/2026-11-01-lr-select-cv-v2/README.txt`. Sau đó xóa `~/.config/rclone/rclone.conf` trên instance.

## 7. Ước lượng thời gian

Căn cứ: CV v2 ngày 01-11 trên Vast 4090, 5 job song song, 17 loài. Thời gian mỗi epoch, đo từ `run_date`/`finished_at`
trong `metrics.json`:

| Model | s/epoch (5 job song song) | Epoch đã chạy, quy tắc cũ (patience 10) |
|---|---:|---:|
| MobileNetV3 | 10–12 | 41–50 |
| ResNet-50 | 10–17 | 38–45 |
| ConvNeXt-Tiny | 18–20 | 13–31 |
| Swin-Tiny | 20–24 | 15–31 |
| EfficientNet-B3 | 14–24 | 34–50 |

Quy tắc mới chạy ít nhất 21 epoch (5 warmup + epoch best + 15 patience). Trong trial ngày 06-11, ConvNeXt fold 3 chạy 34
epoch (cũ: 13). Dữ liệu 14 loài bằng khoảng 86% dữ liệu 17 loài. Suy luận: mỗi run khoảng 7–17 phút. Một job CV (5 fold)
mất khoảng 50–90 phút. Tổng khối lượng khoảng 550–650 phút-slot, chia cho 5 slot ra **khoảng 2–2,5 giờ**, cộng phần đuôi
hàng đợi thành **2,5–3,5 giờ**. Thêm khoảng 30 phút chuẩn bị (mục 3) và 15 phút backup.

So với Final-B tháng 8 (khoảng 7 giờ cho 15 run): lúc đó đọc ảnh full-res nên CPU giải mã PNG là nút cổ chai. Với cache 224,
CV v2 25 run ngày 01-11 chỉ mất 1 giờ 08.

## 8. Lưu ý

1. Số val trong log lúc train tính bằng AMP, trung bình theo batch. Số val trong `metrics.json` của Final-B tính lại bằng
   FP32, trung bình theo ảnh, nên có thể lệch nhẹ (smoke test: 89,04% so với 88,92%). Để báo cáo, dùng số trong
   `metrics.json`.
2. Kết quả không tái lập bit-exact giữa các GPU hoặc các lần chạy (cuDNN/AMP). Trên test Final-B, 1 ảnh = 0,23 điểm (432
   ảnh). Chênh lệch dưới khoảng 2–3 ảnh nên coi là nhiễu.
3. So sánh với số cũ:
   - CV v2 (01-11) dùng quy tắc cũ: val acc, từ epoch 1, patience 10.
   - Final-B tháng 8 dùng LR thực tế khoảng 1e-4 do lỗi nhóm head (xem task 14) và chọn checkpoint theo val acc.
   - Số 14 loài mới khác ở cả quy tắc chọn checkpoint lẫn tập loài. Cột `cv_vs_17species.csv` chỉ cố định fold và tập ảnh
     test.
4. Nếu 3 loài sửa xong và thầy muốn chạy lại 17 loài theo quy tắc mới: dùng cùng launcher với `--species_set all17` và
   cache mới (phải tạo lại cache + sha256 bằng `scripts/build_resized_cache.py`). Việc này ngoài phạm vi tài liệu này.
5. Code task 15 không sửa `src/train/trainer.py` (`TrainingEngine.fit`, GitNexus impact HIGH). Final-B dùng vòng lặp
   `train_epoch`/`validate` của `train_kfold.py`, nên CV và Final-B theo cùng một quy tắc.

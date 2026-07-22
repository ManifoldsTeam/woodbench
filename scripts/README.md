# Scripts

## Entry point chinh

Training dung **Hydra** tai repo root:

```bash
conda activate wood
python train.py experiment=taskB model.backbone=resnet50 train.seed=42
```

Checkpoints: `experiments/{task.id}/{backbone}/seed_{seed}/` (override: `output.root=...`)

## Batch launcher

```bash
python scripts/run_experiments.py --experiment taskB --backbone convnext_tiny --seeds 42 123 456 --max_workers 2
```

## Khong dung

- `scripts/train.py` da xoa — trung voi `train.py` (Hydra), gay nham lan.

## Khac

- `scripts/evaluate.py` — danh gia checkpoint da train
- `scripts/eval_taskA.py` — legacy Task A

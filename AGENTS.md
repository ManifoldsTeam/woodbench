<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **woodbench** (1714 symbols, 2197 relationships, 31 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/woodbench/context` | Codebase overview, check index freshness |
| `gitnexus://repo/woodbench/clusters` | All functional areas |
| `gitnexus://repo/woodbench/processes` | All execution flows |
| `gitnexus://repo/woodbench/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

## Reference Documents (WoodID-S2-2026)

Khi lam viec voi cac feature lien quan den WoodID-S2-2026, **cac tai lieu sau la tai lieu goc va can duoc doi chieu** sau khi co ket qua experiment:

| Document             | Path                                        | Purpose                                      |
| -------------------- | ------------------------------------------- | -------------------------------------------- |
| WoodID Plan          | `docs/references/WoodID_plan.md`            | Project plan, experiment checklist, timeline |
| WoodID Assignment S2 | `docs/references/WoodID_Assignment_S2.docx` | Research assignment specification            |
| Tree Reference       | `docs/references/tree.txt`                  | IC4SD research assignment document           |

**Rule**: Sau khi hoan thanh experiment, **PHAI doi chieu ket qua** voi cac tai lieu goc nay de dam bao alignment voi yeu cau cua de tai.

# Always Use Conda Environment

- **MUST use conda environment `wood` for all commands.** Always run `conda activate wood` before executing any Python, pip, or shell commands. Check the environment is active with `conda info --envs` if unsure.

---

# Experiment Registry and Report Traceability

Khi chạy training/evaluation và tạo báo cáo:

1. **Mỗi đợt tổng hợp run PHẢI tạo một registry CSV riêng** theo mẫu `<name>-<YYYYMMDDTHHMMSSZ>-registry.csv`. Không ghi đè registry của đợt chạy trước.
2. **Mỗi dòng registry tương ứng đúng một training run** (không tách validation/test thành hai dòng).
3. Registry tối thiểu phải có các cột: `run_id`, `run_date`, `notebook`, `git_commit`, `split_version`, `num_classes`, `n_train`, `n_val`, `n_test`, `seed`, `lr`, `epochs`, `batch`, `image_size`, `label_smoothing`, `augmentation`, `checkpoint_path`, `prediction_path`, cùng toàn bộ metric validation/test dùng trong báo cáo.
4. `run_id` phải ổn định và duy nhất trong registry. Đường dẫn checkpoint/prediction phải tương đối theo `ROOT` để chạy được ở local và Colab.
5. **Mọi dòng trong mọi bảng báo cáo phải truy ngược được về registry**: bảng theo từng run phải có `run_id`; bảng tổng hợp nhiều run phải có `source_run_ids`; bảng so sánh phải có cả `run_id` và run đối chứng (ví dụ `baseline_run_id`). Các ID này phải tồn tại trong đúng registry được khai báo trong metadata báo cáo.
6. Logic hậu xử lý (tính lại metric, xuất prediction, tạo report CSV và registry) phải nằm trong một notebook riêng, đặt cùng tầng với các notebook training tương ứng.

---

# Colab Notebook Convention

Khi tao hoac refactor notebook training/evaluation trong `notebooks/`:

1. **Cell 1 (bat buoc) — Root Detection:**

```python
from pathlib import Path
import sys

if Path('/content').exists():
    ROOT = Path('/content/woodbench')
else:
    ROOT = Path.cwd()

sys.path.insert(0, str(ROOT))
```

2. **Moi path dung ROOT**, khong hard-code:
   - `DATA_DIR = ROOT / "data" / "split"`
   - `EXP_DIR  = ROOT / "experiments" / "..."`

3. **Colab-only cells** phai co guard:
```python
if Path('/content').exists():
    !pip install -q -r {ROOT}/requirements.txt
```

4. **Download dataset** tu HuggingFace (chi khi chua co):
```python
if not DATA_DIR.exists():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="ny0g507h3p/woodbench_A", local_dir=str(DATA_DIR))
```

5. Xem `.cursor/rules/timeline.mdc` phan 4 de biet day du quy tac.

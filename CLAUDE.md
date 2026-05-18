<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **woodbench** (1004 symbols, 1243 relationships, 16 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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

Khi làm việc với các feature liên quan đến WoodID-S2-2026, **các tài liệu sau là tài liệu gốc và cần được đối chiếu** sau khi có kết quả experiment:

| Document             | Path                                        | Purpose                                      |
| -------------------- | ------------------------------------------- | -------------------------------------------- |
| WoodID Plan          | `docs/references/WoodID_plan.md`            | Project plan, experiment checklist, timeline |
| WoodID Assignment S2 | `docs/references/WoodID_Assignment_S2.docx` | Research assignment specification            |
| Tree Reference       | `docs/references/tree.txt`                  | IC4SD research assignment document           |

**Rule**: Sau khi hoàn thành experiment, **PHẢI đối chiếu kết quả** với các tài liệu gốc này để đảm bảo alignment với yêu cầu của đề tài.

# SSH Server Convention

## Conda Environment

- **MUST use conda environment `wood` for all commands.** Always run `conda activate wood` before executing any Python, pip, or shell commands. Check the environment is active with `conda info --envs` if unsure.

## Storage Layout
```
# Local experiments (fast SSD, primary storage)
${HOME}/experiments/woodbench/
    task-A/
        resnet50/
        convnext_tiny/
        ...
    task-B/
        ...

# Dataset (HuggingFace)
${HOME}/woodbench/data/task-A/   # or data/task-B/
${HOME}/woodbench/data/split/
```

## Dataset Download from HuggingFace

Dataset is hosted on HuggingFace Hub. Download once per server:

```bash
# Task A dataset (balanced, 7,264 train samples)
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ny0g507h3p/wood-task-A', local_dir='data/task-A')"

# Task B dataset (species-level, full)
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ny0g507h3p/wood-task-B', local_dir='data/task-B')"
```

Or in Python:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ny0g507h3p/wood-task-A", local_dir="data/task-A")
```

## Server Setup

First time on a fresh server:

```bash
# Full setup (system packages + conda + dirs)
bash scripts/setup_ssh_server.sh --full

# Python deps only (if base system already ready)
bash scripts/setup_ssh_server.sh --python

# Check environment
bash scripts/setup_ssh_server.sh --check
```

## SSH Server Notebook Convention

Khi tao hoac refactor notebook training/evaluation trong `notebooks/`:

1. **Cell 1 (bat buoc) — Root Detection:**

```python
from pathlib import Path
import sys

# SSH server: detect from notebook location (cwd)
ROOT = Path.cwd().parent.parent  # notebooks/phase-X/ -> woodbench/

sys.path.insert(0, str(ROOT))

import torch
print(f"[ROOT] {ROOT}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

2. **Moi path dung ROOT**, khong hard-code:

   - `DATA_DIR = ROOT / "data" / "task-A"`
   - `EXP_DIR  = Path.home() / "experiments" / "woodbench" / "task-A" / "convnext_tiny_balanced"`

3. **Output direction** tren server:

   - Luu checkpoint va ket qua truc tiep vao `~/experiments/woodbench/`

4. **Dataset tren server**:

   - Tai tu HuggingFace truoc khi training (chi can tai mot lan)
   - Dataset tren server: `~/woodbench/data/task-A/` hoac `~/woodbench/data/task-B/`
   - Xem phan "Dataset Download from HuggingFace" o tren de biet cach tai

5. Xem `.cursor/rules/timeline.mdc` phan SSH Notebook Convention de biet day du quy tac.

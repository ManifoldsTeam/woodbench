<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **woodbench** (693 symbols, 817 relationships, 9 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST use conda environment `wood` for all commands.** Always run `conda activate wood` before executing any Python, pip, or shell commands. Check the environment is active with `conda info --envs` if unsure.
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

| Document | Path | Purpose |
|---------|------|---------|
| WoodID Plan | `docs/references/WoodID_plan.md` | Project plan, experiment checklist, timeline |
| WoodID Assignment S2 | `docs/references/WoodID_Assignment_S2.docx` | Research assignment specification |
| Tree Reference | `docs/references/tree.txt` | IC4SD research assignment document |

**Rule**: Sau khi hoàn thành experiment, **PHẢI đối chiếu kết quả** với các tài liệu gốc này để đảm bảo alignment với yêu cầu của đề tài.

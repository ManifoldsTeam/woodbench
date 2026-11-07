# Supervisor-feedback artifacts

Canonical location since 2026-10-13 for outputs formerly stored beside the
Phase 2 notebooks. Notebook/script sources remain in
`notebooks/phase 2/supervisor-feedback/` (paths here are relative to woodbench).

| Directory | Contents |
|---|---|
| `reports/` | Combined inference summary, metadata and registry |
| `task_01_fivefold_cv_training/` | Launcher logs and exit records only |
| `task_02_robustness_tests/` | Historical plots/CSV plus timestamped inference runs |
| `task_03_dark_artifact_check/` | Brightness CSV and inspection figures |
| `task_04_degradation_rerun/` | Historical plots/CSV plus timestamped inference runs |
| `task_06_error_analysis/` | Error CSVs and origin plot |
| `task_09_checkpoint_selection/` | 06-11-2026: val-loss / skip-warmup selection, trial run outputs and env record |
| `task_10_near_duplicate_grouping/` | 06-11-2026: image -> near-duplicate group table for Callitris, Cunninghamia, Fokienia |
| `task_11_specimen_inspection/` | 06-11-2026: error, quality and overlap checks of specimens 3384.4 and 3393.1 |
| `task_12_figure1_species_grid/` | 06-11-2026: Fig. 1, 17-species grid (PDF/PNG), candidate sheet and selection table |
| `task_13_imaging_parameters/` | 06-11-2026: imaging facts per species, inconsistencies and questions for the data team |
| `task_14_august_reproduction/` | 07-11-2026: ConvNeXt seed 42 Final-B rerun: August code vs true 1e-5 without warmup |
| `task_15_clean14_runs/` | 07-11-2026: 14-species CV (25 runs), Final-B (15 runs), ConvNeXt grayscale/16px CV (10 runs), run on Vast; executed notebook, `results/`, `logs/`; checkpoints on `drive:wood/2026-11-07-clean14-runs/` |

Checkpoints, fold outputs and LR tuning remain under `data/experiments/`.
No experiment was rerun and no metric, run ID or historical registry content
was edited during this move. Historical flat CSVs and timestamped inference
exports remain distinct; moving them does not validate their scientific claims.

`relocation-20261013.json` records old/new repository-relative paths, sizes and
SHA-256 hashes for every moved file. Relative symlinks at the old locations
preserve historical references without keeping duplicate payloads. Keep those
links with the artifacts; removing them breaks paths embedded in old registries.
Notebook output configuration now uses this canonical directory.

The report document `SUPERVISOR_REPORT_2026-10-03.md` stays beside the source
notebooks; its legacy output links resolve through the compatibility links.
Existing log/checkpoint ignore rules still apply. These artifacts may contain
source-image previews and metadata: this is a local organization change, not
a public release. Only code is authorized for public release.

# Phase 1 reports

Thư mục này là đầu mối duy nhất cho toàn bộ báo cáo và bảng phụ trợ của Phase 1.

- `phase_1_summary_report.md`: báo cáo chuẩn, được inference lại từ 45 checkpoint Task A/Final-B.
- `phase_1_final_b_report.md`: bảng lịch sử từ output training; đã được thay thế do có metric không khớp checkpoint best.
- `task_b_wandb_all_runs_report-hotfix.md`: báo cáo các run Task B strict/origin-safe và phần backfill metric.
- `phase_1_task_b_strict_comparison/`: so sánh split cũ với strict-by-origin.
- `phase_1_leakage_origin_investigation/`: điều tra leakage và artifacts liên quan.
- `task_b_strict_origin_doc_tables/`: bảng CSV dùng cho tài liệu strict/origin.
- `phase_1_run_report_not_strict [DEPRECATED]/`: báo cáo pipeline cũ, chỉ giữ để truy vết lịch sử.

Notebook efficiency: `notebooks/phase 1/06_efficiency_flops_inference.ipynb`.
Notebook tổng hợp Task A/Final-B: `notebooks/phase 1/07_phase1_results_summary.ipynb`.
Artifacts efficiency: `data/experiments/phase_1/efficiency/`.
Artifacts tổng hợp: `data/experiments/phase_1/summary/`.

# Artifact của Người 1

Thư mục này nhận toàn bộ artifact hậu kỳ do
`WoodID/notebooks/person1_generate_artifacts.ipynb` tạo ra.

Notebook chỉ đọc checkpoint local tại:

`woodbench/data/experiments/Task-B-new/19-08-2026/`

Notebook không tải checkpoint, không huấn luyện lại và không dùng checkpoint
trong `Final-B` hoặc `task-B-17-old`.

Sau khi chạy đủ notebook, cấu trúc đầu ra là:

```text
artifact/person1/
├── gradcam/
│   ├── fig_gradcam.pdf
│   ├── fig_gradcam.png
│   ├── gradcam_manifest.csv
│   ├── gradcam_metadata.json
│   ├── sample_selection.csv
│   └── individual/
└── tsne/
    ├── features_convnext_seed42_test.npz
    ├── fig_tsne.pdf
    ├── fig_tsne.png
    ├── tsne_embedding.csv
    └── tsne_metadata.json
```

Ảnh Grad-CAM được chọn bằng quy tắc cố định từ prediction test, không chọn
thủ công theo độ đẹp của heatmap. t-SNE dùng đủ 532 ảnh test; `origin_key`
được nối bằng cách dựng lại đúng quy tắc đặt tên khi materialize group split,
và dùng MD5 làm fallback.

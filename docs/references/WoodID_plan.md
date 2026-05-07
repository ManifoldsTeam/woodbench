| # |
|---|
|PHÒNG THÍ NGHIỆM TÍNH TOÁN THÔNG MINH CHO PHÁT TRIỂN BỀN VỮNG (IC4SD)<br>Học viện Công nghệ Bưu chính Viễn thông (PTIT)<br>ĐỀ BÀI NGHIÊN CỨU<br>NHẬN DẠNG GỖ BẰNG DEEP LEARNING<br>Năm học 2025–2026|

| # | # |
|---|---|
|Supervisor|PGS.TS. Nguyễn Trọng Khánh|
|Email|khanhnt@ptit.edu.vn|
|Điện thoại|0912 314 482|
|Phòng thí nghiệm|IC4SD — Tính toán Thông minh cho Phát triển Bền vững, PTIT|
|Deadline đề cương|Tuần 2 sau khi nhận đề tài|
|Target submit|8-10/2026|

| # |
|---|
|LƯU Ý QUAN TRỌNG<br>Mỗi đề tài hướng tới xuất bản bài báo Q2–Q3 trên tạp chí Springer/Elsevier/IEEE, không thuộc nhà xuất bản MDPI. Dataset sử dụng là bộ sưu tập 260 loài gỗ thương mại nhập khẩu vào Việt Nam do phòng thí nghiệm IC4SD thu thập. SV/Học viên cần đọc toàn bộ tài liệu và chuẩn bị câu hỏi trước buổi gặp đầu tiên.|

<br>

| # |
|---|
|ĐỀ TÀI S2<br>A Systematic Benchmark of Deep Learning Architectures for Conifer Wood Species Identification Using Macroscopic Cross-Section Images<br>Đánh giá hệ thống 5 kiến trúc DL hiện đại trên 20 loài Cupressaceae–Pinaceae–Podocarpaceae–Taxaceae|

| # | # |
|---|---|
|Mã đề tài|WoodID-S2-2026|
|Nhóm thực hiện|2–3 sinh viên CNTT năm cuối|
|Thời gian|5–6 tháng (bắt đầu 4/2026, target submit 9-10/2026)|
|Dataset|20 loài Conifer | 7,050 ảnh macro 50× | 4 họ thực vật|
|Target journal|Signal, Image and Video Processing (Springer, Q2) hoặc IET Image Processing (Wiley, Q2)|

## 1. Giới thiệu & Bối cảnh

Gỗ lá kim (conifer/softwood) từ các họ Cupressaceae và Pinaceae chiếm tỷ trọng lớn trong thương mại gỗ toàn cầu và nhập khẩu vào Việt Nam. Đặc biệt, dataset này chứa 7 loài đặc hữu Việt Nam (Fokienia hodginsii, Xanthocyparis vietnamensis, Calocedrus rupestris, Pinus krempfii, Pinus dalatensis, Pseudotsuga brevifolia, Pinus cernua) và Taxus wallichiana thuộc CITES Appendix II.

Đây là bài báo benchmark — contribution chính không phải là đề xuất kiến trúc mới mà là: (1) cung cấp dataset chuẩn hóa cho nhóm conifer, (2) so sánh có hệ thống 5 SOTA architectures trên 2 task phân cấp (family-level và species-level), (3) phân tích trade-off accuracy vs. efficiency cho từng ứng dụng thực tế.

## 2. Mục tiêu nghiên cứu

Xây dựng benchmark dataset: 20 loài conifer, 7,050 ảnh macro 50×

Task A — Family-level classification: phân loại 4 họ (Cupressaceae / Pinaceae / Podocarpaceae / Taxaceae)

Task B — Species-level classification: phân loại 20 loài

Benchmark 5 architectures: ResNet-50, EfficientNet-B3, MobileNetV3-Large, ConvNeXt-Tiny, Swin-T

Phân tích efficiency: inference time, model size, FLOPs — cho deployment thực tế

Nghiên cứu augmentation đặc thù: ảnh mặt cắt ngang đẳng hướng (isotropic) → rotation 90° có giúp ích không?

## 3. Dataset
### 3.1 Tổng quan

Dataset S2 gồm 20 loài thuộc 4 họ conifer, tổng 7,050 ảnh macro 50°. Lưu ý: có 2 loài sp. (Calocedrus sp., Pinus sp.) cần quyết định xử lý label. 7/20 loài là đặc hữu hoặc quý hiếm Việt Nam — đây là điểm novelty bổ sung khi framing paper theo hướng conservation.

### 3.2 Danh sách 20 loài

| # | # | # | # | # | # |
|---|---|---|---|---|---|
|#|Họ (Family)|Tên khoa học|Tên VN / Thương mại|Ảnh|Ghi chú|
|1|Cupressaceae|Cunninghamia lanceolata|Sa mu (China fir)|556|Nhiều ảnh nhất|
|2|Cupressaceae|Callitris columellaris|White cypress / Bách dầu|517||
|3|Cupressaceae|Fokienia hodginsii|Pơ mu|518|⭐ Endemic VN, CITES|
|4|Cupressaceae|Cupressus funebris|Sa mu dầu / Ngọc am|420||
|5|Cupressaceae|Calocedrus sp.|Cedar|342|⚠ sp. level|
|6|Cupressaceae|Cryptomeria japonica|Thông nhật / Tuyết tùng|335||
|7|Cupressaceae|Xanthocyparis vietnamensis|Bách vàng|318|⭐ Endemic VN|
|8|Cupressaceae|Chamaecyparis obtusa|Gỗ Hinoki|317||
|9|Cupressaceae|Calocedrus rupestris|Bách xanh núi đá|302|⭐ Endemic VN|
|10|Cupressaceae|Thujopsis dolabrata|Tuyết tùng Hiba|327||
|11|Pinaceae|Pinus dalatensis|Thông Đà Lạt|334|⭐ Endemic VN|
|12|Pinaceae|Pinus radiata|Thông vàng Brazil|332||
|13|Pinaceae|Pinus sp.|Thông|324|⚠ sp. level|
|14|Pinaceae|Pinus krempfii|Thông 3 lá dẹt|323|⭐ Endemic VN|
|15|Pinaceae|Picea lauca|Vân sam|304||
|16|Pinaceae|Araucaria araucana|Thông Chile|300||
|17|Pinaceae|Pseudotsuga brevifolia|Thiết sam giả lá ngắn|289|⭐ Endemic VN|
|18|Pinaceae|Pinus cernua|Thông xuân nha 5 lá|287|⭐ Endemic VN|
|19|Podocarpaceae|Podocarpus pilgeri|Thông tre lá ngắn|310||
|20|Taxaceae|Taxus wallichiana|Thông đỏ lá dài|295|⭐ CITES App. II|

### 3.3 Phân chia dữ liệu & Cấu trúc 2 cấp

Cấu trúc folder 2 cấp: data/{split}/{family}/{species}/ — cho phép chạy cả Task A (đọc tên folder family) và Task B (đọc tên folder species) từ cùng một dataset mà không cần copy.

Split: 70/15/15, stratified theo species, seed = 42. Tổng: ~4,935 train | ~1,057 val | ~1,057 test.

## 4. Paper Outline

| # | # |
|---|---|
|Section|Nội dung|
|1. Introduction|Tầm quan trọng conifer; nhiều loài endemic VN; gap: chưa có benchmark DL; 3 contributions. (~500–700 từ)|
|2. Related Work|(2.1) Wood anatomy đặc trưng conifer; (2.2) Wood ID với DL; (2.3) Image classification benchmarks; bảng so sánh. (~400–600 từ)|
|3. Materials & Methods|(3.1) Dataset; (3.2) Task A & B definition; (3.3) Preprocessing & split; (3.4) Augmentation (đặc biệt 90° rotation); (3.5) 5 model architectures; (3.6) Training; (3.7) Metrics + efficiency. (~700–900 từ)|
|4. Results|(4.1) Task A results; (4.2) Task B results; (4.3) Efficiency comparison table; (4.4) Confusion matrix Task B; (4.5) Ablation augmentation; (4.6) Grad-CAM; (4.7) Learning curves. (~900–1100 từ)|
|5. Discussion|Lightweight vs accuracy; Cupressaceae intra-family confusion; 90° rotation result; endemic VN species performance; limitations. (~400–600 từ)|
|6. Conclusion|Tóm tắt; best model cho từng use case; ý nghĩa bảo tồn; hướng tiếp. (~150–250 từ)|

## 5. Điểm khác biệt quan trọng 

| # |
|---|
|S2 có 2 Task (Task A: family, Task B: species) — phải train và report kết quả cho CẢ HAI task.<br>S2 phải đo và so sánh EFFICIENCY (inference time ms/image, model size MB, FLOPs) của 5 models.<br>S2 có ablation đặc thù: 90° rotation augmentation. Hypothesis: ảnh mặt cắt ngang không có "hướng tự nhiên" (isotropic) → rotation 90/180/270° không mất thông tin → nên giúp tăng accuracy. Cần verify bằng experiment.|

## 6. Experiment Checklist
### Phase 0 — Chuẩn bị (Tuần 1–2)

| # | # | # | # |
|---|---|---|---|
|✓|Nhiệm vụ|Phase|Deadline|
|□|Tổ chức folder 2 cấp: data/{split}/{family}/{species}/|Phase 0|Tuần 1|
|□|Tạo label mapping JSON: species_name → family_name|Phase 0|Tuần 1|
|□|Quyết định xử lý Calocedrus sp. và Pinus sp.|Phase 0|Tuần 1|
|□|Stratified split 70/15/15 theo species, seed=42|Phase 0|Tuần 1|
|□|Visualize 4 ảnh mỗi loài, sắp xếp theo family|Phase 0|Tuần 2|
|□|Setup môi trường, test GPU, profile inference time baseline|Phase 0|Tuần 2|

### Phase 1 — Experiments chính (Tuần 3–8)

| # | # | # | # |
|---|---|---|---|
|✓|Nhiệm vụ|Phase|Deadline|
|□|Train 5 models cho Task A (4-class): ResNet-50, EfficientNet-B3, MobileNetV3-L, ConvNeXt-T, Swin-T|Phase 1|Tuần 3-4|
|□|Train 5 models cho Task B (20-class) — cùng config với Task A|Phase 1|Tuần 4-6|
|□|Đo inference time (ms/image, batch=1, 100 runs avg) và model size (MB) cho 5 models|Phase 1|Tuần 6|
|□|Tính FLOPs cho 5 models (dùng thư viện ptflops hoặc fvcore)|Phase 1|Tuần 6|
|□|Ablation: 90° rotation aug có vs không có — best model Task B|Phase 1|Tuần 7|
|□|Ablation: label smoothing 0.0 vs 0.1 vs 0.2|Phase 1|Tuần 7-8|
|□|Chạy 3 seeds cho mỗi experiment, report mean ± std|Phase 1|Ongoing|

### Phase 2 — Analysis & Visualization (Tuần 9–12)

| # | # | # | # |
|---|---|---|---|
|✓|Nhiệm vụ|Phase|Deadline|
|□|Confusion matrix 20×20 cho Task B best model — highlight Cupressaceae intra-family errors|Phase 2|Tuần 9|
|□|Per-species accuracy bar chart, sắp xếp theo family|Phase 2|Tuần 9|
|□|Radar chart 5 models × 4 metrics (accuracy, F1, speed, size)|Phase 2|Tuần 10|
|□|Learning curves (val_accuracy vs epoch) Task A & B — 5 models|Phase 2|Tuần 10|
|□|Grad-CAM: 1 ảnh/family × 3 best models — so sánh activation patterns|Phase 2|Tuần 11|
|□|Efficiency summary table: Accuracy / F1 / Time(ms) / Size(MB) / FLOPs|Phase 2|Tuần 12|

### Phase 3 — Viết paper (Tuần 13–20)

| # | # | # | # |
|---|---|---|---|
|✓|Nhiệm vụ|Phase|Deadline|
|□|Viết Section 3 (Methods & Dataset) ngay từ Phase 1|Phase 3|Tuần 13|
|□|Chuẩn hóa tất cả tables & figures (font, 300 DPI, caption)|Phase 3|Tuần 14|
|□|Viết Related Work — ≥25 references|Phase 3|Tuần 15|
|□|Viết Results + Discussion + Conclusion|Phase 3|Tuần 16-17|
|□|Supervisor review → chỉnh sửa → format journal → submit|Phase 3|Tuần 18-20|

## 
## 7. Báo cáo ĐATN dự kiến

| # | # | # | # | # |
|---|---|---|---|---|
|CHƯƠNG 1||Giới thiệu<br>Bối cảnh · Mục tiêu · Phạm vi · Đóng góp||8–10 trang|
|1.1|Đặt vấn đề||Tầm quan trọng gỗ lá kim trong thương mại nhập khẩu VN; hạn chế kiểm tra thủ công; nhu cầu AI hỗ trợ nhận dạng loài tại cửa khẩu. Nêu số liệu thực tế: VN nhập khẩu bao nhiêu m³ gỗ lá kim/năm.||
|1.2|Mục tiêu nghiên cứu||Liệt kê 4–5 mục tiêu đo đếm được: (1) xây dựng dataset IC4SD-Wood-Conifer20v1; (2) benchmark 5 mô hình DL trên 2 tasks; (3) phân tích efficiency; (4) giải thích Grad-CAM; (5) xây dựng ứng dụng thử nghiệm.<br>[Phải có]||
|1.3|Phạm vi và giới hạn||20 loài Conifer nhập khẩu VN; chỉ dùng ảnh macro 50×; không xét ảnh thực địa; ứng dụng demo (chưa production). Ghi rõ giới hạn để hội đồng không hỏi ngoài phạm vi.||
|1.4|Đóng góp của đề tài||3 đóng góp chính: (1) Dataset 20 loài Conifer có loài endemic VN; (2) Benchmark 5 SOTA models với 2 tasks và efficiency analysis; (3) Ứng dụng demo nhận dạng gỗ qua camera/ảnh.<br>[Phải có]||
|1.5|Bố cục báo cáo||Mô tả ngắn nội dung 6 chương (4–5 câu mỗi chương).||

| # | # | # | # | # |
|---|---|---|---|---|
|CHƯƠNG 2||Cơ sở lý thuyết và tổng quan nghiên cứu<br>Giải phẫu gỗ lá kim · DL · Transfer learning · Related work||15–20 trang|
|2.1|Giải phẫu gỗ lá kim||Đặc điểm mặt cắt ngang Cupressaceae, Pinaceae, Podocarpaceae, Taxaceae; resin canal, tracheid, ray cell; tại sao khó phân biệt bằng mắt. Hình ảnh minh họa 4 họ đại diện.<br>[Hình 2.1–2.2]||
|2.2|Học sâu cho phân loại ảnh||CNN cơ bản (conv, pooling, FC); transfer learning và fine-tuning; ImageNet pretraining; label smoothing.||
|2.3|Các kiến trúc DL sử dụng||ResNet-50 (residual connection), EfficientNet-B3 (compound scaling), MobileNetV3-Large (depthwise separable), ConvNeXt-Tiny (modernized CNN), Swin-T (window attention). Bảng so sánh Params/FLOPs/ImageNet accuracy.<br>[Bảng 2.1]||
|2.4|Grad-CAM và khả năng giải thích||Nguyên lý Grad-CAM; gradient-weighted activation map; ý nghĩa trong ứng dụng giám định gỗ; cách chọn target layer cho từng backbone.||
|2.5|Tổng quan nghiên cứu liên quan||Nhận dạng loài gỗ với DL (2019–2025); bài dùng ảnh cross-section; benchmark DL cho domain cụ thể. Kết thúc bằng phân tích khoảng trống: chưa có benchmark Conifer cross-section macro.<br>[Bảng 2.2 Related work]||
|2.6|Các framework và thư viện sử dụng||PyTorch 2.x, timm, torchvision, pytorch-grad-cam, scikit-learn, Streamlit/Gradio (cho ứng dụng chương 6). Giới thiệu ngắn từng thư viện.||

| # | # | # | # | # |
|---|---|---|---|---|
|CHƯƠNG 3||Dữ liệu và phương pháp nghiên cứu<br>Dataset · 2 tasks · Preprocessing · Models · Config · Metrics||20–28 trang|
|3.1|Bộ dữ liệu IC4SD-Wood-Conifer20v1||Mô tả 20 loài, 4 họ, 7,050 ảnh; quy trình thu thập mẫu và chụp ảnh 50×; 7 loài endemic VN; thông số kỹ thuật. Bảng danh sách 20 loài: STT | Tên KH | Tên VN | Họ | n_train | n_val | n_test. Figure grid 20 ảnh mẫu.<br>[Bảng 3.1] [Hình 3.1]||
|3.2|Định nghĩa 2 tasks phân loại||Task A: phân loại họ (4 class) — baseline validate pipeline; Task B: phân loại loài (20 class) — bài toán chính. Cấu trúc folder 2 cấp data/{split}/{family}/{species}/. Giải thích tại sao cần 2 tasks.<br>[Phải có]||
|3.3|Tiền xử lý và tăng cường dữ liệu||Split 70/15/15 stratified seed=42; Resize 224×224; augmentation train: HFlip, VFlip, Rotation(30°), ColorJitter; normalize ImageNet. GIẢI THÍCH: ảnh cross-section đẳng hướng (isotropic) nên VFlip và Rotation hợp lệ — đây là điểm kỹ thuật quan trọng.<br>[Bảng 3.2 aug config]||
|3.4|Pipeline và cấu hình huấn luyện||ImageFolder → DataLoader → Backbone → Linear head(N_class). Hyperparams: AdamW lr=1e-4, CosineAnnealingLR T_max=50, epochs=50, batch=64, AMP, label_smoothing=0.1. Sơ đồ pipeline tổng thể.<br>[Hình 3.2 pipeline] [Bảng 3.3 hyperparams]||
|3.5|Thiết kế đo lường hiệu suất||Accuracy, Precision, Recall, F1-macro (sklearn); Confusion matrix; Per-class F1; Inference time ms/image (100 runs, batch=1); Model size MB; FLOPs (ptflops). Đây là contribution phân biệt S2 với S1.<br>[Phải có đủ]||
|3.6|Thiết kế ablation study||3 ablation: (A1) 90° rotation on/off — hypothesis isotropic; (A2) label smoothing 0.0 vs 0.1 vs 0.2; (A3) image size 224 vs 384px. Giải thích lý do chọn 3 ablation này.||

| # | # | # | # | # |
|---|---|---|---|---|
|CHƯƠNG 4||Thực nghiệm và kết quả<br>Task A · Task B · Efficiency · Ablation · Grad-CAM · t-SNE · Learning curves||20–30 trang|
|4.1|Môi trường thực nghiệm||GPU; Python 3.10, PyTorch 2.x, CUDA 12.x; thư viện; thời gian huấn luyện ước tính mỗi model. Bảng cấu hình thực nghiệm.<br>[Bảng 4.1]||
|4.2|Kết quả Task A — Phân loại họ||Bảng 5 models × 4 metrics (Acc, Prec, Recall, F1); dự kiến accuracy >97%; model nào đạt đỉnh nhanh nhất; phân tích ngắn.<br>[Bảng 4.2]||
|4.3|Kết quả Task B — Phân loại loài||Bảng 5 models × 4 metrics; per-class F1; loài khó nhất (Cupressaceae intra-family); cặp nhầm nhiều nhất. Confusion matrix 20×20 và per-class F1 bar chart.<br>[Bảng 4.3] [Hình 4.1 CM] [Hình 4.2 F1 chart]||
|4.4|Phân tích hiệu suất tính toán (Efficiency)||Bảng Params(M) / FLOPs(G) / Inference(ms) / Size(MB) / VRAM(GB). Radar chart 5 models × 4 tiêu chí. Khuyến nghị: model nào cho accuracy-first, model nào cho edge deployment.<br>[Bảng 4.4 efficiency] [Hình 4.3 radar] [Phải có]||
|4.5|Kết quả ablation study||3 bảng ablation (A1 rotation / A2 label smoothing / A3 resolution). Phân tích từng kết quả; kết luận về cấu hình tối ưu.<br>[Bảng 4.5 a-c]||
|4.6|Grad-CAM và giải thích mô hình||Grid Grad-CAM: 4 họ × 3 ảnh (original / overlay / heatmap). So sánh activation giữa Task A và Task B trên cùng ảnh. Model học resin canal hay tracheid?<br>[Hình 4.4 Grad-CAM]||
|4.7|Visualize không gian đặc trưng||t-SNE 2D: color = họ (Task A) và loài (Task B). Phân tích cluster separation. Cupressaceae cluster có tách rời không?<br>[Hình 4.5 t-SNE]||
|4.8|Learning curves||Val accuracy vs epoch cho 5 models (Task A + Task B). Convergence speed. Model nào ổn định nhất?<br>[Hình 4.6 curves]||

| # | # | # |
|---|---|---|
|CHƯƠNG 5|Ứng dụng nhận dạng gỗ lá kim<br>Thiết kế app · Demo · Đánh giá thực tế|10–15 trang|
|Chương này là điểm nhấn thực tiễn — hội đồng ĐATN rất thích thấy sản phẩm chạy được. Ứng dụng không cần phức tạp, chỉ cần demo được: upload ảnh → model predict → hiển thị kết quả + Grad-CAM.|||

| # | # | # |
|---|---|---|
|5.1|Yêu cầu và thiết kế hệ thống|Yêu cầu chức năng: upload ảnh/chụp camera, chọn model, hiển thị top-3 predictions, confidence score, Grad-CAM overlay, thông tin loài. Yêu cầu phi chức năng: latency <2s, chạy được trên laptop CPU.<br>[Hình 5.1 use case]|
|5.2|Kiến trúc ứng dụng|Backend: Python + FastAPI (hoặc Flask) load model PyTorch, endpoint /predict nhận ảnh base64, trả JSON (class, confidence, gradcam). Frontend: Streamlit hoặc Gradio — đơn giản, nhanh. Sơ đồ kiến trúc client-server.<br>[Hình 5.2 kiến trúc] [Bảng 5.1 tech stack]|
|5.3|Cài đặt và tối ưu model cho deployment|Load best model checkpoint; model.eval() + torch.no_grad(); TorchScript export (optional); xử lý ảnh đầu vào (resize, normalize) nhất quán với training; Grad-CAM inference. So sánh inference time CPU vs GPU.<br>[Bảng 5.2 inference time]|
|5.4|Giao diện người dùng|Demo màn hình chính: upload zone, kết quả prediction với progress bar confidence, overlay Grad-CAM, thông tin loài (tên KH, tên VN, họ, đặc điểm nhận biết ngắn). Screenshot giao diện.<br>[Hình 5.3–5.5 screenshots]|
|5.5|Kiểm thử và đánh giá ứng dụng|Kiểm thử chức năng: test 20 ảnh test set qua UI — kết quả có khớp model eval không? Kiểm thử performance: đo latency end-to-end (upload → response). Kiểm thử edge cases: ảnh mờ, sai loại, ảnh không phải gỗ.<br>[Bảng 5.3 test cases]|
|5.6|Hướng phát triển ứng dụng|Mobile app (Flutter/React Native + ONNX export); tích hợp camera real-time; multi-language (VN/EN); lưu lịch sử tra cứu; kết nối cơ sở dữ liệu loài gỗ đầy đủ hơn.|

| # | # | # | # | # |
|---|---|---|---|---|
|CHƯƠNG 6||Thảo luận và kết luận<br>Phân tích kết quả · Hạn chế · Hướng phát triển||10–15 trang|
|6.1|Thảo luận kết quả||Tại sao model X tốt hơn? MobileNetV3 vs ConvNeXt trade-off. Cupressaceae khó nhất — vì sao? 90° rotation có giúp không và tại sao? Loài endemic VN (Fokienia, Xanthocyparis) được nhận dạng tốt không?||
|6.2|So sánh với nghiên cứu liên quan||Đặt kết quả vào bối cảnh literature: accuracy so với các bài wood ID cùng số loài; efficiency so với benchmark DL tổng quát.||
|6.3|Hạn chế của nghiên cứu||Dataset giới hạn 20 loài; chưa test ngoài thực địa; sp.-level label chưa chắc chắn; ứng dụng còn ở mức demo.||
|6.4|Kết luận||Tóm tắt 3 đóng góp chính; best model cho accuracy; best model cho edge deployment; ý nghĩa thực tiễn cho kiểm soát gỗ nhập khẩu VN.||
|6.5|Hướng phát triển||Mở rộng thêm loài; microscopic imaging; mobile deployment; multimodal fusion.||

## 
## 8. Tài liệu tham khảo bắt buộc đọc trước
### Về wood identification với deep learning:

Ravindran et al. (2020). XyloTron: Field-deployable instrument for wood ID. PLOS ONE.

Lens et al. (2020). Computer-assisted timber identification. Plant Methods.

Filho et al. (2022). Wood species classification from cross-section images. ESWA.

### Về các backbone sẽ sử dụng:

He et al. (2016). Deep Residual Learning. CVPR. [ResNet]

Tan & Le (2019). EfficientNet. ICML. [EfficientNet]

Liu et al. (2022). A ConvNet for the 2020s. CVPR. [ConvNeXt]

Liu et al. (2021). Swin Transformer. ICCV. [Swin-T]

Dosovitskiy et al. (2021). An Image is Worth 16x16 Words. ICLR. [ViT]

| # |
|---|
|Thông tin liên hệ Supervisor:<br>PGS.TS. Nguyễn Trọng Khánh | khanhnt@ptit.edu.vn | 0912 314 482<br>Phòng thí nghiệm Tính toán Thông minh cho Phát triển Bền vững (IC4SD) — Học viện CNBCVT (PTIT)|
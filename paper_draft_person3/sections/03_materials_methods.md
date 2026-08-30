# 3. Materials and Methods

## 3.1. Dataset Acquisition and Origin-Safe Partition

Tập dữ liệu IC4SD-Wood-Conifer20v1 gồm ảnh macro mặt cắt ngang gỗ lá kim, chụp ở độ phóng đại 50× với độ phân giải cố định 2560×2048 pixel, thu thập tại Phòng thí nghiệm Tính toán Thông minh cho Phát triển Bền vững (IC4SD), Học viện Công nghệ Bưu chính Viễn thông. Thiết kế ban đầu dự kiến 7.050 ảnh của 20 loài thuộc 4 họ. Số ảnh thu thập thực tế là **6.423**, thấp hơn thiết kế 627 ảnh, với chênh lệch tập trung ở sáu loài. Kiểm tra định dạng ghi nhận 0 ảnh hỏng. [CẦN DỮ LIỆU: nguyên nhân chênh lệch 627 ảnh so với thiết kế]

**Cấu trúc mẫu gốc.** Tập dữ liệu có bản chất crop: nhiều ảnh được cắt từ cùng một khúc gỗ. Trong 5.830 ảnh có định danh mẫu gốc, số mẫu gốc độc lập chỉ là 557, tương ứng trung bình 10,47 ảnh mỗi mẫu, và 88,3% số ảnh thuộc các mẫu có từ 24 ảnh trở lên. Đặc điểm này quyết định toàn bộ chiến lược phân chia dữ liệu.

**Vì sao phân chia theo từng ảnh không hợp lệ.** Phép phân chia phân tầng theo loài với tỷ lệ 70/15/15 và seed 42, thao tác ở cấp từng ảnh, tạo ra rò rỉ nghiêm trọng: 97,75% ảnh validation và 97,62% ảnh test có mẫu gốc đã xuất hiện trong tập huấn luyện. Do các ảnh crop từ cùng một khúc gỗ chia sẻ kết cấu, điều kiện chiếu sáng và hướng cắt, mô hình có thể nhận lại mẫu đã thấy thay vì khái quát hóa sang khúc gỗ mới. Việc lọc trùng theo mã băm MD5 không khắc phục được vấn đề này, vì hai crop khác nhau từ cùng một mẫu có mã băm khác nhau.

**Phân chia theo mẫu gốc.** Task B do đó được phân chia lại theo nhóm, lấy mẫu gốc làm đơn vị chia, đảm bảo mọi ảnh cắt từ cùng một khúc gỗ chỉ nằm trong một tập duy nhất. Ràng buộc này bộc lộ ba lớp không thể phân chia hợp lệ, trình bày ở Bảng 1.

**Bảng 1.** Ba lớp bị loại khỏi Task B và số mẫu gốc độc lập của chúng.

| Lớp | Số mẫu gốc | Ảnh train | Ảnh val | Ảnh test |
|---|---:|---:|---:|---:|
| *Calocedrus* sp. | 1 | 238 | 0 | 0 |
| *Pinus* sp. | 2 | 110 | 0 | 106 |
| *Cupressus funebris* (*Cupressus pendula*) | 2 | 163 | 0 | 155 |

Một lớp chỉ có một mẫu gốc không thể đồng thời xuất hiện ở cả ba tập, và với hai mẫu gốc thì luôn thiếu ít nhất một tập. Nguyên nhân loại bỏ là **thiếu mẫu gốc độc lập, không phải thiếu ảnh**: *Calocedrus* sp. có 238 ảnh huấn luyện nhưng không có ảnh validation hay test nào. Hai lớp đầu vốn chỉ được định danh tới cấp chi và đã được ghi nhận là nhãn cần xử lý ngay từ giai đoạn thiết kế. Sau khi loại ba lớp, tập Task B gồm **17 loài** và hoàn toàn an toàn về mặt mẫu gốc. Cả 7 loài đặc hữu Việt Nam và 1 loài thuộc CITES Phụ lục II đều được giữ lại. Phiên bản phân chia được định danh bằng mã băm `task-B-group-split:359be7f5379818d4`, tính theo `sha256(đường dẫn ảnh + sha256 nội dung ảnh)`.

## 3.2. Hierarchical Task Formulation

Bài toán được tách thành hai cấp phân loại trên cùng một cấu trúc thư mục hai cấp `{split}/{họ}/{loài}`, cho phép sinh nhãn ở cả hai cấp mà không nhân bản dữ liệu.

**Task A** phân loại 4 họ (Cupressaceae, Pinaceae, Podocarpaceae, Taxaceae), đóng vai trò kiểm chứng quy trình xử lý. Do phân bố họ mất cân bằng nghiêm trọng, tập huấn luyện Task A được cân bằng bằng tăng cường ngoại tuyến, bổ sung 2.777 ảnh cho hai họ thiểu số Podocarpaceae và Taxaceae. Tập validation và test không chứa ảnh sinh thêm. Task A sử dụng phân chia phân tầng theo loài, **không phải** phân chia theo mẫu gốc, nên kết quả Task A chỉ nên đọc như kiểm chứng quy trình.

**Task B** phân loại 17 loài trên tập đã phân chia theo mẫu gốc, và là bài toán chính của nghiên cứu.

**Bảng 2.** Phân chia dữ liệu của hai bài toán. Tập huấn luyện Task A đã cân bằng bằng tăng cường ngoại tuyến.

| Bài toán | Số lớp | Train | Validation | Test | Chiến lược phân chia |
|---|---:|---:|---:|---:|---|
| Task A (họ) | 4 | 7.264 | 955 | 981 | Phân tầng theo loài, seed 42 |
| Task B (loài) | 17 | 3.365 | 903 | 532 | Theo mẫu gốc |

## 3.3. Data Augmentation and the Isotropic Rotation Hypothesis

Ảnh đầu vào được đưa về 224×224 bằng phép thay đổi kích thước trực tiếp, không cắt giữa, rồi chuẩn hóa theo trung bình `[0,485; 0,456; 0,406]` và độ lệch chuẩn `[0,229; 0,224; 0,225]` của ImageNet. Chuỗi tăng cường trực tuyến áp dụng cho tập huấn luyện theo thứ tự: thay đổi kích thước, lật ngang với xác suất 0,5, lật dọc với xác suất 0,5, phép xoay, biến đổi độ sáng và độ tương phản với biên độ 0,2, chuyển sang tensor, và chuẩn hóa. Tập validation và test chỉ qua ba bước cuối.

**Giả thuyết đẳng hướng.** Ảnh mặt cắt ngang ngang thớ không có hướng chuẩn về mặt sinh học: quản bào tỏa theo hàng xuyên tâm quanh tâm sinh trưởng, tia gỗ chạy theo mọi phương xuyên tâm, và ống dẫn nhựa không phân bố theo trục ưu tiên nào. Một mẫu xoay đi 90°, 180° hay 270° vẫn là mẫu hợp lệ, khác với ảnh tự nhiên vốn có phương trọng lực xác định hướng. Từ nhận xét này, nghiên cứu đề xuất thay phép xoay liên tục bằng phép **xoay rời rạc**, chọn ngẫu nhiên đều một trong bốn góc 0°, 90°, 180° và 270°.

Phép xoay bội số 90° còn có ưu điểm hình học là bảo toàn toàn bộ pixel, trong khi phép xoay một góc bất kỳ trên ảnh vuông tạo bốn vùng trống ở góc. Hai cấu hình xoay loại trừ lẫn nhau trong cài đặt: cấu hình mặc định dùng xoay liên tục trong khoảng ±30°, cấu hình khảo sát dùng xoay rời rạc. Phép so sánh ở Mục 4.5 vì vậy là giữa **hai kiểu xoay**, không phải giữa có xoay và không xoay.

Riêng Task A dùng thêm tăng cường ngoại tuyến để cân bằng lớp, gồm bảy phép biến đổi trên ảnh gốc 2560×2048 (lật ngang, lật dọc, xoay 180°, xoay 270°, cắt và thay đổi kích thước, biến đổi màu, chuyển vị), mỗi ảnh sinh thêm chỉ nhận một phép, chọn luân phiên. Phép xoay 90° cố ý không nằm trong nhóm này để dành riêng cho khảo sát trực tuyến.

## 3.4. Evaluated Architectures and Training Protocol

Năm kiến trúc được khởi tạo từ thư viện timm với trọng số tiền huấn luyện ImageNet. Lớp phân loại gốc được thay bằng một lớp tuyến tính có số đầu ra bằng số lớp của bài toán, và toàn bộ tham số đều được tinh chỉnh.

**Bảng 3.** Năm kiến trúc được đánh giá. Số tham số và GFLOPs đo với đầu vào 1×3×224×224 và 17 lớp đầu ra.

| Kiến trúc | Định danh timm | Nhóm thiết kế | Params (M) | GFLOPs | Batch |
|---|---|---|---:|---:|---:|
| ResNet-50 | `resnet50` | Tích chập dư | 23,54 | 8,263 | 64 |
| EfficientNet-B3 | `tf_efficientnet_b3` | Tỷ lệ hợp thành | 10,72 | 1,877 | 32 |
| MobileNetV3-Large | `mobilenetv3_large_100` | Mạng nhẹ | 4,22 | 0,431 | 64 |
| ConvNeXt-Tiny | `convnext_tiny` | Tích chập hiện đại | 27,83 | 8,910 | 32 |
| Swin-Tiny | `swin_tiny_patch4_window7_224` | Transformer phân cấp | 27,53 | 8,742 | 32 |

**Giao thức huấn luyện** thống nhất cho cả năm kiến trúc, trừ kích thước lô vốn bị giới hạn bởi bộ nhớ GPU. Bộ tối ưu là AdamW với suy giảm trọng số 1×10⁻⁴. Tốc độ học phân tầng: 1×10⁻⁴ cho lớp phân loại và 1×10⁻⁵ cho phần trích đặc trưng, nhằm bảo toàn đặc trưng kết cấu bậc thấp đã học từ ImageNet. Lịch trình tốc độ học là cosine annealing kèm 5 epoch khởi động tuyến tính. Mỗi lượt chạy kéo dài 50 epoch với huấn luyện chính xác hỗn hợp FP16, dừng sớm khi accuracy trên tập validation không cải thiện trong 10 epoch, và checkpoint được lưu theo accuracy validation cao nhất. Mỗi cấu hình chính chạy với ba seed 42, 123 và 456.

**Hàm mất mát** là entropy chéo có làm mượt nhãn với hệ số $\epsilon = 0{,}1$:

$$\mathcal{L}_{LS}(y, p) = (1 - \epsilon)\,\mathcal{L}_{CE}(y, p) + \frac{\epsilon}{K} \sum_{k=1}^{K} -\log p_k$$

trong đó $K$ là số lớp, $p_k$ là xác suất softmax của lớp $k$, và $\mathcal{L}_{CE}(y,p) = -\log p_y$ là entropy chéo chuẩn với nhãn thật $y$. Số hạng thứ hai phân bố đều một phần khối lượng xác suất sang toàn bộ các lớp, làm giảm mức tự tin thái quá của mô hình.

## 3.5. Evaluation Metrics and Efficiency Profiling

**Chỉ số phân loại.** Bốn chỉ số được báo cáo: Top-1 Accuracy, Macro Precision, Macro Recall và Macro-F1, kèm F1 của từng lớp. Trung bình theo lớp được chọn thay cho trung bình theo mẫu vì phân bố lớp không cân bằng, khiến trung bình theo mẫu bị chi phối bởi các lớp đông. Giá trị trung bình và độ lệch chuẩn tính trên ba seed với độ lệch chuẩn tổng thể. Khác biệt giữa các kiến trúc được kiểm định bằng t ghép cặp theo seed.

**Đo tài nguyên tính toán.** Ba chỉ số được đo cho cả năm kiến trúc trên **cùng một thiết bị NVIDIA GeForce RTX 3060** để so sánh công bằng.

Số phép tính đếm bằng thư viện THOP. Vì THOP trả về số phép nhân cộng dồn (MAC), giá trị FLOPs được quy đổi theo quy ước **1 MAC = 2 FLOPs**, tương ứng một phép nhân và một phép cộng.

Dung lượng mô hình đo trên state dict đã serialize, không bao gồm trạng thái bộ tối ưu và bộ lập lịch. Cách đo này cho con số phản ánh dung lượng triển khai thực tế, nhỏ hơn khoảng ba lần so với tệp checkpoint huấn luyện đầy đủ.

Độ trễ suy luận đo với kích thước lô bằng 1 và đầu vào 1×3×224×224, gồm 20 lần chạy làm nóng không tính vào trung bình, tiếp theo là 100 lần đo. Phép đo dùng `time.perf_counter_ns` kết hợp đồng bộ thiết bị trước và sau mỗi lần đo, thay cho CUDA Events. [GHI CHÚ KỸ THUẬT: mô tả này phản ánh đúng mã đã sinh ra số liệu ở Bảng 4 (Mục 4.3). Cài đặt dựa trên CUDA Events có tồn tại trong thư viện nhưng không được dùng cho phép đo chính thức.] Ngoài giá trị trung bình và độ lệch chuẩn, phân vị 95 cũng được ghi nhận để phản ánh độ ổn định của độ trễ.

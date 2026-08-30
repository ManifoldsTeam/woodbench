# 2. Related Work

> **Trạng thái bản thảo.** Kho mã của dự án hiện chỉ chứa **8 tài liệu tham khảo ở dạng rút gọn** (tác giả, năm, tựa, hội nghị hoặc tạp chí), liệt kê trong `docs/references/WoodID_plan.md` dòng 227 đến 243 và lặp lại nguyên văn trong `WoodID_Assignment_S2.docx`. Không tài liệu nào kèm DOI, số trang, hay số liệu kết quả. Đề bài yêu cầu Related Work có **tối thiểu 25 tài liệu tham khảo**. Mọi phát biểu chưa truy được về một nguồn cụ thể đều được đánh dấu `[CẦN NGUỒN: ...]` để bổ sung trước khi nộp. Không tài liệu, số liệu, hay kết quả nào trong mục này được suy đoán.

## 2.1. Wood Anatomical Characteristics of Conifers

Gỗ lá kim, còn gọi là softwood, có cấu tạo giải phẫu đơn giản hơn đáng kể so với gỗ lá rộng. Khác biệt cơ bản nằm ở chỗ gỗ lá kim **không có mạch gỗ (vessels)**, thành phần dẫn nước đặc trưng của hardwood. Chức năng dẫn nước và nâng đỡ cơ học ở lá kim do một loại tế bào duy nhất đảm nhiệm là **quản bào (tracheids)**, chiếm phần lớn thể tích gỗ và xếp thành hàng xuyên tâm đều đặn. [CẦN NGUỒN: tài liệu giải phẫu gỗ mô tả cấu tạo softwood, ví dụ danh mục đặc trưng IAWA cho softwood identification]

Hai thành phần còn lại mang giá trị phân loại cao. **Tia gỗ (ray parenchyma)** chạy theo phương xuyên tâm, khác nhau giữa các họ về chiều cao, độ rộng và thành phần tế bào. **Ống dẫn nhựa (resin canals)** có mặt ở một số họ và vắng mặt ở họ khác, và khi có mặt thì khác nhau về mật độ, kích thước cũng như kiểu phân bố theo vòng sinh trưởng. [CẦN NGUỒN: mô tả giải phẫu so sánh bốn họ Cupressaceae, Pinaceae, Podocarpaceae và Taxaceae, đặc biệt về sự có mặt và phân bố ống dẫn nhựa]

Vì cấu tạo đồng nhất, việc phân biệt ở cấp loài trong cùng một họ khó hơn nhiều so với ở cấp họ, do các loài cùng họ chia sẻ phần lớn đặc trưng giải phẫu và chỉ khác nhau ở các biến thiên định lượng nhỏ. Đây là cơ sở để nghiên cứu này tách bài toán thành hai cấp phân loại riêng biệt. [CẦN NGUỒN: nghiên cứu định lượng về giới hạn phân biệt loài bằng giải phẫu gỗ]

**Tính đẳng hướng của mặt cắt ngang.** Ảnh mặt cắt ngang ngang thớ không có hướng chuẩn về mặt sinh học. Quản bào xếp theo hàng xuyên tâm tỏa đều quanh tâm sinh trưởng, tia gỗ chạy theo mọi phương xuyên tâm, và ống dẫn nhựa phân bố không theo trục ưu tiên nào. Một mẫu cắt ngang xoay đi 90°, 180° hay 270° vẫn là một mẫu hợp lệ về mặt giải phẫu, khác với ảnh tự nhiên vốn có phương trọng lực xác định hướng lên xuống. Tính chất này là căn cứ lý thuyết cho chiến lược tăng cường dữ liệu bằng phép xoay rời rạc bốn góc, trình bày ở Mục 3.4 và khảo sát thực nghiệm ở Mục 4.5. [CẦN NGUỒN: tài liệu xác nhận tính đẳng hướng quang học của mặt cắt ngang gỗ, hoặc công trình đã khai thác tính chất này trong tăng cường dữ liệu]

## 2.2. Computer-Assisted and Deep Learning Wood Identification

Nhận dạng gỗ tự động bằng thị giác máy tính trải qua hai giai đoạn phương pháp luận rõ rệt.

**Giai đoạn đặc trưng thủ công.** Các hệ thống thế hệ đầu mô hình hóa vân gỗ như một bài toán phân loại kết cấu (texture classification), dùng đặc trưng trích thủ công kết hợp bộ phân lớp cổ điển. Hai nhóm đặc trưng phổ biến là ma trận đồng hiện mức xám (GLCM), mô tả quan hệ thống kê bậc hai giữa các cặp điểm ảnh, và mẫu nhị phân cục bộ (LBP), mã hóa quan hệ độ sáng giữa điểm ảnh trung tâm với lân cận. Cách tiếp cận này phụ thuộc nhiều vào điều kiện chụp và đòi hỏi thiết kế đặc trưng riêng cho từng bộ dữ liệu. [CẦN NGUỒN: công trình gốc về GLCM (Haralick), về LBP (Ojala), và ít nhất hai công trình áp dụng GLCM hoặc LBP cho nhận dạng gỗ]

**Giai đoạn học sâu.** Mạng nơ ron tích chập thay thế bước thiết kế đặc trưng bằng học biểu diễn trực tiếp từ ảnh. Ba công trình sau được xác định là liên quan trực tiếp đến nghiên cứu này.

**Ravindran và cộng sự (2020)** giới thiệu XyloTron, một hệ thống nhận dạng gỗ triển khai được tại hiện trường, công bố trên PLOS ONE. Đóng góp đáng chú ý nằm ở việc đưa mô hình học sâu ra khỏi phòng thí nghiệm thành thiết bị dùng được trong kiểm soát thương mại gỗ. **Lens và cộng sự (2020)** trình bày phương pháp định danh gỗ thương phẩm có hỗ trợ máy tính, công bố trên Plant Methods. **Filho và cộng sự (2022)** phân loại loài gỗ trực tiếp từ ảnh mặt cắt ngang, công bố trên Expert Systems with Applications, sử dụng cùng dạng dữ liệu đầu vào với nghiên cứu này.

[CẦN NGUỒN: số loài, quy mô tập dữ liệu, phương pháp chụp ảnh, kiến trúc và độ chính xác báo cáo của ba công trình trên, dùng để điền Bảng 1. Ba tài liệu này mới có ở dạng rút gọn trong đề cương, chưa có bản đầy đủ để trích số liệu.]

## 2.3. Evolution of Visual Backbones in Wood Classification

Các kiến trúc phân loại ảnh dùng làm nền tảng cho nhận dạng gỗ phản ánh bốn bước chuyển trong thiết kế mạng thị giác.

**Mạng tích chập dư.** He và cộng sự (2016) đề xuất kết nối tắt dư, cho phép huấn luyện mạng sâu ổn định và biến ResNet thành đường cơ sở mặc định cho phân loại ảnh trong nhiều năm.

**Tối ưu theo tỷ lệ hợp thành.** Tan và Le (2019) đề xuất EfficientNet với nguyên tắc mở rộng đồng thời chiều sâu, chiều rộng và độ phân giải theo một hệ số hợp thành, hướng tới cân bằng giữa độ chính xác và chi phí tính toán. Cùng hướng tối ưu chi phí, dòng MobileNet nhắm tới thiết bị biên bằng tích chập tách theo chiều sâu. [CẦN NGUỒN: tài liệu gốc của MobileNetV3, hiện không có trong danh mục tham khảo của đề cương]

**Mạng tích chập hiện đại.** Liu và cộng sự (2022) đề xuất ConvNeXt, hiện đại hóa thiết kế tích chập bằng cách tiếp thu các lựa chọn kiến trúc của transformer trong khi giữ nguyên phép toán tích chập.

**Transformer thị giác phân cấp.** Dosovitskiy và cộng sự (2021) chứng minh transformer thuần có thể phân loại ảnh ở quy mô lớn. Liu và cộng sự (2021) bổ sung cấu trúc phân cấp và cơ chế cửa sổ trượt trong Swin Transformer, đưa độ phức tạp về tuyến tính theo kích thước ảnh và tạo bản đồ đặc trưng đa tỷ lệ phù hợp với bài toán phân loại kết cấu.

Việc lựa chọn kiến trúc nào cho nhận dạng gỗ chưa được so sánh có hệ thống trên cùng một tập dữ liệu và cùng một giao thức huấn luyện. [CẦN NGUỒN: xác nhận khoảng trống này sau khi khảo sát đủ 25 tài liệu]

## 2.4. So sánh với các nghiên cứu liên quan

**Bảng 1.** So sánh nghiên cứu này với các công trình nhận dạng gỗ bằng học sâu liên quan. Ô đánh dấu `[CẦN NGUỒN]` cần điền sau khi đọc bản đầy đủ của tài liệu tương ứng.

| Công trình | Số loài | Phương pháp chụp | Kiến trúc sử dụng | Số kiến trúc so sánh | Phân tích hiệu năng tính toán | Kiểm soát rò rỉ theo mẫu gốc |
|---|---|---|---|---:|---|---|
| Ravindran và cộng sự (2020) | [CẦN NGUỒN] | [CẦN NGUỒN] | [CẦN NGUỒN] | [CẦN NGUỒN] | [CẦN NGUỒN] | [CẦN NGUỒN] |
| Lens và cộng sự (2020) | [CẦN NGUỒN] | [CẦN NGUỒN] | [CẦN NGUỒN] | [CẦN NGUỒN] | [CẦN NGUỒN] | [CẦN NGUỒN] |
| Filho và cộng sự (2022) | [CẦN NGUỒN] | Mặt cắt ngang | [CẦN NGUỒN] | [CẦN NGUỒN] | [CẦN NGUỒN] | [CẦN NGUỒN] |
| **Nghiên cứu này** | **17 (loài), 4 (họ)** | **Macroscopic 50×, 2560×2048** | ResNet-50, EfficientNet-B3, MobileNetV3-Large, ConvNeXt-Tiny, Swin-Tiny | **5** | **Có** (FLOPs, dung lượng, độ trễ, cùng thiết bị) | **Có** (chia theo mẫu gốc) |

Ba khoảng trống được xác định từ tổng quan này, tương ứng với ba đóng góp của nghiên cứu. Thứ nhất, các công trình trước tập trung vào một kiến trúc đơn lẻ, chưa có so sánh có kiểm soát giữa nhiều thế hệ kiến trúc trên cùng dữ liệu và cùng giao thức. Thứ hai, chi phí tính toán hiếm khi được báo cáo cùng độ chính xác, trong khi đây là yếu tố quyết định khả năng triển khai tại hiện trường. Thứ ba, rò rỉ dữ liệu theo mẫu gốc, tức nhiều ảnh cắt từ cùng một khúc gỗ bị phân tán vào các tập khác nhau, chưa được xử lý tường minh trong các công trình đã khảo sát. [CẦN NGUỒN: xác nhận cả ba khoảng trống sau khi đọc bản đầy đủ của ba công trình và mở rộng khảo sát lên 25 tài liệu]

---

## Danh mục tài liệu tham khảo đã xác nhận

Tám mục dưới đây trích nguyên văn từ `docs/references/WoodID_plan.md` dòng 227 đến 243. **Toàn bộ đều thiếu DOI, số tập, số trang.** Định dạng cuối cùng phụ thuộc chuẩn trích dẫn của tạp chí đích, hiện chưa xác định.

**Nhận dạng gỗ bằng học sâu**
1. Ravindran và cộng sự (2020). XyloTron: Field-deployable instrument for wood ID. *PLOS ONE*. [CẦN NGUỒN: DOI, số tập, số trang]
2. Lens và cộng sự (2020). Computer-assisted timber identification. *Plant Methods*. [CẦN NGUỒN: DOI, số tập, số trang]
3. Filho và cộng sự (2022). Wood species classification from cross-section images. *Expert Systems with Applications*. [CẦN NGUỒN: DOI, số tập, số trang]

**Kiến trúc nền tảng**
4. He và cộng sự (2016). Deep Residual Learning for Image Recognition. *CVPR*. [CẦN NGUỒN: số trang]
5. Tan và Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*. [CẦN NGUỒN: số trang]
6. Liu và cộng sự (2022). A ConvNet for the 2020s. *CVPR*. [CẦN NGUỒN: số trang]
7. Liu và cộng sự (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV*. [CẦN NGUỒN: số trang]
8. Dosovitskiy và cộng sự (2021). An Image is Worth 16x16 Words. *ICLR*. [CẦN NGUỒN: số trang]

### Tài liệu còn thiếu để đạt mức tối thiểu 25

| Nhóm | Số lượng ước tính | Ghi chú |
|---|---:|---|
| Giải phẫu gỗ lá kim và khóa định danh IAWA | 3 đến 5 | Phục vụ Mục 2.1 |
| Đặc trưng thủ công GLCM và LBP, gồm công trình gốc | 2 | Haralick và Ojala |
| Ứng dụng GLCM hoặc LBP cho nhận dạng gỗ | 2 đến 3 | Phục vụ Mục 2.2 |
| Nhận dạng gỗ bằng học sâu, ngoài ba công trình đã có | 5 đến 8 | Phục vụ Mục 2.2 và Bảng 1 |
| MobileNetV3 | 1 | Kiến trúc đang dùng nhưng chưa có tài liệu |
| Rò rỉ dữ liệu theo nhóm và phương pháp group split | 2 đến 3 | Phục vụ Mục 2.4 và Mục 3.3 |
| Loài đặc hữu Việt Nam và CITES | 2 | Phục vụ Mục 1 và Mục 5 |

Tổng cộng cần bổ sung khoảng 17 đến 24 tài liệu.

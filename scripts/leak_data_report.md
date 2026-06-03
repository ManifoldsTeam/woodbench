# Báo Cáo Phân Tích Rò Rỉ Dữ Liệu (Data Leakage Report)

**Ngày kiểm tra:** Cập nhật theo thời gian thực
**Tên Dataset:** `ny0g507h3p/wood-B` (Hugging Face)
**Công cụ phân tích:** Python (`check_data_leak.py`) sử dụng đối chiếu tên file và mã băm MD5.

---

## 1. Tóm tắt kết quả (Executive Summary)
Quá trình phân tích đã phát hiện **tình trạng rò rỉ dữ liệu (Data Leakage) nghiêm trọng** giữa các tập huấn luyện (train), xác thực (val) và kiểm thử (test) trong dataset `wood-B`. Có sự trùng lặp cả về tên file lẫn nội dung hình ảnh ở mức độ pixel, điều này sẽ làm sai lệch hoàn toàn kết quả đánh giá mô hình học máy.

**Thống kê tổng quan cấu trúc Dataset:**
* **Train split:** 3,973 ảnh
* **Val split:** 933 ảnh
* **Test split:** 924 ảnh

---

## 2. Chi tiết các lỗi phát hiện (Detailed Findings)

### A. Rò rỉ nội dung hình ảnh (Trùng mã băm MD5) - LỖI NGHIÊM TRỌNG
Phát hiện **120 trường hợp** file ảnh có nội dung giống hệt nhau (trùng mã MD5) nằm rải rác ở nhiều tập dữ liệu khác nhau. Việc tác giả thay đổi tên file không làm thay đổi bản chất nhị phân của bức ảnh.

**Các ví dụ tiêu biểu:**
* **Ảnh loài Xanthocyparis vietnamensis (MD5: `fc4ac7a05bad...`)**
  * Tồn tại ở tập Train: `3341. Xanthocyparis vietnamensis.3.(94).png`
  * Bị copy sang tập Test: `3341. Xanthocyparis vietnamensis.4.(124).png`
* **MD5: `68a28147f672...`**: Bị trùng giữa tập `train` và `val`.
* **MD5: `e5034faaf3c7...`**: Bị trùng giữa tập `train` và `test`.

### B. Trùng lặp theo tên file (Lỗi tổ chức thư mục)
Phát hiện **160 file ảnh** có cùng tên xuất hiện đồng thời ở nhiều thư mục phân chia. Điều này cho thấy logic chia tập (train/test split) ban đầu của tác giả đã bị lỗi, dẫn đến việc sao chép nhầm file thay vì di chuyển chúng.

**Các ví dụ tiêu biểu:**
* Xuất hiện ở cả 3 tập (`train`, `val`, `test`): `121.png`, `124.png`, `126.png`, `132.png`, `150.png`
* Xuất hiện ở 2 tập (`train` và `test`): `168.png`, `148.png`
* Xuất hiện ở 2 tập (`train` và `val`): `138.png`, `120.png`

---

## 3. Đánh giá mức độ ảnh hưởng (Impact Analysis)
Tình trạng rò rỉ này sẽ gây ra hậu quả trực tiếp đến quá trình phát triển AI:
1. **Hiện tượng Overfitting giả (Data Memorization):** Mô hình AI sẽ nhìn thấy các bức ảnh ở tập Test ngay trong lúc đang học (Train). 
2. **Ảo tưởng về độ chính xác (Inflated Metrics):** Khi đánh giá, điểm số Accuracy/F1-Score sẽ cao một cách bất thường vì AI chỉ đang "trả bài cũ" thay vì dự đoán dữ liệu mới.
3. **Thất bại trong thực tế:** Khi đem mô hình triển khai vào ứng dụng thực tế với những hình ảnh gỗ hoàn toàn mới, hiệu suất của mô hình sẽ sụt giảm thảm hại.

---

## 4. Hành động đề xuất (Recommended Actions)

Để đảm bảo tính khách quan cho mô hình phân loại gỗ, cần thực hiện ngay các bước sau:

1. **Làm sạch cục bộ (Data Cleaning):**
   * Viết script tự động quét 120 mã MD5 bị trùng lặp.
   * **Quy tắc xóa:** Nếu một ảnh xuất hiện ở tập `train` và các tập khác, **bắt buộc xóa** ảnh đó khỏi tập `val` và `test`. Tập `test` và `val` chỉ được chứa những hình ảnh độc bản chưa từng xuất hiện.
   * Xử lý tương tự với 160 file bị trùng tên.
2. **Báo cáo nguồn dữ liệu:**
   * Mở một *Issue* trên mục Community của Hugging Face repository `ny0g507h3p/wood-B` kèm theo báo cáo này để tác giả có thể cập nhật và vá lỗi dataset cho cộng đồng.
# Hướng dẫn sử dụng Tool Nhận diện Biển số xe

## 1. Giới thiệu
Tool nhận diện biển số xe được xây dựng trên nền tảng Web (Flask), sử dụng mô hình Deep Learning (YOLO + CNN) để tự động phát hiện và đọc ký tự trên biển số xe.

## 2. Cài đặt và Chạy ứng dụng

### Yêu cầu hệ thống
- Python 3.8 trở lên
- Các thư viện: Flask, Ultralytics, OpenCV, PyTorch, Pillow

### Cách chạy
1. Mở terminal tại thư mục gốc của dự án.
2. Chạy lệnh sau để khởi động server:
   ```bash
   python tools/flask_app.py
   ```
3. Mở trình duyệt và truy cập địa chỉ: `http://127.0.0.1:5000`

## 3. Các chức năng chính

### 3.1. Single Process (Xử lý đơn)
- **Chức năng**: Nhận diện biển số từ một ảnh duy nhất.
- **Cách dùng**:
    1. Chọn tab "Single Process".
    2. Nhấn vào khung upload để chọn ảnh biển số xe.
    3. Điều chỉnh thanh trượt "YOLO Confidence" nếu cần (mặc định 0.25).
    4. Kết quả sẽ hiển thị ngay lập tức bao gồm:
        - Ảnh đã được vẽ khung bao quanh ký tự.
        - Biển số xe dạng text.
        - Phân tích chi tiết từng ký tự (độ tin cậy).

### 3.2. Batch Process (Xử lý hàng loạt)
- **Chức năng**: Xử lý nhiều ảnh cùng lúc.
- **Cách dùng**:
    1. Chọn tab "Batch Process".
    2. Nhấn vào khung upload và chọn nhiều file ảnh cùng lúc.
    3. Hệ thống sẽ xử lý lần lượt và hiển thị bảng kết quả (Tên file, Trạng thái, Kết quả nhận diện).

### 3.3. Image Analysis (Phân tích ảnh)
- **Chức năng**: Hiển thị các bước xử lý ảnh trung gian để minh họa thuật toán (Tiêu chí 1.1).
- **Cách dùng**:
    1. Chọn tab "Image Analysis".
    2. Upload ảnh.
    3. Hệ thống sẽ hiển thị 4 bước:
        - Ảnh gốc.
        - Ảnh xám (Grayscale).
        - Làm mờ (Gaussian Blur).
        - Tách biên (Canny Edges).

## 4. Xử lý sự cố
- **Lỗi "No models found"**: Kiểm tra lại thư mục `src/weight` xem đã có file `character_detector.pt` và `character_classifier.v2.npz` chưa.
- **Không nhận diện được**: Thử điều chỉnh "YOLO Confidence" xuống thấp hơn (ví dụ 0.15) hoặc dùng ảnh rõ nét hơn.

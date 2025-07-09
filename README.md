# VHL_Optics

## Tổng quan

VHL_Optics là một dự án AI được thiết kế để phân tích dữ liệu quang học từ các hình ảnh thu thập được từ nhiều thiết bị di động khác nhau. Dự án cung cấp các công cụ để xử lý dữ liệu, phân tích hình ảnh, trích xuất đặc trưng và huấn luyện các mô hình học máy nhằm phân loại và dự đoán các thông số quang học.

## Tính năng chính

1. **Xử lý dữ liệu quang học**:
   - Tổ chức và xử lý dữ liệu hình ảnh từ các thiết bị di động.
   - Tách vùng quan tâm (ROI) và chuẩn hóa hình ảnh.

2. **Trích xuất đặc trưng**:
   - Trích xuất các đặc trưng từ hình ảnh để phục vụ cho các bài toán phân loại và hồi quy.
   - Sử dụng các kỹ thuật như GLCM, entropy, và histogram màu.

3. **Huấn luyện mô hình**:
   - Huấn luyện các mô hình phân loại và hồi quy dựa trên các đặc trưng đã trích xuất.
   - Sử dụng các thuật toán như Random Forest để phân tích dữ liệu.

4. **Pipeline tự động**:
   - Thực hiện toàn bộ quy trình từ xử lý dữ liệu, trích xuất đặc trưng đến huấn luyện mô hình chỉ với một lệnh duy nhất.

## Cấu trúc dự án
```
   VHL_Optics/ 
   ├── src/ 
   │ ├── __init__.py      # Tệp khởi tạo module 
   │ ├── config.py        # Cấu hình dự án 
   │ ├── loading.py       # Tải và tạo metadata 
   │ ├── processing.py    # Xử lý và tổ chức dữ liệu hình ảnh 
   │ ├── normalize.py     # Trích xuất đặc trưng từ hình ảnh 
   │ ├── model.py         # Huấn luyện mô hình phân loại và hồi quy 
   │ ├── roi.py           # Xử lý vùng quan tâm (ROI) 
   │ ├── squares.py       # Phát hiện các vùng hình vuông trong hình ảnh 
   │ ├── main.py          # Pipeline tự động 
   ├── data/              # Thư mục chứa dữ liệu 
   │ ├── full/            # Dữ liệu hình ảnh gốc 
   ├── requirements.txt   # Danh sách các thư viện cần thiết 
   ├── README.md          # Tài liệu dự án 
   ├── LICENSE            # Giấy phép sử dụng
```
## Hướng dẫn cài đặt

1. **Clone dự án**:
   ```bash
   git clone <repository-url>
   cd VHL_Optics
   ```
2. **Cài đặt các thư viện cần thiết**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Chuẩn bị dữ liệu**:
   Tạo thư mục data/full và đặt dữ liệu hình ảnh gốc vào đó.

## Hướng dẫn sử dụng
1. Chạy toàn bộ pipeline
   Chạy toàn bộ quy trình xử lý dữ liệu, trích xuất đặc trưng và huấn luyện mô hình bằng lệnh sau:
   ```bash
   python [main.py]
   ```
2. Thực hiện từng bước riêng lẻ
   - **Xử lý dữ liệu**:
      ```bash
      python src/processing.py
      ```
   - **Trích xuất đặc trưng**:
      ```bash
      python src/normalize.py
      ```
   - **Huấn luyện mô hình**:
      ```bash
      python src/model.py
      ```

## Yêu cầu hệ thống
 - Python 3.9 hoặc cao hơn.
 - Các thư viện được liệt kê trong `requirements.txt`.

## Giấy phép
Dự án được cấp phép theo MIT License.

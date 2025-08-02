Đây là nội dung **README.md** mới đã được cập nhật cho phù hợp với các thay đổi của dự án:

```markdown
## VHL Optics – PPM Prediction (Global Model)

**VHL Optics** là một dự án AI nhằm dự đoán nồng độ (ppm) của mẫu hóa chất dựa trên ảnh chụp từ nhiều thiết bị di động khác nhau. Ở phiên bản đầu, mỗi dòng máy được huấn luyện một mô hình riêng; tuy nhiên phiên bản hiện tại đã được **refactor** để sử dụng **một mô hình chung** cho toàn bộ thiết bị. Thông tin về điện thoại không còn là đặc trưng huấn luyện, giúp đơn giản hóa quy trình và mở rộng dễ dàng cho các thiết bị mới.

### Điểm nổi bật

- **Xử lý dữ liệu tự động**: quét dữ liệu thô, tạo file metadata và cắt vùng quan tâm (ROI) để chuẩn hóa ảnh.
- **Trích xuất đặc trưng hồi quy**: tính toán thống kê kênh màu, contrast GLCM, entropy, mật độ cạnh, v.v. cho mỗi ảnh.
- **Mô hình hồi quy duy nhất**: sử dụng RandomForest hoặc XGBoost để học từ toàn bộ dữ liệu, không cần cột điện thoại.
- **Giao diện trực quan**: hỗ trợ Streamlit để người dùng upload ảnh và nhận kết quả dự đoán ppm ngay lập tức.
- **Batch prediction**: dự đoán hàng loạt từ file đặc trưng tổng hợp và lưu kết quả ra CSV.

### Cấu trúc thư mục

```

├── app.py           # Ứng dụng Streamlit đơn giản cho dự đoán đơn lẻ và batch
├── main.py          # Entry point: e2e pipeline, feature extraction, training, Streamlit
├── config.py        # Định nghĩa đường dẫn dữ liệu và hằng số
├── loading.py       # Tạo và nạp metadata
├── processing.py    # Xử lý ảnh, trích xuất ROI và ảnh vuông
├── normalize.py     # Trích xuất đặc trưng hồi quy (không chứa phone)
├── model.py         # Huấn luyện mô hình hồi quy (per‑phone & global)
├── predict.py       # Dự đoán ppm (đơn lẻ & batch)
├── roi.py           # Các hàm hỗ trợ cắt ROI
├── squares.py       # Phát hiện contour hình vuông
├── data/            # (tự tạo) chứa dữ liệu thô và ảnh đã xử lý

````

### Cài đặt

1. **Clone dự án** và cài đặt các thư viện cần thiết:

```bash
git clone <repository-url>
cd VHL_Optics_Regressor
pip install -r requirements.txt
````

Nếu không có file `requirements.txt`, cài đặt tối thiểu các thư viện:

```
numpy
pandas
opencv-python
scikit-learn
xgboost
streamlit
scikit-image
tqdm
```

2. **Chuẩn bị dữ liệu**: đặt dữ liệu ảnh gốc vào `data/full/HP5_data` theo cấu trúc: *loại hóa chất / điện thoại / lần chụp / ảnh*. Ứng dụng sẽ tự động quét và tạo metadata.

### Sử dụng

#### Chạy toàn bộ pipeline (e2e)

Lệnh sau sẽ tạo metadata, xử lý ảnh, trích xuất đặc trưng và huấn luyện mô hình chung:

```bash
python main.py e2e
```

Sau khi chạy, các file đặc trưng sẽ được lưu trong `data/csv/features_all.csv` và các mô hình huấn luyện được lưu trong `data/models/`.

#### Trích xuất đặc trưng và huấn luyện riêng

Bạn có thể chạy từng bước riêng biệt:

```bash
# Trích xuất đặc trưng (khi đã có ảnh vuông)
python main.py feature

# Huấn luyện mô hình chung (khi đã có features_all.csv)
python main.py train
```

#### Dự đoán ppm cho một ảnh

Sử dụng giao diện Streamlit đơn giản:

```bash
streamlit run app.py
```

Hoặc chạy trực tiếp `python main.py` (mặc định sẽ mở giao diện Streamlit). Người dùng chỉ cần upload ảnh, chọn mô hình (**RF** hoặc **XGB**) và nhận kết quả ppm. Không cần nhập dòng máy.

Bạn cũng có thể gọi hàm dự đoán từ mã Python:

```python
from predict import predict_regression_general

ppm = predict_regression_general('path/to/image.jpg', model_choice='RF')
print(f"Predicted ppm: {ppm:.2f}")
```

#### Batch prediction

Để dự đoán hàng loạt, chạy từ Streamlit (`app.py`) hoặc sử dụng hàm:

```python
from predict import predict_test_set_general

results = predict_test_set_general(output_dir='batch_predictions')
print(results.head())
```

Kết quả được lưu vào `predictions_general.csv` với cột: `id_img`, `true_ppm`, `pred_rf_ppm`, `pred_xgb_ppm`, `diff_rf_pct`, `diff_xgb_pct`.

### Ghi chú về thông tin điện thoại

Phiên bản hiện tại **không sử dụng** bất kỳ thông tin nào về dòng máy làm đặc trưng huấn luyện. Các phiên bản trước huấn luyện riêng từng thiết bị, nhưng điều đó gây khó khăn khi gặp thiết bị mới. Mô hình chung đơn giản hơn và áp dụng được cho mọi ảnh.

### Yêu cầu hệ thống

* Python 3.8 trở lên.
* Bộ nhớ đủ để xử lý ảnh và huấn luyện mô hình (nên dùng môi trường có GPU cho XGBoost nếu dữ liệu lớn).

### Giấy phép

Dự án được phân phối theo giấy phép MIT. Bạn có thể tự do sử dụng và chỉnh sửa mã nguồn.

```

README này đã được viết lại để phản ánh mô hình chung không cần dữ liệu điện thoại và chỉ tập trung vào bài toán hồi quy nồng độ ppm.
```

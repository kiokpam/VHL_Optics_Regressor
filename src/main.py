import os
import sys
import streamlit as st
import pandas as pd

from config import DATA_DIR, META_COLORS
from loading import create_meta_data
from processing import process_data
from normalize import getFeature
from model import train_general_model
from predict import predict_regression_general


def e2e_pipeline() -> None:
    # 1) Tạo metadata
    create_meta_data(
        data_dir=os.path.join(DATA_DIR, 'full', 'HP5_data'),
        out_dir=META_COLORS
    )
    print('Metadata creation Success!')

    # 2) Xử lý ảnh
    process_data(
        data_dir=os.path.join(DATA_DIR, 'full', 'HP5_data'),
        meta_path=META_COLORS
    )
    print('Processing Data Success!')

    # 3) Trích xuất đặc trưng
    getFeature(
        df_path=META_COLORS,
        dir_path=os.path.join(DATA_DIR, 'square image'),
        out_path=os.path.join(DATA_DIR, 'csv'),
        use_square_background=True
    )
    print('Feature Extraction Success!')

    # 4) Huấn luyện mô hình hồi quy chung (3 folds, chỉ XGB)
    try:
        train_general_model(
            features_path=os.path.join(DATA_DIR, 'csv', 'features_all.csv'),
            out_path=os.path.join(DATA_DIR, 'models'),
            test_size=0.2,
            n_splits=3,
            random_state=42
        )
        print('Training Model Success!')
    except Exception as e:
        print(f'Training Model Failed: {e}')


def feature_pipeline() -> None:
    """Chỉ thực thi bước trích xuất đặc trưng."""
    print('Starting feature extraction only...')
    getFeature(
        df_path=META_COLORS,
        dir_path=os.path.join(DATA_DIR, 'square image'),
        out_path=os.path.join(DATA_DIR, 'csv'),
        use_square_background=True
    )
    print('Feature Extraction Completed.')


def train_pipeline() -> None:
    """Chỉ thực thi bước huấn luyện mô hình chung."""
    print('Starting training only...')
    try:
        train_general_model(
            features_path=os.path.join(DATA_DIR, 'csv', 'features_all.csv'),
            out_path=os.path.join(DATA_DIR, 'models'),
            test_size=0.2,
            n_splits=3,
            random_state=42
        )
        print('Training Model Success!')
    except Exception as e:
        print(f'Training Model Failed: {e}')


def streamlit_app():
    # Thiết lập trang Streamlit
    st.set_page_config(page_title="AI_Optics PPM Prediction", layout="centered")
    st.title("AI_Optics PPM Prediction")
    st.markdown(
        "Upload an image and select a regression model to predict ppm concentration."
    )

    # Sidebar options
    st.sidebar.header("Options")
    # Chỉ còn lựa chọn XGBoost để rút ngắn thời gian
    model_choice = st.sidebar.selectbox(
        "Select model", ["XGBoost"]
    )

    # File uploader for single image prediction
    uploaded_file = st.file_uploader(
        "Upload image file", type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded_file:
        # Lưu tạm file
        tmp_dir = os.path.join(DATA_DIR, "temp_uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        img_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Hiển thị ảnh upload
        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        # Dự đoán khi người dùng bấm nút
        if st.button("Predict ppm"):
            ppm = predict_regression_general(img_path, model_choice)
            st.success(f"Predicted concentration: {ppm:.2f} ppm")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == 'e2e':
            e2e_pipeline()
            sys.exit(0)
        elif cmd == 'feature':
            feature_pipeline()
            sys.exit(0)
        elif cmd == 'train':
            train_pipeline()
            sys.exit(0)
    # Mặc định khởi chạy giao diện
    streamlit_app()

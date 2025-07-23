import os
import sys
import streamlit as st
import pandas as pd

from config import DATA_DIR, META_COLORS
from loading import create_meta_data
from processing import process_data
from normalize import getFeature
from model import train_both_pipelines
from predict import predict_regression


def e2e_pipeline() -> None:
    # 1) Create metadata
    create_meta_data(
        data_dir=os.path.join(DATA_DIR, 'full', 'HP5_data'),
        out_dir=META_COLORS
    )
    print('Metadata creation Success!')

    # 2) Process images
    process_data(
        data_dir=os.path.join(DATA_DIR, 'full', 'HP5_data'),
        meta_path=META_COLORS
    )
    print('Processing Data Success!')

    # 3) Feature extraction
    getFeature(
        df_path=META_COLORS,
        dir_path=os.path.join(DATA_DIR, 'square image'),
        out_path=os.path.join(DATA_DIR, 'csv'),
        use_square_background=True
    )
    print('Feature Extraction Success!')

    # 4) Train regressors
    train_both_pipelines(
        meta_path=META_COLORS,
        dir_path=os.path.join(DATA_DIR, 'csv'),
        out_path=os.path.join(DATA_DIR, 'models'),
        test_size=0.2,
        val_size=0.25,
        random_state=42
    )
    print('Training Model Success!')


def feature_pipeline() -> None:
    """
    Chỉ thực thi bước Feature Extraction dựa trên metadata và ảnh đã chuẩn.
    """
    print('Starting feature extraction only...')
    getFeature(
        df_path=META_COLORS,
        dir_path=os.path.join(DATA_DIR, 'square image'),
        out_path=os.path.join(DATA_DIR, 'csv'),
        use_square_background=True
    )
    print('Feature Extraction Completed.')


def train_pipeline() -> None:
    """
    Chỉ thực thi bước Train regressors dựa trên dữ liệu đã extract sẵn.
    """
    print('Starting training only...')
    try:
        train_both_pipelines(
            meta_path=META_COLORS,
            dir_path=os.path.join(DATA_DIR, 'csv'),
            out_path=os.path.join(DATA_DIR, 'models'),
            test_size=0.2,
            val_size=0.25,
            random_state=42
        )
        print('Training Model Success!')
    except Exception as e:
        print(f'Training Model Failed: {e}')


def streamlit_app():
    st.set_page_config(page_title="AI_Optics PPM Prediction", layout="centered")
    st.title("AI_Optics PPM Prediction")
    st.markdown("Upload an image, select your phone type and regression model to predict ppm concentration.")

    try:
        df_meta = pd.read_csv(META_COLORS)
        phones = sorted(df_meta['Phones'].astype(str).unique())
    except Exception as e:
        st.error(f"Failed to load metadata: {e}")
        return

    st.sidebar.header("Options")
    phone = st.sidebar.selectbox("Select phone type", phones)
    model_choice = st.sidebar.selectbox("Select model", ["RandomForest", "XGBoost"])

    uploaded_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file:
        tmp_dir = os.path.join(DATA_DIR, "temp_uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        img_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict ppm"):
            ppm = predict_regression(img_path, phone, model_choice)
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
    # Default: launch Streamlit
    streamlit_app()

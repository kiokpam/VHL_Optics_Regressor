import os
import streamlit as st
import pandas as pd

from predict import (
    predict_regression,
    predict_test_set,
    predict_regression_general,
    predict_test_set_general,
)
from config import META_COLORS, DATA_DIR

st.title("AI Optics: PPM Prediction")

# --- Single Image Prediction (general model) ---
st.sidebar.header("Single Image Prediction")
image_file    = st.sidebar.file_uploader("Upload an image", type=['jpg','png','jpeg'])
# Chỉ còn lựa chọn XGB vì mô hình chung đã loại bỏ RandomForest
model_choice  = st.sidebar.selectbox("Model", ['XGB'])

if st.sidebar.button("Predict PPM"):
    if image_file is not None:
        temp_path = os.path.join(DATA_DIR, 'temp_predict', image_file.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, 'wb') as f:
            f.write(image_file.getbuffer())
        try:
            # Gọi mô hình chung, không sử dụng thông tin điện thoại
            pred = predict_regression_general(temp_path, model_choice)
            st.sidebar.success(f"Predicted PPM: {pred:.2f}")
        except Exception as e:
            st.sidebar.error(f"Prediction error: {e}")
    else:
        st.sidebar.warning("Please upload an image first.")

# --- Batch Prediction with General Model ---
st.header("Batch Prediction on Full Test Set")
if st.button("Run Batch Prediction for All Images"):
    output_dir = os.path.join(DATA_DIR, 'batch_predictions')
    os.makedirs(output_dir, exist_ok=True)
    try:
        result_df = predict_test_set_general(output_dir=output_dir)
        st.success(f"Done! Saved predictions_general.csv in {output_dir}")
        st.write(result_df.head())
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

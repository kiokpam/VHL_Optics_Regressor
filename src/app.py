import os
import streamlit as st
import pandas as pd

from predict import predict_regression, predict_test_set
from config import META_COLORS, DATA_DIR

st.title("AI Optics: PPM Prediction")

# --- Single Image Prediction ---
st.sidebar.header("Single Image Prediction")
image_file    = st.sidebar.file_uploader("Upload an image", type=['jpg','png','jpeg'])
phone_choice  = st.sidebar.selectbox("Phone model", pd.read_csv(META_COLORS)['Phones'].unique())
model_choice  = st.sidebar.selectbox("Model", ['RF', 'XGB'])

if st.sidebar.button("Predict Single Image"):
    if image_file is not None:
        temp_path = os.path.join(DATA_DIR, 'temp_predict', image_file.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, 'wb') as f:
            f.write(image_file.getbuffer())
        try:
            pred = predict_regression(temp_path, phone_choice, model_choice)
            st.sidebar.success(f"Predicted PPM: {pred:.2f}")
        except Exception as e:
            st.sidebar.error(f"Prediction error: {e}")
    else:
        st.sidebar.warning("Please upload an image first.")

# --- Batch Prediction on Test Sets ---
st.header("Batch Prediction on Test Sets")
if st.button("Run Batch Prediction for All Phones"):
    df_meta = pd.read_csv(META_COLORS)
    output_dir = os.path.join(DATA_DIR, 'batch_predictions')
    os.makedirs(output_dir, exist_ok=True)

    for phone in df_meta['Phones'].unique():
        st.write(f"Processing {phone}...")
        try:
            predict_test_set(phone, output_dir=output_dir)
            st.write(f"  • Saved predictions_{phone}.csv")
        except Exception as e:
            st.error(f"Error for {phone}: {e}")

    st.success(f"Done! CSV files for each phone are in:\n{output_dir}")

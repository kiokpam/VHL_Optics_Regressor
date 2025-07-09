import streamlit as st
import pandas as pd
import os

from config import DATA_DIR, META_COLORS
from predict import predict_regression

# Streamlit app for AI_Optics ppm prediction

def main():
    st.set_page_config(page_title="AI_Optics PPM Prediction", layout="wide")
    st.title("AI_Optics PPM Prediction")
    st.markdown("Upload an image, select your phone type and regression model to predict ppm concentration.")

    # Load phone list from metadata
    try:
        df_meta = pd.read_csv(META_COLORS)
        phones = sorted(df_meta['Phones'].astype(str).unique())
    except Exception as e:
        st.error(f"Failed to load metadata: {e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        phone = st.selectbox("Select phone type", phones)
    with col2:
        model_choice = st.selectbox("Select regression model", ["RandomForest", "XGBoost"])

    uploaded_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file:
        # Save upload to temp folder
        tmp_dir = os.path.join(DATA_DIR, "temp_uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        img_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict ppm"):
            try:
                ppm = predict_regression(img_path, phone, model_choice)
                st.success(f"Predicted concentration: {ppm:.2f} ppm")
            except Exception as e:
                st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()

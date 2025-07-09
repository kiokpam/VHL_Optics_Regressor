import os
import joblib
import pandas as pd
import cv2

from roi import runROI
from normalize import FeatureExtractor
from config import DATA_DIR


def predict_regression(
    image_path: str,
    phone: str,
    model_choice: str,
    temp_dir: str = None
) -> float:
    """
    1) Run ROI detection and crop
    2) Extract regression features
    3) Load scaler and chosen regressor (RF or XGB) for the given phone
    4) Return predicted ppm
    """
    # Prepare temporary directory
    if temp_dir is None:
        temp_dir = os.path.join(DATA_DIR, "temp_predict")
    os.makedirs(temp_dir, exist_ok=True)

    # 1) ROI detection
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    squared, sample, background, _ = runROI(image=image, kit=None)
    if squared is None:
        raise ValueError("ROI failed: could not detect square region")
    squared_path = os.path.join(temp_dir, os.path.basename(image_path))
    cv2.imwrite(squared_path, squared)

    # 2) Feature extraction
    extractor = FeatureExtractor()
    _, reg_feats = extractor.extract_features(squared_path, os.path.basename(image_path))
    if reg_feats is None:
        raise ValueError("Feature extraction failed: no regression features")
    df = pd.DataFrame([reg_feats])
    # Drop extraneous columns if present
    df = df.drop(columns=[col for col in ['id_img', 'ppm'] if col in df.columns])

    # 3) Scaling
    scaler_path = os.path.join(DATA_DIR, "models", f"regressor_scaler_{phone}.pkl")
    scaler = joblib.load(scaler_path)
    Xs = scaler.transform(df)

    # 4) Load model
    choice = model_choice.lower()
    if choice == 'randomforest' or choice == 'rf':
        model_file = f"regressor_rf_{phone}.pkl"
    elif choice in ('xgboost', 'xgb'):
        model_file = f"regressor_xgb_{phone}.pkl"
    else:
        raise ValueError("Model choice must be 'RandomForest' or 'XGBoost'")
    model_path = os.path.join(DATA_DIR, 'models', model_file)
    model = joblib.load(model_path)

    # 5) Predict
    pred = model.predict(Xs)
    return float(pred[0])

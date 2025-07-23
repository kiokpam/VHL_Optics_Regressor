import os
import joblib
import pandas as pd
import numpy as np
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
    3) Load pipeline (feature-select + scale + model) for the given phone and model_choice
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
    reg_feats = extractor.extract_features(
        squared_path,
        use_square_background=True,
        id_img=os.path.basename(image_path),
        ppm=None
    )
    if reg_feats is None:
        raise ValueError("Feature extraction failed: no regression features")
    df = pd.DataFrame([reg_feats])
    df = df.drop(columns=[c for c in ['id_img', 'ppm'] if c in df.columns])

    # 3) Load pipeline
    choice = model_choice.lower()
    if choice in ('randomforest', 'rf'):
        pipeline_file = f"pipe_rf_{phone}.pkl"
    elif choice in ('xgboost', 'xgb'):
        pipeline_file = f"pipe_xgb_{phone}.pkl"
    else:
        raise ValueError("Model choice must be 'rf' or 'xgb' for predict_regression")

    pipeline_path = os.path.join(DATA_DIR, 'models', pipeline_file)
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)

    # 4) Predict
    preds = pipeline.predict(df)
    return float(np.round(preds[0], 2))


def predict_test_set(
    phone: str,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Batch predict on the test feature CSV for the given phone using both RF and XGB pipelines.
    Returns a DataFrame with columns:
      - id_img
      - true_ppm        (rounded to 2 decimals)
      - pred_rf_ppm     (rounded to 2 decimals)
      - pred_xgb_ppm    (rounded to 2 decimals)
      - diff_rf_pct     (|pred_rf - true| / true * 100, rounded to 2 decimals)
      - diff_xgb_pct    (|pred_xgb - true| / true * 100, rounded to 2 decimals)
    And writes to predictions_<phone>.csv if output_dir is provided.
    """
    # 1) Load test features CSV
    csv_path = os.path.join(DATA_DIR, 'csv', f'features_{phone}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Features CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'ppm' not in df.columns:
        raise ValueError("Features CSV must contain 'ppm' column for true labels")

    # 2) Load pipelines
    rf_pipe_path  = os.path.join(DATA_DIR, 'models', f'pipe_rf_{phone}.pkl')
    xgb_pipe_path = os.path.join(DATA_DIR, 'models', f'pipe_xgb_{phone}.pkl')
    rf_pipe  = joblib.load(rf_pipe_path)
    xgb_pipe = joblib.load(xgb_pipe_path)

    # 3) Prepare results
    ids = df['id_img'] if 'id_img' in df.columns else df.index.astype(str)
    true = df['ppm']
    X = df.drop(columns=[c for c in ['id_img', 'ppm'] if c in df.columns])

    # 4) Predict
    pred_rf  = rf_pipe.predict(X)
    pred_xgb = xgb_pipe.predict(X)

    # 5) Compute percent difference
    diff_rf_pct  = np.abs(pred_rf - true) / (true + 1e-6) * 100
    diff_xgb_pct = np.abs(pred_xgb - true) / (true + 1e-6) * 100

    # 6) Build DataFrame and round values
    result_df = pd.DataFrame({
        'id_img':       ids,
        'true_ppm':     np.round(true, 2),
        'pred_rf_ppm':  np.round(pred_rf, 2),
        'pred_xgb_ppm': np.round(pred_xgb, 2),
        'diff_rf_pct':  np.round(diff_rf_pct, 2),
        'diff_xgb_pct': np.round(diff_xgb_pct, 2),
    })

    # 7) Save to CSV per phone
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'predictions_{phone}.csv')
        result_df.to_csv(out_path, index=False)
        print(f"Saved batch predictions to {out_path}")

    return result_df

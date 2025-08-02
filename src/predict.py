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
    Hàm dự đoán ppm cho mô hình cũ, vẫn cần truyền tên điện thoại.
    Giữ lại để đảm bảo tương thích với phiên bản trước, nhưng không còn được sử dụng trong mô hình chung.
    """
    if temp_dir is None:
        temp_dir = os.path.join(DATA_DIR, "temp_predict")
    os.makedirs(temp_dir, exist_ok=True)

    # 1) Xác định ROI và lưu ảnh vuông tạm
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    squared, sample, background, _ = runROI(image=image, kit=None)
    if squared is None:
        raise ValueError("ROI failed: could not detect square region")
    squared_path = os.path.join(temp_dir, os.path.basename(image_path))
    cv2.imwrite(squared_path, squared)

    # 2) Trích xuất đặc trưng
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

    # 3) Nạp pipeline tương ứng với dòng máy
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

    preds = pipeline.predict(df)
    return float(np.round(preds[0], 2))

def predict_regression_general(
    image_path: str,
    model_choice: str,
    phone: str = None,
    temp_dir: str = None
) -> float:
    """
    Dự đoán ppm bằng mô hình chung (không cần dòng máy).
    Tham số phone ở đây chỉ giữ để tương thích cũ; mô hình bỏ qua hoàn toàn.
    """
    if temp_dir is None:
        temp_dir = os.path.join(DATA_DIR, "temp_predict")
    os.makedirs(temp_dir, exist_ok=True)

    # 1) ROI
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    squared, sample, background, _ = runROI(image=image, kit=None)
    if squared is None:
        raise ValueError("ROI failed: could not detect square region")
    squared_path = os.path.join(temp_dir, os.path.basename(image_path))
    cv2.imwrite(squared_path, squared)

    # 2) Trích xuất đặc trưng
    extractor = FeatureExtractor()
    reg_feats = extractor.extract_features(
        squared_path,
        use_square_background=True,
        id_img=os.path.basename(image_path),
        ppm=None
    )
    if reg_feats is None:
        raise ValueError("Feature extraction failed: no regression features")

    # Tạo DataFrame và loại bỏ cột id_img, ppm (phone_norm không tồn tại)
    df = pd.DataFrame([reg_feats])
    df = df.drop(columns=[c for c in ['id_img', 'ppm', 'phone_norm'] if c in df.columns], errors='ignore')

    # 3) Nạp pipeline chung
    choice = model_choice.lower()
    if choice in ('randomforest', 'rf'):
        pipeline_file = 'pipe_rf_general.pkl'
    elif choice in ('xgboost', 'xgb'):
        pipeline_file = 'pipe_xgb_general.pkl'
    else:
        raise ValueError("Model choice must be 'rf' or 'xgb' for predict_regression_general")

    pipeline_path = os.path.join(DATA_DIR, 'models', pipeline_file)
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)

    preds = pipeline.predict(df)
    return float(np.round(preds[0], 2))

def predict_test_set(
    phone: str,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Hàm batch prediction cũ cho từng dòng máy (giữ lại để tương thích). 
    Sử dụng hai mô hình riêng (RF và XGB) để dự đoán và so sánh sai số.
    """
    csv_path = os.path.join(DATA_DIR, 'csv', f'features_{phone}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Features CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'ppm' not in df.columns:
        raise ValueError("Features CSV must contain 'ppm' column for true labels")

    rf_pipe_path  = os.path.join(DATA_DIR, 'models', f'pipe_rf_{phone}.pkl')
    xgb_pipe_path = os.path.join(DATA_DIR, 'models', f'pipe_xgb_{phone}.pkl')
    rf_pipe  = joblib.load(rf_pipe_path)
    xgb_pipe = joblib.load(xgb_pipe_path)

    ids = df['id_img'] if 'id_img' in df.columns else df.index.astype(str)
    true = df['ppm']
    X = df.drop(columns=[c for c in ['id_img', 'ppm'] if c in df.columns])

    pred_rf  = rf_pipe.predict(X)
    pred_xgb = xgb_pipe.predict(X)

    diff_rf_pct  = np.abs(pred_rf - true) / (true + 1e-6) * 100
    diff_xgb_pct = np.abs(pred_xgb - true) / (true + 1e-6) * 100

    result_df = pd.DataFrame({
        'id_img':       ids,
        'true_ppm':     np.round(true, 2),
        'pred_rf_ppm':  np.round(pred_rf, 2),
        'pred_xgb_ppm': np.round(pred_xgb, 2),
        'diff_rf_pct':  np.round(diff_rf_pct, 2),
        'diff_xgb_pct': np.round(diff_xgb_pct, 2),
    })

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'predictions_{phone}.csv')
        result_df.to_csv(out_path, index=False)
        print(f"Saved batch predictions to {out_path}")

    return result_df

def predict_test_set_general(
    output_dir: str = None
) -> pd.DataFrame:
    """
    Batch predict trên toàn bộ đặc trưng `features_all.csv` bằng mô hình chung.
    Kết quả không còn cột phone_norm; chỉ hiển thị id_img, true_ppm, pred_rf_ppm, pred_xgb_ppm và % sai lệch.
    """
    csv_path = os.path.join(DATA_DIR, 'csv', 'features_all.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Combined features CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'ppm' not in df.columns:
        raise ValueError("Combined features CSV must contain 'ppm' column for true labels")

    ids  = df['id_img'] if 'id_img' in df.columns else df.index.astype(str)
    true = df['ppm']
    drop_cols = ['id_img', 'ppm', 'phone_norm']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    rf_path  = os.path.join(DATA_DIR, 'models', 'pipe_rf_general.pkl')
    xgb_path = os.path.join(DATA_DIR, 'models', 'pipe_xgb_general.pkl')
    rf_pipe  = joblib.load(rf_path)
    xgb_pipe = joblib.load(xgb_path)

    pred_rf  = rf_pipe.predict(X)
    pred_xgb = xgb_pipe.predict(X)

    diff_rf_pct  = np.abs(pred_rf - true) / (true + 1e-6) * 100
    diff_xgb_pct = np.abs(pred_xgb - true) / (true + 1e-6) * 100

    result_df = pd.DataFrame({
        'id_img':      ids,
        'true_ppm':    np.round(true, 2),
        'pred_rf_ppm': np.round(pred_rf, 2),
        'pred_xgb_ppm': np.round(pred_xgb, 2),
        'diff_rf_pct':  np.round(diff_rf_pct, 2),
        'diff_xgb_pct': np.round(diff_xgb_pct, 2),
    })

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'predictions_general.csv')
        result_df.to_csv(out_path, index=False)
        print(f"Saved batch predictions to {out_path}")

    return result_df

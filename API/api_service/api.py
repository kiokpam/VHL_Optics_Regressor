import os
import glob
import joblib
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

# Sử dụng imports tuyệt đối cho package
from api_service.roi import runROI
from api_service.normalize import FeatureExtractor
from api_service.config import DATA_DIR
from api_service.normalize import FeatureExtractor
from api_service.config import DATA_DIR

app = FastAPI(
    title="AI Optics PPM Prediction API",
    description="REST API để dự đoán nồng độ ppm từ ảnh",
    version="1.0.0"
)

# Load các pipeline XGB đã huấn luyện
models_dir = os.path.join(DATA_DIR, "models")
pipelines = {}
for path in glob.glob(os.path.join(models_dir, "pipe_xgb_*.pkl")):
    phone = os.path.basename(path).replace("pipe_xgb_", "").replace(".pkl", "")
    pipelines[phone] = joblib.load(path)

extractor = FeatureExtractor()

@app.post("/predict/", summary="Dự đoán nồng độ ppm từ ảnh")
async def predict_ppm(
    phone: str,
    file: UploadFile = File(...)
):
    # Kiểm tra phone
    if phone not in pipelines:
        raise HTTPException(status_code=404, detail=f"Unknown phone '{phone}'. Available: {list(pipelines.keys())}")

    # Đọc ảnh
    contents = await file.read()
    img_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Phát hiện ROI
    squared, _, _, _ = runROI(image=image)
    if squared is None:
        raise HTTPException(status_code=500, detail="ROI detection failed")

    # Lưu tạm và trích xuất đặc trưng
    tmp_dir = os.path.join(DATA_DIR, "temp_api")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, file.filename)
    cv2.imwrite(tmp_path, squared)

    feats = extractor.extract_features(
        tmp_path,
        use_square_background=True
    )
    if not feats:
        raise HTTPException(status_code=500, detail="Feature extraction error")

    # Chuẩn bị DataFrame và dự đoán
    df = pd.DataFrame([{k: v for k, v in feats.items() if k not in {"id_img", "ppm"}}])
    pred = pipelines[phone].predict(df)[0]

    return JSONResponse({"phone": phone, "ppm": float(round(pred, 2))})
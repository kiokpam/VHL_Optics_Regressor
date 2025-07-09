import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

from config import DATA_DIR, META_COLORS, IMAGE_EXTENSIONS
from loading import load_data

class FeatureExtractor:
    def __init__(self, hue_bins=16):
        self.hue_bins = hue_bins

    def extract_features(self, image_path: str, id_img: str, types: str = '', ppm: float = 0.0):
        classification_features = {}
        regression_features = {}

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None, None

        classification_features['id_img'] = id_img
        classification_features['type']   = types
        regression_features['id_img'] = id_img
        regression_features['ppm']    = ppm

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        hsv_resized = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        gray_resized = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        h, s, v = cv2.split(hsv_resized)
        # Hue histogram
        hue_hist = cv2.calcHist([h], [0], None, [self.hue_bins], [0, 180]).flatten()
        hue_hist = hue_hist / hue_hist.sum() if hue_hist.sum() != 0 else hue_hist
        for i in range(self.hue_bins):
            classification_features[f'hue_hist_{i}'] = hue_hist[i]
        # Stats
        classification_features['mean_hue'] = np.mean(h)
        classification_features['std_hue']  = np.std(h)
        classification_features['mean_sat'] = np.mean(s)
        classification_features['std_sat']  = np.std(s)
        classification_features['mean_val'] = np.mean(v)
        classification_features['std_val']  = np.std(v)

        # GLCM
        gray_scaled = np.uint8(gray_resized / 4)
        glcm = graycomatrix(gray_scaled, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)
        regression_features['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
        # Entropy
        regression_features['entropy'] = shannon_entropy(gray_resized)
        # Edge density
        edges = cv2.Canny(gray_resized, 100, 200)
        regression_features['edge_density'] = np.sum(edges > 0) / edges.size
        # Contour
        _, thresh = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area else 0
        else:
            area = solidity = 0
        regression_features['contour_area'] = area
        regression_features['solidity'] = solidity

        # Hue/Sat ratio
        mean_hue_total = classification_features['mean_hue']
        mean_sat_total = classification_features['mean_sat']
        regression_features['hue_sat_ratio'] = mean_hue_total / mean_sat_total if mean_sat_total else 0

        return classification_features, regression_features


def getFeature(
    df_path: str = None,
    dir_path: str = None,
    out_path: str = None
) -> None:
    """
    Extract features from squared images using metadata Filename.
    """
    if df_path is None:
        df_path = META_COLORS
    df = pd.read_csv(df_path)

    if dir_path is None:
        dir_path = os.path.join(DATA_DIR, 'square image')
    if out_path is None:
        out_path = os.path.join(DATA_DIR, 'csv')
    os.makedirs(out_path, exist_ok=True)

    extractor = FeatureExtractor()
    phones = df['Phones'].unique().tolist()

    for phone in phones:
        df_phone = df[df['Phones'] == phone]
        clf_list, rgs_list = [], []
        total = len(df_phone)
        with tqdm(total=total, desc=f"Feature Extracting [{phone}]", unit="image") as pbar:
            for _, row in df_phone.iterrows():
                filename = row['Filename']
                types    = row['Types']
                ppm      = row['ppm']
                id_img   = row['Id_imgs']
                image_path = os.path.join(dir_path, phone, types, filename)
                if not os.path.exists(image_path):
                    pbar.update(1)
                    continue
                clf_feat, rgs_feat = extractor.extract_features(image_path, id_img, types, ppm)
                if clf_feat and rgs_feat:
                    clf_list.append(clf_feat)
                    rgs_list.append(rgs_feat)
                pbar.update(1)

        clf_df = pd.DataFrame(clf_list)
        rgs_df = pd.DataFrame(rgs_list)
        clf_df.to_csv(os.path.join(out_path, f'clf_{phone}.csv'), index=False)
        rgs_df.to_csv(os.path.join(out_path, f'rgs_{phone}.csv'), index=False)

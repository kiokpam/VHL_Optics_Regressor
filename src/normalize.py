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
    def __init__(self, hue_bins: int = 16):
        self.hue_bins = hue_bins

    def extract_features(
            self,
            image_path: str,
            use_square_background: bool = False,
            id_img: str = None,
            ppm: float = None
    ) -> dict:
        feats = {}
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None

        if id_img is not None:
            feats['id_img'] = id_img
        if ppm is not None:
            feats['ppm'] = ppm

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        hsv     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h, s, v = cv2.split(hsv)

        # Base regression features
        feats['mean_sat'] = float(np.mean(s))
        feats['std_sat']  = float(np.std(s))
        feats['mean_val'] = float(np.mean(v))
        feats['std_val']  = float(np.std(v))

        gray_scaled = np.uint8(gray / 4)
        glcm = graycomatrix(
            gray_scaled, distances=[1], angles=[0],
            levels=64, symmetric=True, normed=True
        )
        feats['glcm_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])

        feats['entropy'] = float(shannon_entropy(gray))

        edges = cv2.Canny(gray, 100, 200)
        feats['edge_density'] = float(np.sum(edges > 0) / edges.size)

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            lc = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(lc)
            hull_area = cv2.contourArea(cv2.convexHull(lc))
            solidity = area / (hull_area + 1e-6)
        else:
            area = solidity = 0.0
        feats['contour_area'] = float(area)
        feats['solidity']     = float(solidity)

        mean_h = float(np.mean(h))
        feats['hue_sat_ratio'] = float(mean_h / (feats['mean_sat'] + 1e-6))

        # Delta & ratio features (optional background-based)
        if use_square_background:
            extras = self._compute_bg_delta_ratio(img_rgb)
            feats.update(extras)

        return feats

    def _compute_bg_delta_ratio(self, img_rgb: np.ndarray) -> dict:
        eps = 1e-6
        h, w = img_rgb.shape[:2]
        bw = int(min(h, w) * 0.1)

        bg_pixels = np.vstack([
            img_rgb[:bw, :, :].reshape(-1, 3),
            img_rgb[-bw:, :, :].reshape(-1, 3),
            img_rgb[:, :bw, :].reshape(-1, 3),
            img_rgb[:, -bw:, :].reshape(-1, 3),
        ])
        roi_pixels = img_rgb[bw:-bw, bw:-bw, :].reshape(-1, 3)

        feats = {}
        mean_bg  = bg_pixels.mean(axis=0)
        std_bg   = bg_pixels.std(axis=0)
        mean_roi = roi_pixels.mean(axis=0)
        std_roi  = roi_pixels.std(axis=0)

        # Z-score, SNR, and channel mean/std ratios
        for i, ch in enumerate(['r', 'g', 'b']):
            z_score = (mean_roi[i] - mean_bg[i]) / (std_bg[i] + eps)
            snr     = (mean_roi[i] - mean_bg[i]) / (std_bg[i] + eps)
            feats[f'z_{ch}']   = float(z_score)
            feats[f'snr_{ch}'] = float(snr)
            feats[f'mean_{ch}_ratio'] = float(mean_roi[i] / (mean_bg[i] + eps))
            feats[f'std_{ch}_ratio']  = float(std_roi[i]  / (std_bg[i]  + eps))

        hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        v    = hsv[:, :, 2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # KL divergence on Value histogram
        bg_v  = np.concatenate([
            v[:bw, :].flatten(), v[-bw:, :].flatten(),
            v[:, :bw].flatten(), v[:, -bw:].flatten()
        ])
        roi_v = v[bw:-bw, bw:-bw].flatten()
        hist_bg, _  = np.histogram(bg_v,  bins=16, range=(0,255), density=True)
        hist_roi, _ = np.histogram(roi_v, bins=16, range=(0,255), density=True)
        hist_bg += eps; hist_roi += eps
        kld_roi_bg = np.sum(hist_roi * np.log(hist_roi / hist_bg))
        kld_bg_roi = np.sum(hist_bg * np.log(hist_bg / hist_roi))
        feats['kl_divergence_val']   = float(kld_roi_bg)
        feats['kl_divergence_delta'] = float(kld_roi_bg - kld_bg_roi)
        feats['kl_divergence_ratio'] = float(kld_roi_bg / (kld_bg_roi + eps))

        # GLCM contrast delta/ratio
        gray_bg  = gray[:bw, :]
        gray_roi = gray[bw:-bw, bw:-bw]
        glcm_bg  = graycomatrix(gray_bg, distances=[1], angles=[0],
                                levels=256, symmetric=True, normed=True)
        glcm_roi = graycomatrix(gray_roi, distances=[1], angles=[0],
                                levels=256, symmetric=True, normed=True)
        cb = graycoprops(glcm_bg,  'contrast')[0,0]
        cr = graycoprops(glcm_roi, 'contrast')[0,0]
        feats['glcm_contrast_delta'] = float(cr - cb)
        feats['glcm_contrast_ratio'] = float(cr / (cb + eps))

        # Entropy delta/ratio
        eb = shannon_entropy(gray_bg)
        er = shannon_entropy(gray_roi)
        feats['entropy_delta'] = float(er - eb)
        feats['entropy_ratio'] = float(er / (eb + eps))

        # Edge density delta/ratio
        edges_bg  = cv2.Canny(gray_bg, 100, 200)
        edges_roi = cv2.Canny(gray_roi,100,200)
        ed_b = np.sum(edges_bg>0)/edges_bg.size
        ed_r = np.sum(edges_roi>0)/edges_roi.size
        feats['edge_density_delta'] = float(ed_r - ed_b)
        feats['edge_density_ratio'] = float(ed_r / (ed_b + eps))

        return feats

def getFeature(
        df_path: str = None,
        dir_path: str = None,
        out_path: str = None,
        use_square_background: bool = False
    ) -> None:
    """
    Trích xuất đặc trưng hồi quy từ ảnh theo metadata. Kết quả được lưu
    vào các file CSV riêng theo từng dòng máy (features_<phone>.csv) và
    một file tổng hợp (features_all.csv) không chứa thông tin điện thoại.
    """
    if df_path is None:
        df_path = META_COLORS
    df = pd.read_csv(df_path)

    if dir_path is None:
        dir_path = os.path.join(DATA_DIR, 'square image')
    if out_path is None:
        out_path = os.path.join(DATA_DIR, 'csv')

    os.makedirs(out_path, exist_ok=True)

    # giữ lại tên hiển thị và chuẩn hoá khóa
    df['PhoneDisplay'] = df['Phones']
    df['Phones_norm']  = df['Phones'].str.lower()

    extractor  = FeatureExtractor()
    phone_dirs = os.listdir(dir_path)

    # tổng hợp đặc trưng toàn bộ thiết bị
    all_feats = []

    for norm in df['Phones_norm'].unique():
        display       = df.loc[df['Phones_norm']==norm, 'PhoneDisplay'].iloc[0]
        matches       = [d for d in phone_dirs if d.lower()==norm]
        folder        = matches[0] if matches else display
        phone_path    = os.path.join(dir_path, folder)
        category_dirs = os.listdir(phone_path)

        df_phone   = df[df['Phones_norm']==norm].sort_values(by='Types')
        feats_list = []
        total      = len(df_phone)

        with tqdm(total=total, desc=f"Extracting Features [{display}]", unit="image") as pbar:
            for _, row in df_phone.iterrows():
                raw_cat = row['Types']
                ppm     = row['ppm']
                filename= row['Filename']

                # tìm thư mục chứa ảnh theo category
                cands = [c for c in category_dirs if raw_cat.lower() in c.lower()]
                if not cands:
                    pbar.update(1)
                    continue
                sorted_c = [c for c in cands if 'sorted' in c.lower()]
                category_folder = sorted_c[0] if sorted_c else cands[0]

                base = os.path.join(phone_path, category_folder)
                img_path = os.path.join(base, filename)

                # fallback nếu thiếu đuôi
                if not os.path.exists(img_path):
                    root, ext = os.path.splitext(filename)
                    if ext == '':
                        for e in IMAGE_EXTENSIONS:
                            cand = os.path.join(base, root + e)
                            if os.path.exists(cand):
                                img_path = cand
                                break

                if not os.path.exists(img_path):
                    pbar.update(1)
                    continue

                feats = extractor.extract_features(
                    img_path,
                    use_square_background=use_square_background,
                    id_img=filename,
                    ppm=ppm
                )
                if feats:
                    feats_list.append(feats)
                pbar.update(1)

        # lưu file đặc trưng riêng theo điện thoại
        df_feats = pd.DataFrame(feats_list)
        if not df_feats.empty:
            df_feats.to_csv(
                os.path.join(out_path, f'features_{display}.csv'),
                index=False
            )
            all_feats.extend(feats_list)
        else:
            print(f"[Skipping] No features extracted for phone '{display}', CSV not written.")

    # cuối cùng, ghi file tổng hợp features_all.csv (không chứa cột phone)
    if all_feats:
        df_all = pd.DataFrame(all_feats)
        df_all.to_csv(
            os.path.join(out_path, 'features_all.csv'),
            index=False
        )

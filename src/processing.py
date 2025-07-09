import os
import cv2
import pandas as pd
from tqdm import tqdm

from config import DATA_DIR, META_COLORS
from loading import load_data
from roi import runROI

def process_data(
    data_dir: str = os.path.join(DATA_DIR, 'full', 'HP5_data'),
    meta_path: str = None,
) -> None:
    """
    1) Load metadata (Filename, Types, Phones, Repeat)
    2) Map cleaned Types to actual chemical folders
    3) For each (chem, phone), process all images:
       - Build input path from actual folder
       - Run ROI, save outputs into subfolders phone/types
       - Show progress
    4) Log failures
    5) If there are new failures, write a cleaned metadata file *_cleaned.csv
       instead of overwriting the original.
    """
    # 1) Read old failures
    failed_csv = os.path.join(data_dir, 'failed_images.csv')
    if os.path.exists(failed_csv):
        lst_failed = pd.read_csv(failed_csv)[['Failed Images','Error']].apply(tuple, axis=1).tolist()
    else:
        lst_failed = []

    # 1) Load metadata
    df, lst_failed = load_data(df_path=meta_path, lst_failed=lst_failed)

    # Base output dirs
    base_square = os.path.join(DATA_DIR, 'square image')
    base_roi    = os.path.join(DATA_DIR, 'roi')
    base_bg     = os.path.join(DATA_DIR, 'background')

    # 2) Build Types mapping: cleaned -> actual folder
    real_types = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    type_map = {d.lower().replace('sorted', '').strip('_'): d for d in real_types}

    lst_failed_local = []
    # 3) Iterate by chemical and phone
    for chem, df_chem in df.groupby('Types'):
        # Determine actual folder name
        key = chem.lower().replace('sorted', '').strip('_')
        folder_chem = type_map.get(key)
        if folder_chem is None:
            print(f"[Warning] No folder match for Types='{chem}'")
            continue
        print(f"\n--- Processing chemical: {chem} (folder: {folder_chem}) ---")

        for phone, df_phone in df_chem.groupby('Phones'):
            total = len(df_phone)
            print(f"Phone: {phone} | Total images: {total}")
            # Prepare output subdirs
            sq_dir  = os.path.join(base_square, phone, chem)
            roi_dir = os.path.join(base_roi,    phone, chem)
            bg_dir  = os.path.join(base_bg,     phone, chem)
            for d in (sq_dir, roi_dir, bg_dir):
                os.makedirs(d, exist_ok=True)

            success = 0
            with tqdm(total=total, desc=f"{chem}/{phone}", unit="img") as pbar:
                for row in df_phone.itertuples(index=False):
                    img_path = os.path.join(
                        data_dir,
                        folder_chem,
                        row.Phones,
                        row.Repeat,
                        row.Filename
                    )
                    basename = row.Filename
                    if not os.path.exists(img_path):
                        lst_failed_local.append((basename, 'Not found'))
                        pbar.update(1)
                        continue
                    try:
                        image = cv2.imread(img_path)
                        squared, sample, background, _ = runROI(image=image, kit=None)
                        cv2.imwrite(os.path.join(sq_dir,  basename), squared)
                        cv2.imwrite(os.path.join(roi_dir, basename), sample)
                        cv2.imwrite(os.path.join(bg_dir,  basename), background)
                        success += 1
                    except Exception as e:
                        lst_failed_local.append((basename, str(e)))
                    pbar.update(1)

            print(f">>> {chem} | {phone}: {success}/{total} images processed successfully.")

    # 4) Write combined failure log
    all_failed = set(lst_failed + lst_failed_local)
    df_failed = pd.DataFrame(list(all_failed), columns=['Failed Images', 'Error'])
    df_failed.to_csv(failed_csv, index=False)

    # 5) Update metadata only if new failures occurred
    if lst_failed_local:
        cleaned_meta = os.path.splitext(meta_path)[0] + '_cleaned.csv'
        failed_ids = df_failed['Failed Images'].apply(lambda x: x.split('_')[0]).tolist()
        cleaned_df = df[~df['Id_imgs'].astype(str).isin(failed_ids)]
        cleaned_df.to_csv(cleaned_meta, index=False)
        print(f"Metadata updated; cleaned file written to: {cleaned_meta}")
    else:
        print("No new failures; original metadata left unchanged.")

if __name__ == "__main__":
    process_data()

import os
import pandas as pd
from api_service.config import DATA_DIR, META_COLORS, IMAGE_EXTENSIONS

def create_meta_data(
    data_dir: str = None,
    out_dir: str = None,
    cols: list = None,
    lst_failed: list = None,
) -> list:
    """
    Recursively scan:
      data_dir/
        <chemical>/
          <phone>/
            <repeat>/
              <files>.jpg

    Build metadata rows with columns:
      [Id_imgs, Types, ppm, Phones, Repeat, Date, Filename]
    for every image, then write to out_dir.
    Returns list of failures.
    """
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, 'full', 'HP5_data')
    if out_dir is None:
        out_dir = META_COLORS
    if cols is None:
        cols = ['Id_imgs', 'Types', 'ppm', 'Phones', 'Repeat', 'Date', 'Filename']
    if lst_failed is None:
        lst_failed = []

    rows = []
    for chem in sorted(os.listdir(data_dir)):
        chem_path = os.path.join(data_dir, chem)
        if not os.path.isdir(chem_path):
            continue

        for phone in sorted(os.listdir(chem_path)):
            phone_path = os.path.join(chem_path, phone)
            if not os.path.isdir(phone_path):
                continue

            for repeat in sorted(os.listdir(phone_path)):
                rep_path = os.path.join(phone_path, repeat)
                if not os.path.isdir(rep_path):
                    continue

                for filename in sorted(os.listdir(rep_path)):
                    name_l = filename.lower()
                    if not any(name_l.endswith(ext) for ext in IMAGE_EXTENSIONS):
                        continue
                    if 'ppm' not in name_l:
                        continue
                    try:
                        parts = filename.split('_')
                        id_img   = parts[0]
                        ppm_part = next(p for p in parts if 'ppm' in p)
                        ppm_val  = float(ppm_part.replace('ppm', '').replace(',', '.'))
                        date_raw = parts[2]
                        rows.append([id_img, chem, ppm_val, phone, repeat, date_raw, filename])
                    except Exception as e:
                        err = (filename, str(e))
                        if err not in lst_failed:
                            lst_failed.append(err)

    df = pd.DataFrame(rows, columns=cols)
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(out_dir, index=False)
    return lst_failed


def load_data(
    df_path: str = None,
    lst_failed: list = None
) -> tuple[pd.DataFrame, list]:
    """
    Load metadata CSV (with Filename column).
    Also returns updated lst_failed.
    """
    if df_path is None:
        df_path = META_COLORS
    if lst_failed is None:
        lst_failed = []

    df = pd.read_csv(df_path)
    expected = {'Id_imgs', 'Types', 'ppm', 'Phones', 'Repeat', 'Date', 'Filename'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")

    return df, lst_failed

if __name__ == "__main__":
    create_meta_data(
        data_dir=os.path.join(DATA_DIR, 'full', 'HP5_data'),
        out_dir=META_COLORS
    )
    print(f"Metadata generated at {META_COLORS}")

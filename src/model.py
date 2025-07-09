import os
import glob
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import DATA_DIR, META_COLORS


def train_regressors(
    meta_path: str = None,
    dir_path: str = None,
    out_path: str = None,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42
) -> None:
    """
    Train RandomForest and XGBoost regressors for each phone listed in metadata.

    Splits: test=test_size of full, val=val_size of remaining.
    Evaluates on val/test sets and saves models + scalers.
    """
    if dir_path is None:
        dir_path = os.path.join(DATA_DIR, 'csv')
    if meta_path is None:
        meta_path = META_COLORS
    if out_path is None:
        out_path = os.path.join(DATA_DIR, 'models')
    os.makedirs(out_path, exist_ok=True)

    df_meta = pd.read_csv(meta_path)
    phones = df_meta['Phones'].astype(str).unique()

    for phone in phones:
        # find regression CSV case-insensitive
        pattern = os.path.join(dir_path, f"rgs_*{phone.lower()}*.csv")
        matches = glob.glob(pattern)
        if not matches:
            print(f"[Skipping] No CSV for phone '{phone}' (pattern: {pattern})")
            continue
        reg_path = matches[0]
        df = pd.read_csv(reg_path)

        # prepare data
        X = df.drop(columns=['id_img', 'ppm'], errors='ignore')
        y = df['ppm']

        # split
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size, random_state=random_state
        )

        # scale
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        # init models
        rf = RandomForestRegressor(random_state=random_state)
        xgb = XGBRegressor(
            objective='reg:squarederror', random_state=random_state
        )

        # train
        rf.fit(X_train_s, y_train)
        xgb.fit(X_train_s, y_train)

        # eval
        def eval_model(m, Xs, ys, split):
            preds = m.predict(Xs)
            mse = mean_squared_error(ys, preds)
            mae = mean_absolute_error(ys, preds)
            r2  = r2_score(ys, preds)
            print(f"[{phone}] {m.__class__.__name__} {split} -> MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        print(f"\n--- Evaluating {phone} ---")
        for model in (rf, xgb):
            eval_model(model, X_val_s, y_val, 'Validation')
            eval_model(model, X_test_s, y_test, 'Test')

        # save
        joblib.dump(scaler, os.path.join(out_path, f"regressor_scaler_{phone}.pkl"))
        joblib.dump(rf,     os.path.join(out_path, f"regressor_rf_{phone}.pkl"))
        joblib.dump(xgb,    os.path.join(out_path, f"regressor_xgb_{phone}.pkl"))

if __name__ == '__main__':
    train_regressors()

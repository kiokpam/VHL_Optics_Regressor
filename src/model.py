import os
import glob
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import DATA_DIR, META_COLORS

def train_both_pipelines(
    meta_path=None,
    dir_path=None,
    out_path=None,
    test_size=0.2,
    val_size: float = None,
    n_splits=5,
    random_state=42
):
    dir_path  = dir_path or os.path.join(DATA_DIR, 'csv')
    meta_path = meta_path or META_COLORS
    out_path  = out_path or os.path.join(DATA_DIR, 'models')
    os.makedirs(out_path, exist_ok=True)

    df_meta = pd.read_csv(meta_path)
    df_meta['norm'] = df_meta['Phones'].str.lower()
    norms = df_meta['norm'].unique()
    phone_map = {n: df_meta.loc[df_meta['norm']==n, 'Phones'].iloc[0] for n in norms}

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for norm in norms:
        display = phone_map[norm]
        pattern = os.path.join(dir_path, f"features_*{norm}*.csv")
        files = glob.glob(pattern)
        if not files:
            print(f"[Skipping] no data for {display}")
            continue

        df = pd.read_csv(files[0])
        if df.empty or 'ppm' not in df.columns:
            print(f"[Skipping] invalid CSV for {display}")
            continue

        X = df.drop(columns=['id_img','ppm'], errors='ignore')
        y = df['ppm']

        # 1) Tách test set
        X_tr_full, X_test, y_tr_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 2) Common preprocessing steps
        select = SelectKBest(score_func=f_regression)
        scale  = StandardScaler()

        # --- RandomForest pipeline & grid ---
        pipe_rf = Pipeline([
            ("select", select),
            ("scale", scale),
            ("model", RandomForestRegressor(random_state=random_state))
        ])
        rf_param_grid = {
            "select__k": [10, 20, 30, "all"],
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }
        grid_rf = GridSearchCV(
            pipe_rf,
            rf_param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_rf.fit(X_tr_full, y_tr_full)
        best_rf_pipe = grid_rf.best_estimator_
        print(f"[{display}] RF best params: {grid_rf.best_params_}")

        # --- XGB pipeline & grid ---
        pipe_xgb = Pipeline([
            ("select", select),
            ("scale", scale),
            ("model", XGBRegressor(objective='reg:squarederror', random_state=random_state))
        ])
        xgb_param_grid = {
            "select__k": [10, 20, 30, "all"],
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 6, 10],
            "model__learning_rate": [0.01, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }
        grid_xgb = GridSearchCV(
            pipe_xgb,
            xgb_param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_xgb.fit(X_tr_full, y_tr_full)
        best_xgb_pipe = grid_xgb.best_estimator_
        print(f"[{display}] XGB best params: {grid_xgb.best_params_}")

        # 3) Đánh giá final trên test set
        for name, model in [("RF", best_rf_pipe), ("XGB", best_xgb_pipe)]:
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2  = r2_score(y_test, preds)
            print(f"[{display}] {name} Test -> MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # 4) Lưu hai pipeline
        joblib.dump(best_rf_pipe,  os.path.join(out_path, f"pipe_rf_{display}.pkl"))
        joblib.dump(best_xgb_pipe, os.path.join(out_path, f"pipe_xgb_{display}.pkl"))


def train_general_model(
    features_path: str = None,
    out_path: str = None,
    test_size: float = 0.2,
    n_splits: int = 3,
    random_state: int = 42
) -> None:
    """
    Train a single regression model across all phones using the combined features CSV.
    The features CSV should include a categorical column 'phone_norm' if available.
    Saves the global XGB model as pipe_xgb_general.pkl.
    """
    # Default paths
    if features_path is None:
        features_path = os.path.join(DATA_DIR, 'csv', 'features_all.csv')
    if out_path is None:
        out_path = os.path.join(DATA_DIR, 'models')
    os.makedirs(out_path, exist_ok=True)

    # Load combined features
    df = pd.read_csv(features_path)
    if df.empty or 'ppm' not in df.columns:
        raise ValueError("Features CSV must contain 'ppm' column and not be empty")

    # Separate target and features
    y = df['ppm']
    # Drop identifier columns and phone column if present
    drop_cols = ['ppm', 'id_img', 'phone_norm']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # For the general model, all remaining features are numeric
    cat_features = []
    num_features = list(X.columns)

    # Build transformers
    numeric_transformer = Pipeline([
        ('select', SelectKBest(score_func=f_regression)),
        ('scale', StandardScaler())
    ])

    transformers = []
    if num_features:
        transformers.append(('num', numeric_transformer, num_features))
    if cat_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_features))
    preprocessor = ColumnTransformer(transformers)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Cross-validation strategy with 3 folds
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Note: Removed the RandomForest training for the general model to speed up
    # training. Only the XGBoost model is trained below.

    # ----- XGB pipeline & grid -----
    pipe_xgb = Pipeline([
        ('preprocess', preprocessor),
        ('model', XGBRegressor(objective='reg:squarederror', random_state=random_state))
    ])
    xgb_param_grid = {
        'preprocess__num__select__k': [10, 20, 30, 'all'],
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 6, 10],
        'model__learning_rate': [0.01, 0.1],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0]
    }
    grid_xgb = GridSearchCV(
        pipe_xgb,
        xgb_param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_xgb.fit(X_train, y_train)
    best_xgb = grid_xgb.best_estimator_
    print(f"[General] XGB best params: {grid_xgb.best_params_}")

    # Evaluate only the XGB model on the test set
    preds = best_xgb.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    print(f"[General] XGB Test -> MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Save only the XGB pipeline
    joblib.dump(best_xgb, os.path.join(out_path, 'pipe_xgb_general.pkl'))



if __name__ == '__main__':
    train_both_pipelines()

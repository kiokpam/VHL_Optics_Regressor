import json
import numpy as np
import optuna
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore', message='.*')

import config


def tune_regressor(
    X, y,
    model_type=None,
    use_random_search=True,
    random_search_iter=30,
    early_stopping_rounds=None,
    val_fraction=0.1
):
    """
    Tune and return a trained regressor for given data X, y, with optional early stopping.
    Returns: best_estimator, best_params
    """
    rt = model_type or config.REGRESSOR_TYPE
    hp_grid = config.HYPERPARAMS.get(rt, {}).copy()

    # Fix invalid 'auto' for RandomForest max_features
    if rt == 'rf' and 'max_features' in hp_grid:
        hp_grid['max_features'] = ['sqrt' if v == 'auto' else v for v in hp_grid['max_features']]

    # Prepare train/val split for early stopping
    X_train, y_train = X, y
    fit_kwargs = {}
    if early_stopping_rounds and rt in ('xgb', 'lgbm'):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_fraction, random_state=42
        )
        fit_kwargs = {
            'eval_set': [(X_val, y_val)],
            'early_stopping_rounds': early_stopping_rounds,
            'verbose': False
        }

    # Instantiate base estimator
    if rt == 'rf':
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
    elif rt == 'xgb':
        base = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror',
            n_estimators=hp_grid.get('n_estimators', 500),
            verbosity=0
        )
    elif rt == 'lgbm':
        base = LGBMRegressor(
            random_state=42,
            n_jobs=-1,
            n_estimators=hp_grid.get('n_estimators', 500),
            min_child_samples=hp_grid.get('min_child_samples', 5),
            min_split_gain=hp_grid.get('min_split_gain', 0.0),
            verbosity=-1
        )
    elif rt == 'cat':
        base = CatBoostRegressor(random_state=42, verbose=0)
    elif rt == 'stack':
        estimators = []
        for m in config.HYPERPARAMS['stack']['base_models']:
            if m == 'rf':
                est = RandomForestRegressor(random_state=42, n_jobs=-1)
            elif m == 'xgb':
                est = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror', verbosity=0)
            elif m == 'lgbm':
                est = LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
            else:
                continue
            estimators.append((m, est))
        base = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), n_jobs=-1)
        hp_grid = {}
    else:
        raise ValueError(f"Unknown model_type {rt}")

    # Determine search strategy
    if hp_grid:
        sizes = [len(v) for v in hp_grid.values()]
        grid_size = int(np.prod(sizes)) if sizes else 0
        if use_random_search and grid_size > random_search_iter:
            search = RandomizedSearchCV(
                base, hp_grid, n_iter=random_search_iter, cv=5,
                n_jobs=-1, random_state=42, error_score='raise'
            )
        else:
            search = GridSearchCV(base, hp_grid, cv=5, n_jobs=-1, error_score='raise')
        search.fit(X_train, y_train, **fit_kwargs)
        best = search.best_estimator_
        best_params = search.best_params_
    else:
        base.fit(X_train, y_train, **fit_kwargs)
        best = base
        best_params = {}

    return best, best_params


def tune_regressor_optuna(
    X, y,
    model_type=None,
    n_trials=50,
    early_stopping_rounds=50,
    val_fraction=0.2
):
    """
    Tune and return regressor using Optuna Bayesian optimization.
    Returns: final_model, best_params
    """
    rt = model_type or config.REGRESSOR_TYPE
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=42
    )

    def objective(trial):
        if rt == 'xgb':
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [200, 500, 1000]),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
            }
            model = XGBRegressor(**params, random_state=42, verbosity=0, objective='reg:squarederror')
        elif rt == 'lgbm':
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [200, 500, 1000]),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'min_split_gain': trial.suggest_loguniform('min_split_gain', 1e-8, 1.0)
            }
            model = LGBMRegressor(**params, random_state=42, verbosity=-1)
        elif rt == 'cat':
            params = {
                'iterations': trial.suggest_categorical('iterations', [200, 500, 1000]),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0)
            }
            model = CatBoostRegressor(**params, random_state=42, verbose=0)
        else:
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 300, 500]),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)] if rt in ('xgb', 'lgbm', 'cat') else None,
            early_stopping_rounds=early_stopping_rounds if rt in ('xgb', 'lgbm', 'cat') else None,
            verbose=False
        )
        preds = model.predict(X_val)
        return np.corrcoef(y_val, preds)[0, 1]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    # Train final model on full data
    if rt == 'xgb':
        final = XGBRegressor(**best_params, random_state=42, verbosity=0, objective='reg:squarederror')
    elif rt == 'lgbm':
        final = LGBMRegressor(**best_params, random_state=42, verbosity=-1)
    elif rt == 'cat':
        final = CatBoostRegressor(**best_params, random_state=42, verbose=0)
    else:
        final = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    final.fit(X, y)
    return final, best_params

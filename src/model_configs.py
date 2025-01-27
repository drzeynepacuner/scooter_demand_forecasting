## MODIFY MODEL CONFIGS - LARGER RANGE AVAILABLE

from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    xgboost_installed = True
except ImportError:
    xgboost_installed = False

def get_model_candidates():
    """
    Returns a list of dicts describing model classes + param grids.
    """
    candidates = []

    # 2.1 RandomForestRegressor candidate
    rf_param_grid = [
    # Each dict is one combination of hyperparams
    {"max_depth": None, "min_samples_leaf": 1, "max_features": "log2"},
    {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"max_depth": 5, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"max_depth": 5, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"max_depth": 10, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"max_depth": 10, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"max_depth": 15, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"max_depth": 15, "min_samples_leaf": 5, "max_features": "sqrt"},
    ]

    candidates.append({
    "name": "RandomForest",
    "estimator": RandomForestRegressor,  # class (not instance)
    "param_grid": rf_param_grid
    })

    # 2.2 XGBRegressor candidate (if installed)
    if xgboost_installed:
        from xgboost import XGBRegressor
        xgb_param_grid = [
            {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
            {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
            {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
            {
                "n_estimators": 300,
                "max_depth": 7,
                "learning_rate": 0.01,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
        ]

        candidates.append({
            "name": "XGBoost",
            "estimator": XGBRegressor,
            "param_grid": xgb_param_grid
        })
        return candidates

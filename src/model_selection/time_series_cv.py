import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from ..metrics import rmse, mae, mape

def evaluate_candidate(data, features, model_class, param_dict, n_splits=3):
    """
    Perform TSCV for one model + param combo. Return avg RMSE, MAE, MAPE.
    """
    X = data[features]
    y = data["demand"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes_, mapes_ = [], [], []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(random_state=42, **param_dict)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmses.append(rmse(y_val, preds))
        maes_.append(mae(y_val, preds))
        mapes_.append(mape(y_val, preds))

    return np.mean(rmses), np.mean(maes_), np.mean(mapes_)

def evaluate_model_candidates(data, features, model_candidates, n_splits=3):
    """
    Check all (model, param) combos. Return best by RMSE + associated MAE, MAPE.
    """
    best_rmse = float('inf')
    best_mae_ = None
    best_mape_ = None
    best_model = None
    best_params = None

    for mc in model_candidates:
        model_name = mc["name"]
        cls_ = mc["estimator"]
        grid = mc["param_grid"]

        for params in grid:
            avg_rmse, avg_mae, avg_mape_ = evaluate_candidate(data, features, cls_, params, n_splits)
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_mae_ = avg_mae
                best_mape_ = avg_mape_
                best_model = (model_name, cls_)
                best_params = params

    return best_rmse, best_mae_, best_mape_, best_model, best_params

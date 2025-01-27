import os
import numpy as np
import pandas as pd
from joblib import dump

from src.data_preparation import load_and_prepare_data
from src.model_configs import get_model_candidates
from src.model_selection.forward_selection import forward_feature_selection

from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from src.metrics import mean_absolute_percentage_error
except ImportError:
    from src.metrics import mape as mean_absolute_percentage_error


def train_pipeline(rides_csv, weather_csv, test_cutoff_date, model_out_path="final_model.joblib"):
    """
    1. Load & prepare data from CSVs (including raw demand).
    2. Create log_demand but keep the original 'demand' for final evaluation.
    3. Overwrite 'demand' column with 'log_demand' so forward_selection sees log space.
    4. Split final test set by 'date'.
    5. Run forward feature selection + TSCV on log(demand).
    6. Retrain final model on the entire train+val portion (log(demand)).
    7. Evaluate on test in the original domain (via expm1).
    8. Save final model.
    """

    # ------------------------------------------------------------------
    # 1) LOAD & PREPARE DATA (with original 'demand')
    # ------------------------------------------------------------------
    df = load_and_prepare_data(rides_csv, weather_csv)

    # Convert 'date' to datetime for splitting
    df["date"] = pd.to_datetime(df["date"])

    # Store original demand in a safe column for final evaluation
    df["orig_demand"] = df["demand"].copy()

    # ------------------------------------------------------------------
    # 2) CREATE LOG-DEMAND
    # ------------------------------------------------------------------
    df["log_demand"] = np.log1p(df["demand"])

    # ------------------------------------------------------------------
    # 3) Overwrite 'demand' with 'log_demand'
    #    so all references to data['demand'] use log(demand)
    # ------------------------------------------------------------------
    df.drop(columns=["demand"], inplace=True)
    df.rename(columns={"log_demand": "demand"}, inplace=True)

    # Now data['demand'] is log(demand).
    # Meanwhile, data['orig_demand'] is the original demand.

    # ------------------------------------------------------------------
    # 4) SPLIT FINAL TEST SET
    # ------------------------------------------------------------------
    train_val_df = df[df["date"] < test_cutoff_date].copy()
    test_df      = df[df["date"] >= test_cutoff_date].copy()

    # Features for the model (excluding the target 'demand' and 'orig_demand')
    all_feats = [
        "demand_lag_1",    # note: this is still lag of the raw demand from data_preparation
        "day_of_week",
        "is_weekend",
        "temperature",
        "max_temperature",
        "min_temperature",
        "precipitation"
    ]

    # ------------------------------------------------------------------
    # 5) FORWARD FEATURE SELECTION (on log(demand))
    # ------------------------------------------------------------------
    model_candidates = get_model_candidates()
    (
        best_features,
        best_rmse_log,
        best_mae_log,
        best_mape_log,
        best_model_info,
        best_params
    ) = forward_feature_selection(train_val_df, all_feats, model_candidates, n_splits=3)

    print("\n=== FINAL SELECTION (IN LOG SPACE) ===")
    print("Features:", best_features)
    print(f"CV => RMSE(log): {best_rmse_log:.3f}, MAE(log): {best_mae_log:.3f}, MAPE(log): {best_mape_log:.2f}%")
    print("Best model:", best_model_info[0])
    print("Hyperparams:", best_params)

    # ------------------------------------------------------------------
    # 6) RETRAIN FINAL MODEL ON THE ENTIRE train+val portion (log(demand))
    # ------------------------------------------------------------------
    X_trainval = train_val_df[best_features]
    y_trainval = train_val_df["demand"]   # log(demand)

    final_model_class = best_model_info[1]
    final_model = final_model_class(random_state=42, **best_params)
    final_model.fit(X_trainval, y_trainval)

    # ------------------------------------------------------------------
    # 7) EVALUATE ON TEST SET IN ORIGINAL DOMAIN
    # ------------------------------------------------------------------
    X_test = test_df[best_features]

    # Predictions in log space
    preds_log = final_model.predict(X_test)
    # Convert back to linear domain
    preds_linear = np.expm1(preds_log)

    # Actual original demand
    y_test_linear = test_df["orig_demand"].values

    # Evaluate in real domain
    test_rmse = mean_squared_error(y_test_linear, preds_linear, squared=False)
    test_mae  = mean_absolute_error(y_test_linear, preds_linear)
    test_mape = mean_absolute_percentage_error(y_test_linear, preds_linear)

    print("\n=== TEST RESULTS (ORIGINAL DEMAND SPACE) ===")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Test MAE : {test_mae:.3f}")
    print(f"Test MAPE: {test_mape:.2f}%")

    # ------------------------------------------------------------------
    # 8) SAVE FINAL MODEL
    # ------------------------------------------------------------------
    dump(final_model, model_out_path)
    print(f"Model saved to {model_out_path}")


# Optional: CLI wrapper
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rides_csv", type=str, required=True)
    parser.add_argument("--weather_csv", type=str, required=True)
    parser.add_argument("--test_cutoff_date", type=str, default="2020-08-25")
    parser.add_argument("--model_out_path", type=str, default="final_model.joblib")

    args = parser.parse_args()
    train_pipeline(
        rides_csv=args.rides_csv,
        weather_csv=args.weather_csv,
        test_cutoff_date=args.test_cutoff_date,
        model_out_path=args.model_out_path
    )

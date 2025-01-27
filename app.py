import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

USE_LOG_MODEL = True

# Corrected FEATURES list with commas
FEATURES = [
    "day_of_week",
    "precipitation",
    "demand_lag_1",
]

MODEL_PATH = os.environ.get("MODEL_PATH", "final_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


def preprocess_inference_auto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the preprocessing steps from the training pipeline:
    - Computes day_of_week and is_weekend from start_time.
    - Selects the required FEATURES for prediction.
    - Ensures that the DataFrame has all necessary columns.
    """
    # Parse 'start_time' to datetime
    if "start_time" not in df.columns:
        raise ValueError("Missing 'start_time' in input to compute day_of_week.")

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    if df["start_time"].isnull().any():
        raise ValueError("Some 'start_time' entries could not be parsed as datetime.")

    # Compute 'day_of_week' (0=Monday, 6=Sunday)
    df["day_of_week"] = df["start_time"].dt.dayofweek

    # Compute 'is_weekend' (1 if Saturday or Sunday, else 0)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Check for required weather columns
    required_weather_cols = ["temperature", "max_temperature", "min_temperature", "precipitation"]
    missing_weather_cols = set(required_weather_cols) - set(df.columns)
    if missing_weather_cols:
        raise ValueError(f"Missing weather columns: {missing_weather_cols}")

    # Ensure 'demand_lag_1' is provided
    if "demand_lag_1" not in df.columns:
        raise ValueError("Missing 'demand_lag_1' in input data.")

    # Select the FEATURES required by the model
    # 'is_weekend' is computed but not used; included here for potential future use
    selected_features = FEATURES  # ['day_of_week', 'precipitation', 'demand_lag_1']

    # Verify all selected features are present
    missing_features = set(selected_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features after preprocessing: {missing_features}")

    # Reorder the DataFrame to match FEATURES
    df_prepared = df[selected_features].copy()

    return df_prepared


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "API is alive"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a JSON array of objects, each containing:
    - start_time (e.g., "2020-09-12 14:35:00")
    - temperature
    - max_temperature
    - min_temperature
    - precipitation
    - demand_lag_1

    Example:
    [
      {
        "start_time": "2020-09-12 14:35:00",
        "temperature": 17.3,
        "max_temperature": 21.0,
        "min_temperature": 14.2,
        "precipitation": 0.5,
        "demand_lag_1": 1
      }
    ]

    Returns:
    {
      "predictions": [42.31]
    }
    """
    try:
        # Parse JSON input
        data_json = request.get_json(force=True)
        if not isinstance(data_json, list):
            return jsonify({"error": "Input must be a JSON list"}), 400

        # Convert to DataFrame
        df_input = pd.DataFrame(data_json)

        # Preprocess the input to match training features
        df_prepared = preprocess_inference_auto(df_input)

        # Predict using the model
        preds_log_or_linear = model.predict(df_prepared)

        if USE_LOG_MODEL:
            # Invert log transform to get actual demand
            preds_real = np.expm1(preds_log_or_linear)
            preds = [float(p) for p in preds_real]
        else:
            preds = [float(p) for p in preds_log_or_linear]

        return jsonify({"predictions": preds}), 200

    except ValueError as ve:
        # Handle known errors gracefully
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

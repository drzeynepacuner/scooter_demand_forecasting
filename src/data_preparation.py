import pandas as pd
import numpy as np  # Added for log transform
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the dynamic path for ../input
input_dir = os.path.join(current_dir, '..', 'input')

def load_and_prepare_data(rides_csv: str, weather_csv: str):
    """
    Reads CSVs, merges them, creates daily demand & features like lag, day_of_week, etc.
    Returns a DataFrame with columns:
      [date, demand, demand_lag_1, day_of_week, is_weekend,
       temperature, max_temperature, min_temperature, precipitation, log_demand]
    """
    rides_df = pd.read_csv(os.path.join(input_dir, 'voiholm.csv'), parse_dates=["start_time"])
    weather_df = pd.read_csv(os.path.join(input_dir, 'weather_data.csv'), parse_dates=["date"])

    # Daily demand
    rides_df["date"] = rides_df["start_time"].dt.date
    daily_demand = rides_df.groupby("date").agg({"ride_id": "count"}).reset_index()
    daily_demand.rename(columns={"ride_id": "demand"}, inplace=True)

    weather_df["date"] = weather_df["date"].dt.date

    # Merge
    merged = pd.merge(daily_demand, weather_df, on="date", how="left")

    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged.sort_values(by="date_dt", inplace=True)

    # day_of_week, is_weekend
    merged["day_of_week"] = merged["date_dt"].dt.dayofweek
    merged["is_weekend"] = (merged["day_of_week"] >= 5).astype(int)

    # 1-day lag of raw demand
    merged["demand_lag_1"] = merged["demand"].shift(1)
    merged.dropna(subset=["demand_lag_1"], inplace=True)

    # Final columns
    final_cols = [
        "date",
        "demand",
        "demand_lag_1",
        "day_of_week",
        "is_weekend",
        "temperature",
        "max_temperature",
        "min_temperature",
        "precipitation"
    ]
    final_df = merged[final_cols].copy()

    # >>> Add log transform of 'demand' <<<
    # log1p(demand) helps avoid issues if demand=0 on some days.
    final_df["log_demand"] = np.log1p(final_df["demand"])

    return final_df

# src/forecasting.py
import pandas as pd
import joblib
import os
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_lag_features


def forecast_future_load(df, model, steps=10, num_lags=5):
    df = df.copy()
    df = create_lag_features(df, target_col='LOAD', num_lags=num_lags)
    last_row = df.iloc[-1].copy()

    predictions = []

    for _ in range(steps):
        # Prepare lag feature input
        lag_features = [last_row[f'LOAD_t-{i}'] for i in range(1, num_lags + 1)]
        features = pd.DataFrame([[
            last_row['P_IN'], last_row['T_IN'],
            last_row['P_OUT'], last_row['T_OUT'],
            *lag_features
        ]], columns=['P_IN', 'T_IN', 'P_OUT', 'T_OUT'] + [f'LOAD_t-{i}' for i in range(1, num_lags + 1)])

        # Predict next load
        next_load = model.predict(features)[0]
        predictions.append(next_load)

        # Shift lags and insert new prediction
        for i in range(num_lags, 1, -1):
            last_row[f'LOAD_t-{i}'] = last_row[f'LOAD_t-{i - 1}']
        last_row['LOAD_t-1'] = next_load

    return predictions


if __name__ == '__main__':
    # Load model and data
    model = joblib.load('../model.pkl')
    df = load_and_clean_data('../data/combinedddddd_dataset.xlsx')

    # Forecast next 10 steps
    forecast = forecast_future_load(df, model, steps=10, num_lags=5)

    print("Next 10 predicted LOAD values:")
    for i, val in enumerate(forecast, 1):
        print(f"Step {i}: {val:.2f}")

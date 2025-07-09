import os
import joblib
import pandas as pd
from src.data_preprocessing import load_and_clean_data


def load_model(model_path='arima_model.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_next(df, model, target_col='LOAD'):
    # Ensure series is sorted
    df = df.sort_values('DATE')
    series = df[target_col]

    # Forecast the next step
    forecast = model.forecast(steps=1)
    return forecast[0]


def forecast_from_manual_input(model, input_data, steps=1):
    """
    Forecast using ARIMA model based on manual input.
    Assumes manual LOAD value is provided, but ARIMA uses internal state.
    """
    last_load = input_data.get('LOAD', None)
    if last_load is None:
        raise ValueError("Manual input must include a 'LOAD' value.")

    # Optionally, you could use last_load to override internal state or retrain ARIMA,
    # but here we'll use existing model as-is to generate forecast.
    forecast = model.forecast(steps=steps)
    return forecast.tolist()


if __name__ == '__main__':
    model = load_model('../arima_model.pkl')
    df = load_and_clean_data('../data/combinedddddd_dataset.xlsx')

    # Predict next step using dataset
    prediction = predict_next(df, model)
    print(f"Predicted next LOAD value: {prediction:.4f}")

    # Predict from manual input
    manual_input = {'LOAD': df['LOAD'].iloc[-1]}  # example: last value
    future = forecast_from_manual_input(model, manual_input, steps=3)
    print("Forecast from manual input:", future)

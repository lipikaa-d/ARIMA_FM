import pandas as pd
import joblib
import os
from src.utils import generate_features
from src.data_preprocessing import load_and_clean_data

MODEL_PATH = 'models/sarimax_model.pkl'
DATA_PATH = 'data/combinedddddd_dataset.xlsx'

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def forecast_next_steps(model, steps=1):
    df = load_and_clean_data(DATA_PATH)
    df = df.sort_values("DATE")

    if len(df) < 13:
        raise ValueError("At least 13 rows needed to generate lag features.")

    recent_df = df.tail(13)
    features_df = generate_features(recent_df)
    X_latest = features_df.drop(columns=["LOAD"]).iloc[-1]
    exog_df = pd.DataFrame([X_latest] * steps)

    forecast = model.forecast(steps=steps, exog=exog_df)
    return forecast.tolist()

def forecast_from_manual_input(model, manual_df, steps=1):
    """
    Accepts a DataFrame of 13 rows from user input to forecast future LOAD.
    """
    if manual_df.shape[0] < 13:
        raise ValueError("Manual input must have at least 13 rows.")

    features_df = generate_features(manual_df)
    X_latest = features_df.drop(columns=["LOAD"]).iloc[-1]
    exog_df = pd.DataFrame([X_latest] * steps)

    forecast = model.forecast(steps=steps, exog=exog_df)
    return forecast.tolist()


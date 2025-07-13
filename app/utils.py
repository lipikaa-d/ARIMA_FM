import os
import joblib
import pandas as pd
from src.data_preprocessing import load_and_clean_data

MODEL_PATH = 'models/sarimax_model.pkl'
DATA_PATH = 'data/combinedddddd_dataset.xlsx'


def generate_features(df):
    """
    Generate lag and rolling mean features with naming that matches training phase.
    Example: P_IN_lag1, P_IN_lag2, P_IN_roll3
    """
    df = df.copy()

    for col in ['P_IN', 'T_IN', 'P_OUT', 'T_OUT', 'LOAD']:
        for lag in range(1, 4):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df[f'{col}_roll3'] = df[col].shift(1).rolling(window=3).mean()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def forecast_next_steps(model, steps=1):
    """
    Forecast LOAD for next 'steps' using most recent data from the dataset.
    """
    df = load_and_clean_data(DATA_PATH)
    df = df.sort_values("DATE")

    if len(df) < 13:
        raise ValueError("At least 13 rows needed to generate lag features.")

    recent_df = df.tail(13)
    features_df = generate_features(recent_df)
    X_latest = features_df.drop(columns=["LOAD"]).iloc[-1]
    exog_df = pd.DataFrame([X_latest] * steps)

    # Ensure correct column order
    expected_exog = model.model.exog_names

    # Debug prints
    print("exog_df columns:", exog_df.columns.tolist())
    print("Expected columns:", expected_exog)

    # Ensure all required columns exist
    if not all(col in exog_df.columns for col in expected_exog):
        missing = [col for col in expected_exog if col not in exog_df.columns]
        raise ValueError(f"Missing required exogenous columns: {missing}")

    # Reorder exog_df to match model's expected order
    exog_df = exog_df[expected_exog]

    forecast = model.forecast(steps=steps, exog=exog_df)
    return forecast.tolist()


def forecast_from_manual_input(model, manual_df, steps=1):
    """
    Forecast LOAD from user-provided 13-row DataFrame (manual input mode).
    """
    if manual_df.shape[0] < 13:
        raise ValueError("Manual input must have at least 13 rows.")

    features_df = generate_features(manual_df)
    X_latest = features_df.drop(columns=["LOAD"]).iloc[-1]
    exog_df = pd.DataFrame([X_latest] * steps)

    # Ensure correct column order
    expected_exog = model.model.exog_names

    # Debug logs to help identify mismatch
    print("exog_df columns:", exog_df.columns.tolist())
    print("Expected columns:", expected_exog)

    # Check if all required columns are present
    if not all(col in exog_df.columns for col in expected_exog):
        missing = [col for col in expected_exog if col not in exog_df.columns]
        raise ValueError(f"Missing required exogenous columns: {missing}")

    # Reorder columns to match model
    exog_df = exog_df[expected_exog]

    forecast = model.forecast(steps=steps, exog=exog_df)
    return forecast.tolist()

def get_latest_input_data():
    df = load_and_clean_data(DATA_PATH)
    df = df.sort_values("DATE")

    if len(df) < 13:
        raise ValueError("Not enough data to extract latest input row.")

    recent_df = df.tail(13)
    features_df = generate_features(recent_df)
    latest_row = features_df.iloc[-1]

    # Extract the original values (not lagged) for display
    latest_values = {
        'P_IN': round(df.iloc[-1]['P_IN'], 2),
        'T_IN': round(df.iloc[-1]['T_IN'], 2),
        'P_OUT': round(df.iloc[-1]['P_OUT'], 2),
        'T_OUT': round(df.iloc[-1]['T_OUT'], 2),
    }

    return latest_values

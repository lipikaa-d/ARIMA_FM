# src/model_training.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
import warnings
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.data_preprocessing import load_and_clean_data

warnings.filterwarnings("ignore")

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "combinedddddd_dataset.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sarimax_model.pkl")


# === Feature Engineering: Lag & Rolling Mean ===
def add_lag_and_rolling(df, exog_cols, lag=1, window=3):
    for col in exog_cols:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_roll{window}"] = df[col].rolling(window).mean()
    return df


# === Percentage Error Metrics ===
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return round(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100, 2)

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return round(np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100, 2)


def train_and_save_model(data_path):
    print("Checking if file exists:", os.path.exists(data_path))
    print("Full path being used:", data_path)

    # Step 1: Load and clean data
    df = load_and_clean_data(data_path)
    print("Columns in DataFrame:", df.columns.tolist())

    df.set_index('DATE', inplace=True)

    # Step 2: Define base exogenous columns
    base_exog_cols = ['P_IN', 'T_IN', 'P_OUT', 'T_OUT']
    target_col = 'LOAD'

    # Step 3: Add lag and rolling mean features
    df = add_lag_and_rolling(df, base_exog_cols, lag=1, window=3)
    df.dropna(inplace=True)

    # Step 4: Select final exogenous columns
    exog_cols = []
    for col in base_exog_cols:
        exog_cols.extend([col, f"{col}_lag1", f"{col}_roll3"])

    y = df[target_col]
    exog = df[exog_cols]

    # Step 5: Chronological split
    split_index = int(len(df) * 0.9)
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    exog_train, exog_test = exog.iloc[:split_index], exog.iloc[split_index:]

    # Step 6: Train SARIMAX
    print("\nTraining SARIMAX model with order (2, 0, 1)...")
    model = SARIMAX(endog=y_train, exog=exog_train, order=(2, 0, 1))
    model_fit = model.fit(disp=False)

    # Step 7: Forecast
    y_pred = model_fit.forecast(steps=len(y_test), exog=exog_test)

    # Step 8: Evaluation
    rmse = round(sqrt(mean_squared_error(y_test, y_pred)), 4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)
    mape_val = mean_absolute_percentage_error(y_test, y_pred)
    smape_val = smape(y_test, y_pred)

    # Step 9: Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model_fit, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Step 10: Display results
    print("\nEvaluation Metrics:")
    print(f"R²    : {r2}")
    print(f"RMSE  : {rmse}")
    print(f"MAE   : {mae}")
    print(f"MAPE  : {mape_val}%")
    print(f"SMAPE : {smape_val}%")

    return model_fit, rmse, mae, mape_val, smape_val


if __name__ == '__main__':
    train_and_save_model(DATA_PATH)



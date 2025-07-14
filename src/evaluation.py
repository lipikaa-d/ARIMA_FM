import os
import joblib
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_preprocessing import load_and_clean_data
from app.utils import generate_features  # Import to create lag/rolling features

MODEL_PATH = 'models/sarimax_model.pkl'
DATA_PATH = 'data/combinedddddd_dataset.xlsx'


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return round(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100, 2)


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return round(np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100, 2)


def get_evaluation_metrics():
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Load and preprocess data
    df = load_and_clean_data(DATA_PATH)
    df = df.sort_values("DATE")  # ensure chronological
    df = generate_features(df)   # ✅ Add lag and rolling features

    if 'DATE' in df.columns:
        df = df.drop(columns=['DATE'])

    X = df.drop(columns=['LOAD'])
    y = df['LOAD']

    # Split chronologically
    split_index = int(len(df) * 0.9)
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    # Ensure column order matches training
    X_test = X_test[model.model.exog_names]

    # Forecast
    y_pred = model.forecast(steps=len(X_test), exog=X_test)

    # Evaluation metrics
    rmse_val = round(sqrt(mean_squared_error(y_test, y_pred)), 4)
    mae_val = round(mean_absolute_error(y_test, y_pred), 4)
    mape_val = mean_absolute_percentage_error(y_test, y_pred)
    smape_val = smape(y_test, y_pred)

    return {
        'rmse': rmse_val,
        'mae': mae_val,
        'mape': mape_val,
        'smape': smape_val,
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist()
    }


# CLI usage
if __name__ == '__main__':
    metrics = get_evaluation_metrics()
    print("Evaluation Metrics:")
    print(f"RMSE  : {metrics['rmse']}")
    print(f"MAE   : {metrics['mae']}")
    print(f"MAPE  : {metrics['mape']}%")
    print(f"SMAPE : {metrics['smape']}%")

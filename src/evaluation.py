import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data_preprocessing import load_and_clean_data
from statsmodels.tools.eval_measures import rmse

def load_model(model_path='model.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

def evaluate_arima_model(model, series, test_size=0.2):
    n_test = int(len(series) * test_size)
    train, test = series[:-n_test], series[-n_test:]

    predictions = model.predict(start=len(train), end=len(train) + len(test) - 1)

    # Metrics
    mse_val = mean_squared_error(test, predictions)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(test, predictions)

    return {
        'rmse': round(rmse_val, 4),
        'mae': round(mae_val, 4),
        'y_test': test.tolist(),
        'y_pred': predictions.tolist()
    }

from src.data_preprocessing import load_and_clean_data
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from math import sqrt
import os

MODEL_PATH = 'model.pkl'
DATA_PATH = 'data/combinedddddd_dataset.xlsx'

def get_evaluation_metrics():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    df = load_and_clean_data(DATA_PATH)
    df.set_index('DATE', inplace=True)

    target_col = 'LOAD'
    exog_cols = ['P_IN', 'T_IN', 'P_OUT', 'T_OUT']

    y = df[target_col]
    exog = df[exog_cols]

    split_index = int(len(df) * 0.9)
    y_test = y.iloc[split_index:]
    exog_test = exog.iloc[split_index:]

    y_pred = model.forecast(steps=len(y_test), exog=exog_test)

    rmse = round(sqrt(mean_squared_error(y_test, y_pred)), 4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    mape = round(np.mean(np.abs((y_test - y_pred) / y_test)) * 100, 2)
    smape = round(
        100 * np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred))), 2
    )

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist()
    }


# Run manually
if __name__ == '__main__':
    metrics = get_evaluation_metrics()
    print("ARIMA Model Evaluation Metrics:")
    print(f"RMSE : {metrics['rmse']}")
    print(f"MAE  : {metrics['mae']}")

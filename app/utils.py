import pandas as pd
import joblib
from src.data_preprocessing import load_and_clean_data

MODEL_PATH = 'model.pkl'
DATA_PATH = 'data/combinedddddd_dataset.xlsx'

def load_trained_model():
    return joblib.load(MODEL_PATH)

def get_latest_input_data():
    from src.data_preprocessing import load_and_clean_data
    df = load_and_clean_data("data/combinedddddd_dataset.xlsx")
    latest_row = df.iloc[-1]
    return {
        'P_IN': latest_row['P_IN'],
        'T_IN': latest_row['T_IN'],
        'P_OUT': latest_row['P_OUT'],
        'T_OUT': latest_row['T_OUT'],
        'LOAD': latest_row['LOAD']
    }


def forecast_next_steps(model, steps=1):
    from src.data_preprocessing import load_and_clean_data

    df = load_and_clean_data("data/combinedddddd_dataset.xlsx")
    df.set_index('DATE', inplace=True)

    exog_cols = ['P_IN', 'T_IN', 'P_OUT', 'T_OUT']
    latest_exog = df[exog_cols].iloc[-1:]

    repeated_exog = pd.concat([latest_exog] * steps, ignore_index=True)

    forecast = model.forecast(steps=steps, exog=repeated_exog)
    return forecast.tolist()


import pandas as pd
import numpy as np

def forecast_from_manual_input(model, manual_input, steps):
    # Extract raw input
    p_in = manual_input['P_IN']
    t_in = manual_input['T_IN']
    p_out = manual_input['P_OUT']
    t_out = manual_input['T_OUT']

    # Create DataFrame for engineered features
    exog_df = pd.DataFrame([{
        'P_IN': p_in,
        'T_IN': t_in,
        'P_OUT': p_out,
        'T_OUT': t_out,
        'P_IN_lag1': p_in,
        'T_IN_lag1': t_in,
        'P_OUT_lag1': p_out,
        'T_OUT_lag1': t_out,
        'P_IN_roll3': p_in,
        'T_IN_roll3': t_in,
        'P_OUT_roll3': p_out,
        'T_OUT_roll3': t_out
    }])

    # Forecast
    forecast = model.forecast(steps=steps, exog=pd.concat([exog_df]*steps, ignore_index=True))

    return forecast.tolist()


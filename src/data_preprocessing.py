import pandas as pd
import os

REQUIRED_COLUMNS = ['DATE', 'P_IN', 'T_IN', 'P_OUT', 'T_OUT', 'LOAD']

def load_and_clean_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read the file and explicitly load DATE column
    df = pd.read_excel(filepath)

    # Strip whitespace and standardize column names
    df.columns = [col.strip().upper() for col in df.columns]

    # Ensure DATE column is datetime
    df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')

    # Drop rows with invalid DATE
    df.dropna(subset=["DATE"], inplace=True)

    # Reset index if DATE was index
    if df.index.name == "DATE" or "DATE" not in df.columns:
        df.reset_index(inplace=True)

    # Validate required columns exist
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Sort chronologically
    df = df.sort_values("DATE").reset_index(drop=True)

    return df


# Debug mode
if __name__ == '__main__':
    df = load_and_clean_data('data/combinedddddd_dataset.xlsx')
    print(df.head())
    print(f"\nLoaded {len(df)} rows")

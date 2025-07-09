import pandas as pd
import os

def load_and_clean_data(filepath):
    import pandas as pd

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read the file and explicitly load DATE column
    df = pd.read_excel(filepath)

    # Reset index in case DATE was set as index
    df.reset_index(inplace=True)

    # Strip whitespace from column names
    df.columns = [col.strip().upper() for col in df.columns]

    # Rename to consistent names
    df.rename(columns={"DATE": "DATE"}, inplace=True)  # Ensures it's uppercase

    # Ensure DATE is datetime
    df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')

    # Drop rows where date parsing failed
    df.dropna(subset=["DATE"], inplace=True)

    return df


# Debug mode
if __name__ == '__main__':
    df = load_and_clean_data('data/combinedddddd_dataset.xlsx')
    print(df.head())
    print("\nIndex frequency:", df.index.freq)

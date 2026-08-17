import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = "data/raw/rainfall_dataset.csv"
CLEAN_DATA_PATH = "data/cleaned/rainfall_dataset_cleaned.csv"


def load_data():
    print("Loading raw data...")
    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df):
    print("Cleaning data...")

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Drop columns not useful for ML
    drop_cols = [
        "country", "location_name", "timezone",
        "last_updated", "sunrise", "sunset",
        "moonrise", "moonset"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Handle missing values
    df = df.fillna(method="ffill")

    return df


def create_target(df):
    print("Creating rainfall target column...")

    # Binary classification target
    df["rain"] = np.where(df["precip_mm"] > 0, 1, 0)

    return df


def save_clean_data(df):
    os.makedirs("data/cleaned", exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print("Cleaned data saved to:", CLEAN_DATA_PATH)


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = create_target(df)
    save_clean_data(df)

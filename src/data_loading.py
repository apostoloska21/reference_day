import pandas as pd

import config
import config_processed
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)


def load_data(file_path, dataset_name):
    df = pd.read_csv(file_path)
    # print(df.describe())
    print(f"\n{dataset_name} Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print(f"\n{dataset_name} Summary:")
    print("Shape:", df.shape)
    print("\nDescriptive Statistics:")
    print(df.describe())

    return df


def validate_time_consistency(df, dataset_name):
    time_diff = df['dtutc'].diff()
    unique_intervals = time_diff.unique()
    print(f"\n{dataset_name} Time Intervals:")
    print(unique_intervals)
    return unique_intervals


def check_datetime_column(df, dataset_name, datetime_col="dtutc"):
    if df is not None and datetime_col in df.columns:
        print(f"\n{dataset_name} DateTime Info:")
        print(df[datetime_col].dtype)
        print("\nDateTime range:")
        print(f"Start: {df[datetime_col].min()}")
        print(f"End: {df[datetime_col].max()}")
    else:
        print(f"\n{dataset_name} does not contain '{datetime_col}' column or failed to load.")


def main():
    datasets = {
        # "Elia Solar Data": config.ELIA_SOLAR_DATA_PATH,
        # "ECMWF Solar Data": config.ECMWF_SOLAR_DATA_PATH,
        # "Elia Wind Data": config.ELIA_WIND_DATA_PATH,
        # "Imbalance Price Data": config.IMBALANCE_PRICE_DATA_PATH,
        # "Meteologica Wind Data": config.METEOLOGICA_WIND_DATA_PATH,
        # "Price Data": config.PRICE_DATA_PATH,
        "Elia Solar Data": config_processed.ELIA_SOLAR_DATA_PATH,
        "ECMWF Solar Data": config_processed.ECMWF_SOLAR_DATA_PATH,
        "Elia Wind Data": config_processed.ELIA_WIND_DATA_PATH,
        "Imbalance Price Data": config_processed.IMBALANCE_PRICE_DATA_PATH,
        "Meteologica Wind Data": config_processed.METEOLOGICA_WIND_DATA_PATH,
        "Price Data": config_processed.PRICE_DATA_PATH,
    }

    dataframes = {}

    for name, path in datasets.items():
        dataframes[name] = load_data(path, name)

    for name, df in dataframes.items():
        check_datetime_column(df, name)


if __name__ == "__main__":
    main()

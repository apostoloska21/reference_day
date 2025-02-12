import pandas as pd
import logging
import config


def load_dataset(file_path, dataset_name):
    df = pd.read_csv(file_path)
    print(f"\n{dataset_name} Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print(f"\n{dataset_name} Summary:")
    print("Shape:", df.shape)
    print("\nDescription:")
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
        "meteo_da_price": config.METEO_DA_PRICE,
        "meteo_neso_solar": config.METEO_NESO_SOLAR,
        "meteo_neso_wind": config.METEO_NESO_WIND,
        "neso_demand_forecast": config.NESO_DEMAND_FORECAST,
    }

    dataframes = {}

    for name, path in datasets.items():
        dataframes[name] = load_dataset(path, name)

    for name, df in dataframes.items():
        check_datetime_column(df, name)


if __name__ == "__main__":
    main()

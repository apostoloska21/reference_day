import pandas as pd
import numpy as np
import os
import shutil


def standardize_timestamps(df):
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'index'})

    df['dtutc'] = pd.to_datetime(df['dtutc'])
    df['dtcet'] = df['dtutc'] + pd.Timedelta(hours=1)
    return df


def handle_outliers_statistical(df):
    wind_cols = ['elia_wind_da_p10', 'elia_wind_da_p90']

    for col in wind_cols:
        # IQR = Q3 - Q1 -> Q1 (25) Q2(50) Q3(75)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower_bound, upper_bound)

        print(f"\nOutlier Statistics for {col}:")
        print(f"Q1: {Q1:.2f}")
        print(f"Q3: {Q3:.2f}")
        print(f"IQR: {IQR:.2f}")
        print(f"Lower bound: {lower_bound:.2f}")
        print(f"Upper bound: {upper_bound:.2f}")

    return df


def handle_missing_values(df, dataset_name):
    print(f"\nHandling missing values for {dataset_name}")
    print("Missing values before handling:")
    print(df.isnull().sum())

    df['dtutc'] = pd.to_datetime(df['dtutc'])
    df = df.set_index('dtutc')

    if dataset_name == "Imbalance Price Data":
        if 'nrv' in df.columns:
            df['nrv'] = df['nrv'].ffill(limit=3)
            df['nrv'] = df['nrv'].bfill(limit=3)
            df['nrv'] = df['nrv'].interpolate(method='time', limit_direction='both')

    df = df.interpolate(method='time', limit_direction='both')
    df = df.reset_index()
    print("\nMissing values after handling:")
    print(df.isnull().sum())
    return df


def calculate_rolling_statistics(df, window='24H'):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    rolling_mean = df[numeric_cols].rolling(window=window).mean()
    rolling_std = df[numeric_cols].rolling(window=window).std()

    return rolling_mean, rolling_std


def align_time_granularity(df, dataset_name, freq='1h'):
    print(f"\nAligning time granularity for {dataset_name}")
    print(f"Original shape: {df.shape}")
    df['dtutc'] = pd.to_datetime(df['dtutc'])
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_resampled = df.set_index('dtutc').resample(freq)[numeric_cols].mean()
    print(f"Shape after resampling: {df_resampled.shape}")
    return df_resampled


def main():
    preprocessing_dir = '../../data/preprocessing'
    os.makedirs(preprocessing_dir, exist_ok=True)
    backup_dir = '../../data/preprocessing/backup'
    os.makedirs(backup_dir, exist_ok=True)

    source_dir = '../../data/preprocessing'
    for file in os.listdir(source_dir):
        if file.endswith('_processed.csv'):
            shutil.copy2(os.path.join(source_dir, file), backup_dir)
    for file in os.listdir(source_dir):
        if file.endswith('_processed.csv'):
            os.remove(os.path.join(source_dir, file))

    datasets = {
        "Elia Solar Data": pd.read_csv('../../data/raw/elia_solar_data.csv'),
        "ECMWF Solar Data": pd.read_csv('../../data/raw/ecmwf_solar_data.csv'),
        "Elia Wind Data": pd.read_csv('../../data/raw/Elia_wind_data.csv'),
        "Imbalance Price Data": pd.read_csv('../../data/raw/Imbalance_price_data.csv'),
        "Meteologica Wind Data": pd.read_csv('../../data/raw/Meteologica_wind_data.csv'),
        "Price Data": pd.read_csv('../../data/raw/Price_data.csv'),
    }

    processed_dfs = {}
    for name, df in datasets.items():
        print(f"\nProcessing {name}")
        df_clean = handle_missing_values(df.copy(), name)
        df_aligned = align_time_granularity(df_clean, name)
        processed_dfs[name] = df_aligned

        if name == "Elia Wind Data":
            df_clean = handle_outliers_statistical(df_clean)

        df_aligned = align_time_granularity(df_clean, name)
        processed_dfs[name] = df_aligned

    for name, df in processed_dfs.items():
        output_path = f'../../data/preprocessing/{name.lower().replace(" ", "_")}_processed.csv'
        df.to_csv(output_path)
        print(f"Saved processed file to: {output_path}")


if __name__ == "__main__":
    processed_data = main()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config
import os
import shutil


def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df['dtutc'] = pd.to_datetime(df['dtutc'])
        if not df['dtutc'].dt.tz:
            df['dtutc'] = df['dtutc'].dt.tz_localize('UTC')

        df['dtcet'] = df['dtutc'].dt.tz_convert('Europe/Paris')

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[['dtcet'] + list(numeric_cols)]

        df.set_index('dtcet', inplace=True)

        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        raise

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df_hourly = df.resample('h').mean()
        df_hourly = df_hourly.interpolate(method='polynomial', order=2)
        return df_hourly

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise


def align_datasets(datasets: dict) -> pd.DataFrame:
    try:

        start_date = pd.Timestamp('2024-06-01', tz='UTC')
        end_date = pd.Timestamp('2024-10-31', tz='UTC')

        aligned_dfs = []
        for name, df in datasets.items():

            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_df = df[mask]


            filtered_df = filtered_df.add_prefix(f'{name}_')
            aligned_dfs.append(filtered_df)

        combined_df = pd.concat(aligned_dfs, axis=1)
        return combined_df

    except Exception as e:
        print(f"Error in alignment: {str(e)}")
        raise


def save_preprocessed_data(df: pd.DataFrame, filename: str):
    try:
        processing_dir = 'data/processed'
        backup_dir = 'data/processed/backup'
        os.makedirs(processing_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        file_path = os.path.join(processing_dir, filename)
        if os.path.exists(file_path):
            shutil.copy2(file_path, os.path.join(backup_dir, f"backup_{filename}"))
        df.to_csv(file_path)
        print(f"Saved processed data to {file_path}")

    except Exception as e:
        print(f"Error saving data: {str(e)}")
        raise


def main():
    try:

        datasets = {
            "demand": load_dataset(config.NESO_DEMAND_FORECAST),
            "price": load_dataset(config.METEO_DA_PRICE),
            "wind": load_dataset(config.METEO_NESO_WIND),
            "solar": load_dataset(config.METEO_NESO_SOLAR)
        }
        for name in datasets:
            datasets[name] = preprocess_dataset(datasets[name])
        aligned_data = align_datasets(datasets)
        save_preprocessed_data(aligned_data, "processed_data.csv")
    except Exception as e:
        print(f"Error in main processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()

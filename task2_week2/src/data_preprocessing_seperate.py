import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config


def handle_missing_values(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    print(f"\nHandling missing values for {dataset_name}")

    # Copy to avoid warnings
    df = df.copy()

    # Show initial missing values
    missing_before = df.isnull().sum()
    print("\nMissing values before handling:")
    print(missing_before)

    # Handle based on dataset type
    if 'solar' in dataset_name.lower():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].ffill()
            df[col] = df[col].fillna(0)

    elif 'wind' in dataset_name.lower():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].interpolate(method='polynomial', order=2)
            df[col] = df[col].ffill().bfill()

    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].interpolate(method='polynomial', order=2)
            df[col] = df[col].ffill().bfill()

    # Show final missing values
    missing_after = df.isnull().sum()
    print("\nMissing values after handling:")
    print(missing_after)

    return df


def main():
    try:

        datasets = {
            "meteo_da_price": pd.read_csv(config.METEO_DA_PRICE),
            "meteo_neso_solar": pd.read_csv(config.METEO_NESO_SOLAR),
            "meteo_neso_wind": pd.read_csv(config.METEO_NESO_WIND),
            "neso_demand_forecast": pd.read_csv(config.NESO_DEMAND_FORECAST)
        }

        processed_datasets = {}
        for name, df in datasets.items():
            processed_df = handle_missing_values(df, name)
            processed_datasets[name] = processed_df

            output_path = f"data/preprocessed/{name}_cleaned.csv"
            processed_df.to_csv(output_path, index=False)
            print(f"\nSaved cleaned dataset to: {output_path}")

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()

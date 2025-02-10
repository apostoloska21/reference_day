import pandas as pd
import numpy as np
from pathlib import Path


def validate_data_quality(df, feature_type='solar'):
    validation_report = {}

    neg_counts = (df < 0).sum()
    validation_report['negative_values'] = neg_counts

    if feature_type == 'solar':
        unrealistic = (df > 5000).sum()  # Max solar capacity in Belgium - fix this
        validation_report['unrealistic_values'] = unrealistic

    time_gaps = df.index.to_series().diff().value_counts()
    validation_report['time_gaps'] = time_gaps

    if 'da_forecast' in df.columns and 'actual' in df.columns:
        da_actual_corr = df['da_forecast'].corr(df['actual'])
        validation_report['da_actual_correlation'] = da_actual_corr

    return validation_report


def main():
    # Path to your processed data
    data_dir = Path('../../data/preprocessing')

    # List of your processed files
    processed_files = [
        'ecmwf_solar_data_processed.csv',
        'elia_solar_data_processed.csv',
        'elia_wind_data_processed.csv',
        'imbalance_price_data_processed.csv',
        'meteologica_wind_data_processed.csv',
        'price_data_processed.csv'
    ]

    for file in processed_files:
        df = pd.read_csv(data_dir / file)
        feature_type = 'solar' if 'solar' in file else 'wind' if 'wind' in file else 'price'

        print(f"\nValidation Report for {file}:")
        validation_report = validate_data_quality(df, feature_type)
        print(validation_report)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np


def extract_time_features(df):

    df['dtutc'] = pd.to_datetime(df['dtutc'])
    df['hour'] = df['dtutc'].dt.hour
    df['day'] = df['dtutc'].dt.day
    df['day_of_week'] = df['dtutc'].dt.dayofweek
    df['month'] = df['dtutc'].dt.month
    df['is_weekend'] = df['dtutc'].dt.dayofweek.isin([5, 6]).astype(int)

    df['is_business_hour'] = df['hour'].between(9, 17).astype(int)

    df['season'] = df['month'].map({12: 1, 1: 1, 2: 1,
                                    3: 2, 4: 2, 5: 2,
                                    9: 4, 10: 4, 11: 4})

    return df


def main():
    processed_files = [
        'elia_solar_data_processed.csv',
        'ecmwf_solar_data_processed.csv',
        'elia_wind_data_processed.csv',
        'imbalance_price_data_processed.csv',
        'meteologica_wind_data_processed.csv',
        'price_data_processed.csv'
    ]

    for file in processed_files:
        df = pd.read_csv(f'../../data/preprocessing/{file}')

        df_with_features = extract_time_features(df)

        # Save enhanced dataset
        output_file = file.replace('processed', 'featured')
        df_with_features.to_csv(f'../../data/preprocessing/featured/{output_file}', index=False)
        print(f"Added features to {file} and saved in featured folder")


if __name__ == "__main__":
    main()
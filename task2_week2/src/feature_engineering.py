import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    daily_metrics = df.groupby(df.index.date).agg({
        'demand_National Demand Forecast (NDF) - GB (MW)': [
            ('mean', 'mean'),
            ('max', 'max'),
            ('min', 'min'),
            ('spread', lambda x: x.max() - x.min()),
            ('std', 'std')
        ],
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)': [
            ('mean', 'mean'),
            ('max', 'max'),
            ('min', 'min'),
            ('spread', lambda x: x.max() - x.min()),
            ('std', 'std')
        ],
        'wind_wind_fc_neso_mw': [
            ('mean', 'mean'),
            ('max', 'max'),
            ('min', 'min'),
            ('spread', lambda x: x.max() - x.min()),
            ('std', 'std')
        ],
        'solar_solar_fc_neso_mw': [
            ('mean', 'mean'),
            ('max', 'max'),
            ('min', 'min'),
            ('spread', lambda x: x.max() - x.min()),
            ('std', 'std')
        ]
    })


    daily_metrics.columns = [f"{col[0]}_{col[1]}" for col in daily_metrics.columns]

    return daily_metrics


def main():
    try:

        df = pd.read_csv('data/processed/processed_data.csv', parse_dates=['dtcet'])
        df.set_index('dtcet', inplace=True)


        daily_metrics = calculate_daily_metrics(df)

        daily_metrics.to_csv('data/processed/daily_metrics.csv')
        print("Daily metrics calculated and saved successfully")


        print("\nSample of daily metrics:")
        print(daily_metrics.head())

    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        raise


if __name__ == "__main__":
    main()

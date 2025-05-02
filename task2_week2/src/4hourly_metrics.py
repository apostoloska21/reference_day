import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_4hourly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    df['4hour_block'] = df.index.floor('4h')

    four_hourly_metrics = df.groupby('4hour_block').agg({
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
            ('std', 'std'),
            ('rolling_std_7d', lambda x: x.rolling(7 * 6).std().iloc[-1] if len(x) >= 7 * 6 else np.nan),
            ('price_range', lambda x: x.max() - x.min()),
            ('cv', lambda x: (x.std() / x.mean()) if x.mean() != 0 else 0),
        ],
        'solar_solar_fc_meteo_mw': [
            ('mean', 'mean'),
            ('max', 'max'),
            ('min', 'min'),
            ('spread', lambda x: x.max() - x.min()),
            ('std', 'std')
        ],
        'wind_wind_fc_meteo_mw': [
            ('mean', 'mean'),
            ('max', 'max'),
            ('min', 'min'),
            ('spread', lambda x: x.max() - x.min()),
            ('std', 'std')
        ]
    })

    four_hourly_metrics.columns = [f"{col[0]}_{col[1]}" for col in four_hourly_metrics.columns]

    four_hourly_metrics['price_block_volatility'] = df.groupby('4hour_block')[
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)'].std()

    four_hourly_metrics['day_of_week'] = four_hourly_metrics.index.dayofweek
    four_hourly_metrics['is_weekend'] = (four_hourly_metrics['day_of_week'] >= 5).astype(int)

    four_hourly_metrics['hour_block'] = four_hourly_metrics.index.hour // 4


    return four_hourly_metrics


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
            ('std', 'std'),
            ('rolling_std_7d', lambda x: x.rolling(7).std().iloc[-1] if len(x) >= 7 else np.nan),
            ('price_range', lambda x: x.max() - x.min()),
            ('cv', lambda x: (x.std() / x.mean()) if x.mean() != 0 else 0),
        ],
        'solar_solar_fc_meteo_mw': [
            ('mean', 'mean'),
            ('max', 'max'),
            ('min', 'min'),
            ('spread', lambda x: x.max() - x.min()),
            ('std', 'std')
        ],
        'wind_wind_fc_meteo_mw': [
            ('mean', 'mean'),
            ('max', 'max'),
            ('min', 'min'),
            ('spread', lambda x: x.max() - x.min()),
            ('std', 'std')
        ]
    })

    daily_metrics.columns = [f"{col[0]}_{col[1]}" for col in daily_metrics.columns]

    daily_metrics['price_intraday_volatility'] = df[
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)'].resample('h').std().groupby(
        df.index.date).mean()

    daily_metrics['day_of_week'] = pd.to_datetime(daily_metrics.index).dayofweek
    daily_metrics['is_weekend'] = (daily_metrics['day_of_week'] >= 5).astype(int)

    # Removed as per Abhiram's suggestion
    # daily_metrics['month'] = pd.to_datetime(daily_metrics.index).month

    return daily_metrics


def main():
    try:
        df = pd.read_csv('data/processed/processed_data.csv', parse_dates=['dtcet'])
        df.set_index('dtcet', inplace=True)

        four_hourly_metrics = calculate_4hourly_metrics(df)

        four_hourly_metrics.to_csv('data/processed/4hourly_metrics.csv')

        print("4-hourly metrics calculated and saved successfully")
        print("\nSample of 4-hourly metrics:")
        print(four_hourly_metrics.head())


    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        raise


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def find_optimal_clusters(X, min_clusters, max_clusters):
    min_clusters = max(1, min_clusters)
    max_clusters = max(min_clusters + 1, max_clusters)

    inertia = []
    cluster_range = range(min_clusters, max_clusters + 1)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=45, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        print("stored inertia for each cluster")
    if len(inertia) <= 1:
        return min_clusters
    #  rate of change in inertia
    inertia_diffs = np.diff(inertia)

    if len(inertia_diffs) <= 1:
        if inertia_diffs[0] < 0:
            return min_clusters
        return min_clusters + 1

    inertia_diffs_rate = np.diff(inertia_diffs)
    # ovde -> best cluster count where the rate of decrease slows down.
    if len(inertia_diffs_rate) > 0:
        elbow_idx = np.argmax(inertia_diffs_rate) + min_clusters + 1

        elbow_idx = min(elbow_idx, max_clusters)
        elbow_idx = max(elbow_idx, min_clusters)
    else:

        elbow_idx = min_clusters

    return elbow_idx


def find_reference_days(daily_metrics_file, delivery_dates, lookback=60, min_clusters=2, max_clusters=20):
    daily_metrics = pd.read_csv(daily_metrics_file)

    if 'Unnamed: 0' in daily_metrics.columns and 'date' not in daily_metrics.columns:
        daily_metrics.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

    daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])

    features = [
        'demand_National Demand Forecast (NDF) - GB (MW)_mean',
        'demand_National Demand Forecast (NDF) - GB (MW)_max',
        'demand_National Demand Forecast (NDF) - GB (MW)_min',
        'demand_National Demand Forecast (NDF) - GB (MW)_spread',
        'demand_National Demand Forecast (NDF) - GB (MW)_std',
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_mean',
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_max',
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_min',
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_spread',
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_std',
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_rolling_std_7d',
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_price_range',
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_cv',
        'solar_solar_fc_meteo_mw_mean',
        'solar_solar_fc_meteo_mw_max',
        'solar_solar_fc_meteo_mw_min',
        'solar_solar_fc_meteo_mw_spread',
        'solar_solar_fc_meteo_mw_std',
        'wind_wind_fc_meteo_mw_mean',
        'wind_wind_fc_meteo_mw_max',
        'wind_wind_fc_meteo_mw_min',
        'wind_wind_fc_meteo_mw_spread',
        'wind_wind_fc_meteo_mw_std',

    ]

    missing_cols = [col for col in features if col not in daily_metrics.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    reference_mapping = {}

    delivery_dates = [pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d for d in delivery_dates]

    for delivery_date in delivery_dates:
        start_date = delivery_date - pd.Timedelta(days=lookback)

        historical_data = daily_metrics[(daily_metrics['date'] >= start_date) &
                                        (daily_metrics['date'] <= delivery_date)].copy()

        historical_data[features] = historical_data[features].fillna(historical_data[features].median())

        # Add this right after filling with medians
        if historical_data[features].isnull().any().any():
            print(f"Warning: NaNs remain in historical data for {delivery_date.date()}")
            print(historical_data[features].isnull().sum())
            fallback_date = (delivery_date - timedelta(days=1)).date()
            reference_mapping[delivery_date.date()] = fallback_date
            continue

        if historical_data.empty:
            print(f"No historical data found for {delivery_date}")
            fallback_date = (delivery_date - timedelta(days=1)).date()
            reference_mapping[delivery_date.date()] = fallback_date
            continue

        X = historical_data[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_samples = len(historical_data)
        if n_samples < 3 * min_clusters:
            n_clusters = min(2, n_samples)
        else:

            actual_max_clusters = min(max_clusters, n_samples // 3)

            actual_min_clusters = max(1, min(min_clusters, actual_max_clusters))
            actual_max_clusters = max(actual_min_clusters, actual_max_clusters)

            if actual_max_clusters - actual_min_clusters < 2:
                n_clusters = actual_min_clusters
            else:
                n_clusters = find_optimal_clusters(X_scaled, actual_min_clusters, actual_max_clusters)

        print(f"For {delivery_date.date()}, using {n_clusters} clusters")

        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=45, n_init=10)
        historical_data['cluster'] = kmeans.fit_predict(X_scaled)

        target_info = historical_data[historical_data['date'] == delivery_date]

        if target_info.empty:
            print(f"Target date {delivery_date} not found in historical data")
            fallback_date = (delivery_date - timedelta(days=1)).date()
            reference_mapping[delivery_date.date()] = fallback_date
            continue

        target_cluster = target_info['cluster'].values[0]
        target_status = target_info['is_weekend'].values[0]

        past_days = historical_data[
            (historical_data['date'] < delivery_date) &
            (historical_data['is_weekend'] == target_status) &
            (historical_data['cluster'] == target_cluster)
            ]

        if past_days.empty:
            past_days = historical_data[
                (historical_data['date'] < delivery_date) &
                (historical_data['is_weekend'] == target_status)
                ]

            if past_days.empty:
                fallback_date = (delivery_date - timedelta(days=1)).date()
                reference_mapping[delivery_date.date()] = fallback_date
                print(f"Warning: No reference day found for {delivery_date.date()}. Using {fallback_date} as fallback.")
                continue

        reference_date = past_days.iloc[-1]['date'].date()
        reference_mapping[delivery_date.date()] = reference_date

    reference_df = pd.DataFrame(list(reference_mapping.items()),
                                columns=['delivery_date', 'reference_date'])

    return reference_df


def recreate_clusters_for_date(delivery_date, daily_metrics_df, lookback=60):
    if not isinstance(delivery_date, pd.Timestamp):
        delivery_date = pd.Timestamp(delivery_date)

    start_date = delivery_date - pd.Timedelta(days=lookback)
    historical_data = daily_metrics_df[(daily_metrics_df['date'] >= start_date) &
                                       (daily_metrics_df['date'] <= delivery_date)].copy()

    if historical_data.empty:
        return None, None, None, None


def assign_time_aware_clusters(daily_metrics_df, lookback=60):
    dates = daily_metrics_df['date'].sort_values().unique()
    result_df = pd.DataFrame()

    for i, date in enumerate(dates):
        if i % 10 == 0:
            print(f"Processing date {i + 1}/{len(dates)}: {date.date()}")

        if i < lookback:
            continue
        historical_data, _, kmeans, _ = recreate_clusters_for_date(
            date, daily_metrics_df, lookback=lookback, n_clusters=min(5, i // 3))

        if historical_data is not None:
            current_date_data = historical_data[historical_data['date'] == date].copy()
            if not current_date_data.empty:
                print(f"Date {date.date()}: found {len(current_date_data)} matching rows")
                result_df = pd.concat([result_df, current_date_data])

    return result_df


def main():
    daily_metrics_file = '../src/data/processed/daily_metrics.csv'

    daily_metrics = pd.read_csv(daily_metrics_file)
    if 'Unnamed: 0' in daily_metrics.columns and 'date' not in daily_metrics.columns:
        daily_metrics.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])

    da_actuals = pd.read_csv('../src/data/processed/gb_epex_da_actuals.csv', parse_dates=['dtutc'])
    da_actuals['date'] = da_actuals['dtutc'].dt.date
    delivery_dates = np.unique(da_actuals['date'])

    available_dates = daily_metrics['date'].dt.date.unique()
    valid_delivery_dates = [d for d in delivery_dates if d in available_dates]

    print(f"Total valid delivery dates: {len(valid_delivery_dates)}")

    reference_df = find_reference_days(daily_metrics_file, valid_delivery_dates)

    output_file = '../src/data/processed/time_aware_reference_mapping.csv'
    reference_df.to_csv(output_file, index=False)

    print(f"Reference day mapping saved to '{output_file}'")
    print(f"Total delivery dates mapped: {len(reference_df)}")


if __name__ == "__main__":
    main()

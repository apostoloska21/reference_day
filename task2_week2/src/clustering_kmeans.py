import os

import features
import pandas as pd
import numpy as np
import scaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer

daily_metrics_file = '../src/data/processed/daily_metrics.csv'
daily_metrics = pd.read_csv(daily_metrics_file)

if 'Unnamed: 0' in daily_metrics.columns and 'date' not in daily_metrics.columns:
    daily_metrics.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
daily_metrics.sort_values(by='date', inplace=True)
daily_metrics.reset_index(drop=True, inplace=True)

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
    #     'price_intraday_volatility',
    'day_of_week',
    'is_weekend',
    'month'
]

missing_cols = [col for col in features if col not in daily_metrics.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

X = daily_metrics[features].values
scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent', etc.
X_imputed = imputer.fit_transform(daily_metrics[features])
X_scaled = scaler.fit_transform(X_imputed)

# n_clusters = 10
num_clusters = []
cluster_range = range(2, 20)

# if len(X_scaled) < n_clusters:
#     n_clusters = len(X_scaled)


for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=45, n_init="auto")
    kmeans.fit(X_scaled)
    num_clusters.append(kmeans.inertia_)
    daily_metrics['cluster'] = kmeans.fit_predict(X_scaled)

print(daily_metrics['cluster'].value_counts())
print("Clustered Daily Metrics:")
print(daily_metrics[['date', 'cluster']].head())
print("\nCluster counts:")
print(daily_metrics['cluster'].value_counts())

output_file = '../src/data/processed/daily_metrics_with_clusters_sklearn.csv'
daily_metrics.to_csv(output_file, index=False)
print(f"\nClustered data saved to {output_file}")

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

daily_metrics_file = '../src/data/processed/daily_metrics.csv'
daily_metrics = pd.read_csv(daily_metrics_file)

if 'Unnamed: 0' in daily_metrics.columns and 'date' not in daily_metrics.columns:
    daily_metrics.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
daily_metrics.sort_values(by='date', inplace=True)
daily_metrics.reset_index(drop=True, inplace=True)

features = [
    'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_mean',
    'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_std',
    'demand_National Demand Forecast (NDF) - GB (MW)_mean',
    'solar_solar_fc_meteo_mw_mean',
    'wind_wind_fc_meteo_mw_mean'
]


missing_cols = [col for col in features if col not in daily_metrics.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")



X = daily_metrics[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_clusters = 10

if len(X_scaled) < n_clusters:
    n_clusters = len(X_scaled)

kmeans = KMeans(n_clusters=n_clusters, init="random", random_state=42, n_init="auto")
daily_metrics['cluster'] = kmeans.fit_predict(X_scaled)

print("Clustered Daily Metrics:")
print(daily_metrics[['date', 'cluster']].head())
print("\nCluster counts:")
print(daily_metrics['cluster'].value_counts())

output_file = '../src/data/processed/daily_metrics_with_clusters_sklearn.csv'
daily_metrics.to_csv(output_file, index=False)
print(f"\nClustered data saved to {output_file}")
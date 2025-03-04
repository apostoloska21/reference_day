import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr
from similarity_metrics import *

def prepare_vectors(current_day, past_day):
    numeric_cols = current_day.index[current_day.apply(np.isreal)]

    v1 = np.nan_to_num(current_day[numeric_cols].astype(float).values)
    v2 = np.nan_to_num(past_day[numeric_cols].astype(float).values)

    return v1, v2


def find_reference_days_euclidean(df, lookback=60, top_n=1):
    results = []

    for i in range(lookback, len(df)):
        current_day = df.iloc[i]
        past_days = df.iloc[i - lookback:i]

        distances = []
        for _, past_day in past_days.iterrows():
            v1, v2 = prepare_vectors(current_day.iloc[1:], past_day.iloc[1:])
            try:
                dist = euclidean(v1, v2)
                distances.append((past_day['date'], dist))
            except Exception as e:
                print(f"Error calculating euclidean distance: {e}")
                continue

        if distances:
            distances.sort(key=lambda x: x[1])
            closest_days = [d[0] for d in distances[:top_n]]
            closest_distance = distances[0][1]

            results.append({
                'date': current_day['date'],
                'closest_reference_days': closest_days,
                'distance': closest_distance
            })
    __all__ = ['find_reference_days_euclidean']
    return pd.DataFrame(results)


def find_reference_days_cosine(df, lookback=60, top_n=1):
    results = []

    for i in range(lookback, len(df)):
        current_day = df.iloc[i]
        past_days = df.iloc[i - lookback:i]

        distances = []
        for _, past_day in past_days.iterrows():
            v1, v2 = prepare_vectors(current_day.iloc[1:], past_day.iloc[1:])
            try:
                if np.all(v1 == 0) or np.all(v2 == 0):
                    dist = 1.0
                else:
                    dist = cosine(v1, v2)
                distances.append((past_day['date'], dist))
            except Exception as e:
                print(f"Error calculating cosine distance: {e}")
                continue

        if distances:
            distances.sort(key=lambda x: x[1])
            closest_days = [d[0] for d in distances[:top_n]]
            closest_distance = distances[0][1]

            results.append({
                'date': current_day['date'],
                'closest_reference_days': closest_days,
                'distance': closest_distance
            })
    __all__ = ['find_reference_days_cosine']
    return pd.DataFrame(results)


def find_reference_days_correlation(df, lookback=60, top_n=1):
    results = []

    for i in range(lookback, len(df)):
        current_day = df.iloc[i]
        past_days = df.iloc[i - lookback:i]

        distances = []
        for _, past_day in past_days.iterrows():
            v1, v2 = prepare_vectors(current_day.iloc[1:], past_day.iloc[1:])
            try:
                if np.all(v1 == v1[0]) or np.all(v2 == v2[0]):
                    dist = 1.0
                else:
                    corr, _ = pearsonr(v1, v2)
                    dist = 1 - abs(corr)
                distances.append((past_day['date'], dist))
            except Exception as e:
                print(f"Error calculating correlation distance: {e}")
                continue

        if distances:
            distances.sort(key=lambda x: x[1])
            closest_days = [d[0] for d in distances[:top_n]]
            closest_distance = distances[0][1]

            results.append({
                'date': current_day['date'],
                'closest_reference_days': closest_days,
                'distance': closest_distance
            })
    __all__ = ['find_reference_days_correlation']
    return pd.DataFrame(results)


def main():
    df = pd.read_csv('../src/data/processed/daily_metrics.csv', parse_dates=['Unnamed: 0'])
    df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    df = df.sort_values(by='date')

    print("Calculating Euclidean distances...")
    ref_euclidean = find_reference_days_euclidean(df, lookback=60, top_n=5)
    ref_euclidean.to_csv('../src/data/processed/reference_days_euclidean_table.csv', index=False)

    print("Calculating Cosine distances...")
    ref_cosine = find_reference_days_cosine(df, lookback=60, top_n=5)
    ref_cosine.to_csv('../src/data/processed/reference_days_cosine_table.csv', index=False)

    print("Calculating Correlation distances...")
    ref_correlation = find_reference_days_correlation(df, lookback=60, top_n=5)
    ref_correlation.to_csv('../src/data/processed/reference_days_correlation_table.csv', index=False)

    print("Reference days analysis completed. Results saved.")


if __name__ == "__main__":
    main()

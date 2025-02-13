import pandas as pd
from scipy.spatial.distance import euclidean

df = pd.read_csv('../src/data/processed/daily_metrics.csv', parse_dates=['Unnamed: 0'])
df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
df = df.sort_values(by='date')


def find_reference_days(df, lookback=60, top_n=1):
    results = []

    for i in range(lookback, len(df)):
        current_day = df.iloc[i]
        past_days = df.iloc[i - lookback:i]

        distances = []
        for _, past_day in past_days.iterrows():
            dist = euclidean(current_day.iloc[1:], past_day.iloc[1:])
            distances.append((past_day['date'], dist))

        distances.sort(key=lambda x: x[1])
        closest_days = [d[0] for d in distances[:top_n]]
        closest_distance = distances[0][1] if distances else None

        results.append({
            'date': current_day['date'],
            'closest_reference_days': closest_days,
            'distance': closest_distance
        })

    return pd.DataFrame(results)


reference_df = find_reference_days(df, lookback=60, top_n=5)
reference_df.to_csv('../src/data/processed/reference_days.csv', index=False)

print("Reference days analysis complete. Results saved to reference_days.csv")

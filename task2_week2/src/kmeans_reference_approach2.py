import pandas as pd
import numpy as np
from datetime import timedelta

daily_metrics = pd.read_csv('../src/data/processed/daily_metrics_with_clusters_sklearn.csv')
daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])

da_actuals = pd.read_csv('../src/data/processed/gb_epex_da_actuals.csv', parse_dates=['dtutc'])
da_actuals['date'] = da_actuals['dtutc'].dt.date
delivery_dates = np.unique(da_actuals['date'])


def find_reference_day(target_date, daily_metrics_df, lookback=60):
    if not isinstance(target_date, pd.Timestamp):
        target_date = pd.Timestamp(target_date)

    target_info = daily_metrics_df[daily_metrics_df['date'] == target_date]

    if target_info.empty:
        return None

    target_cluster = target_info['cluster'].values[0]
    target_status = target_info['is_weekend'].values[0]

    past_days = daily_metrics_df[
        # dates before target date
        (daily_metrics_df['date'] < target_date) &
        # dates within the past 60 days
        (daily_metrics_df['date'] >= target_date - pd.Timedelta(days=lookback)) &
        # with the same weekend status
        (daily_metrics_df['is_weekend'] == target_status)
        ]

    if past_days.empty:
        return None

    # filter past_days to keep only those belonging to the same cluster
    same_cluster_days = past_days[past_days['cluster'] == target_cluster]

    # if no days in same cluster, use most recent similar day
    if same_cluster_days.empty:
        return past_days.iloc[-1]['date'].date()

    # else return day from the same cluster
    return same_cluster_days.iloc[-1]['date'].date()


original_daily_metrics = daily_metrics.copy()

reference_mapping = {}

available_dates = daily_metrics['date'].dt.date.unique()
valid_delivery_dates = [d for d in delivery_dates if d in available_dates]

valid_delivery_dates = sorted(valid_delivery_dates)


print(f"Total valid delivery dates: {len(valid_delivery_dates)}")

for del_date in valid_delivery_dates:
    target_date = pd.Timestamp(del_date)

    # Filter daily_metrics up to target_date
    filtered_metrics = original_daily_metrics[original_daily_metrics['date'] <= target_date]

    # Find reference day using only data available up to target_date
    ref_day = find_reference_day(target_date, filtered_metrics, lookback=60)

    if ref_day is not None:
        reference_mapping[del_date] = ref_day
    else:
        fallback_date = (target_date - timedelta(days=1)).date()
        reference_mapping[del_date] = fallback_date
        print(f"Warning: No reference day found for {del_date}. Using {fallback_date} as fallback.")
        reference_mapping[del_date] = fallback_date
# Save the results
reference_df = pd.DataFrame(list(reference_mapping.items()), columns=['delivery_date', 'reference_date'])
reference_df.to_csv('../src/data/processed/kmeans_reference_mapping_approach2.csv', index=False)

print(f"Reference day mapping saved to '../src/data/processed/kmeans_reference_mapping_approach2.csv'")
print(f"Total delivery dates mapped: {len(reference_mapping)}")

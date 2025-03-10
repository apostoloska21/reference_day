import pandas as pd
import numpy as np
from datetime import timedelta

daily_metrics = pd.read_csv('../src/data/processed/daily_metrics_with_clusters_sklearn.csv')
daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])

da_actuals = pd.read_csv('../src/data/processed/gb_epex_da_actuals.csv', parse_dates=['dtutc'])
da_actuals['date'] = da_actuals['dtutc'].dt.date
delivery_dates = np.unique(da_actuals['date'])


# extract unique del date for red days


def find_reference_day(target_date, lookback=60):
    if not isinstance(target_date, pd.Timestamp):
        target_date = pd.Timestamp(target_date)
    # retrieve row from daily metrics where the data matches target day
    target_info = daily_metrics[daily_metrics['date'] == target_date]
    if target_info.empty:
        return None
    # this should exract cluster label
    target_cluster = target_info['cluster'].values[0]
    target_status = target_info['is_weekend'].values[0]

    past_days = daily_metrics[
        # dates befor target date
        (daily_metrics['date'] < target_date) &
        # dates withing the past 60 days
        (daily_metrics['date'] >= target_date - pd.Timedelta(days=lookback)) &
        # print with the same weekend status
        (daily_metrics['is_weekend'] == target_status)
        ]

    if past_days.empty:
        return None
    # filters past_days to keep only those belonging to the same cluster as the target date
    same_cluster_days = past_days[past_days['cluster'] == target_cluster]
    # but if there are no past dates go back to more recent similar day
    if same_cluster_days.empty:
        return past_days.iloc[-1]['date'].date()
    # else return day from the same cluster
    return same_cluster_days.iloc[-1]['date'].date()


reference_mapping = {}
# i added this filter to filter the dates only on daily metrics
available_dates = daily_metrics['date'].dt.date.unique()
valid_delivery_dates = [d for d in delivery_dates if d in available_dates]

print(f"Total valid delivery dates: {len(valid_delivery_dates)}")

for del_date in valid_delivery_dates:
    target_date = pd.Timestamp(del_date)
    ref_day = find_reference_day(target_date, lookback=60)

    if ref_day is not None:
        reference_mapping[del_date] = ref_day
    else:
        fallback_date = (target_date - timedelta(days=1)).date()
        reference_mapping[del_date] = fallback_date
        print(f"Warning: No reference day found for {del_date}. Using {fallback_date} as fallback.")
reference_df = pd.DataFrame(list(reference_mapping.items()), columns=['delivery_date', 'reference_date'])

reference_df.to_csv('../src/data/processed/kmeans_reference_mapping_approach2.csv', index=False)

print(f"Reference day mapping saved to '../src/data/processed/kmeans_reference_mapping_approach2.csv'")

print(f"Total delivery dates mapped: {len(reference_mapping)}")

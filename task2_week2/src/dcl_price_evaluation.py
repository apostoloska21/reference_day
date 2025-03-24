import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from math import sqrt

reference_mapping = pd.read_csv("../src/data/processed/kmeans_reference_mapping_approach2.csv")
reference_mapping['delivery_date'] = pd.to_datetime(reference_mapping['delivery_date'])
reference_mapping['reference_date'] = pd.to_datetime(reference_mapping['reference_date'])

dcl_prices = pd.read_csv("../src/data/processed/gb_dcl_prices_gbp_per_mw_per_hour.csv")
dcl_prices['dtutc'] = pd.to_datetime(dcl_prices['dtutc'])

dcl_prices['date'] = dcl_prices['dtutc'].dt.date


def calculate_dcl_price_rmse(reference_date, delivery_date, dcl_prices_df):
    if isinstance(reference_date, pd.Timestamp):
        reference_date = reference_date.date()
    if isinstance(delivery_date, pd.Timestamp):
        delivery_date = delivery_date.date()

    ref_date_records = dcl_prices_df[dcl_prices_df['date'] == reference_date]
    del_date_records = dcl_prices_df[dcl_prices_df['date'] == delivery_date]

    if len(ref_date_records) == 0:
        print(f"No DCL price records found for reference date {reference_date}")
    if len(del_date_records) == 0:
        print(f"No DCL price records found for delivery date {delivery_date}")

    ref_prices = dcl_prices_df[dcl_prices_df['date'] == reference_date].sort_values('dtutc')
    del_prices = dcl_prices_df[dcl_prices_df['date'] == delivery_date].sort_values('dtutc')

    if len(ref_prices) == 0 or len(del_prices) == 0:
        print(
            f"Missing prices for ref date {reference_date} ({len(ref_prices)} records) or delivery date {delivery_date} ({len(del_prices)} records)")
        return None

    if len(ref_prices) != 6 or len(del_prices) != 6:
        print(
            f"Warning: Expected 6 prices each. Got {len(ref_prices)} for reference day {reference_date} and {len(del_prices)} for delivery day {delivery_date}.")

        if len(ref_prices) == 0 or len(del_prices) == 0:
            return None

    min_length = min(len(ref_prices), len(del_prices))
    ref_price_vector = ref_prices.iloc[:min_length]['dcl_price'].values
    del_price_vector = del_prices.iloc[:min_length]['dcl_price'].values

    rmse = sqrt(mean_squared_error(del_price_vector, ref_price_vector))
    return rmse


start_date = pd.Timestamp('2024-08-01')
end_date = pd.Timestamp('2024-10-31')

eval_period_refs = reference_mapping[(reference_mapping['delivery_date'] >= start_date) &
                                     (reference_mapping['delivery_date'] <= end_date)]

date_counts = dcl_prices.groupby('date').size()
print(f"DCL price data available for {len(date_counts)} unique dates")
print(f"Date range: {dcl_prices['date'].min()} to {dcl_prices['date'].max()}")
print(f"Most common number of prices per day: {date_counts.value_counts().index[0]}")

results = []
for _, row in eval_period_refs.iterrows():
    rmse = calculate_dcl_price_rmse(row['reference_date'], row['delivery_date'], dcl_prices)

    if rmse is not None:
        results.append({
            'delivery_date': row['delivery_date'].date(),
            'reference_date': row['reference_date'].date(),
            'rmse': rmse
        })

results_df = pd.DataFrame(results)

if not results_df.empty:
    total_rmse = results_df['rmse'].sum()
    avg_rmse = results_df['rmse'].mean()

    print(f"Evaluation period: {start_date.date()} to {end_date.date()}")
    print(f"Number of days evaluated: {len(results_df)}")
    print(f"Total RMSE: {total_rmse:.2f}")
    print(f"Average RMSE: {avg_rmse:.2f}")
else:
    print("No valid results were obtained. Check the data availability.")

results_df.to_csv("../src/data/processed/reference_day_performance_dcl.csv", index=False)

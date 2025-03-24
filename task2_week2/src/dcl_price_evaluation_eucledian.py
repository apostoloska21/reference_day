import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from math import sqrt
import re

reference_mapping = pd.read_csv("../src/data/processed/reference_days_euclidean_table.csv")

print("Euclidean reference mapping columns:", reference_mapping.columns.tolist())
print("First few rows of reference mapping:")
print(reference_mapping.head(2))

reference_mapping.rename(columns={'date': 'delivery_date'}, inplace=True)

reference_mapping['delivery_date'] = pd.to_datetime(reference_mapping['delivery_date'])


def extract_reference_date(ref_str):

    if pd.isna(ref_str):
        return None

    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(ref_str))
    if match:
        try:
            return pd.to_datetime(match.group(1))
        except:
            pass
    return None


reference_mapping['reference_date'] = reference_mapping['closest_reference_days'].apply(extract_reference_date)

print("\nExtracted reference dates:")
print(reference_mapping[['delivery_date', 'reference_date']].head(3))

valid_refs = reference_mapping['reference_date'].notna().sum()
print(f"Valid reference dates: {valid_refs} out of {len(reference_mapping)}")

dcl_prices = pd.read_csv("../src/data/processed/gb_dcl_prices_gbp_per_mw_per_hour.csv")
dcl_prices['dtutc'] = pd.to_datetime(dcl_prices['dtutc'])
dcl_prices['date'] = dcl_prices['dtutc'].dt.date


def calculate_dcl_price_rmse(reference_date, delivery_date, dcl_prices_df):

    if isinstance(reference_date, pd.Timestamp):
        reference_date = reference_date.date()
    if isinstance(delivery_date, pd.Timestamp):
        delivery_date = delivery_date.date()

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
print(f"Delivery dates in evaluation period: {len(eval_period_refs)}")

date_counts = dcl_prices.groupby('date').size()
print(f"DCL price data available for {len(date_counts)} unique dates")
print(f"Date range: {dcl_prices['date'].min()} to {dcl_prices['date'].max()}")
print(f"Most common number of prices per day: {date_counts.value_counts().index[0]}")

results = []
for _, row in eval_period_refs.iterrows():
    if pd.notna(row['reference_date']):  # Only process rows with valid reference dates
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

if not results_df.empty:
    results_df.to_csv("../src/data/processed/euclidean_reference_day_performance_dcl.csv", index=False)
else:
    print("No results to save.")

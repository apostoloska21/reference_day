import pandas as pd
import numpy as np


def calculate_rmse(actuals, references):
    return np.sqrt(np.mean((np.array(actuals) - np.array(references)) ** 2))


def load_dcl_prices(file_path):
    dcl_prices = pd.read_csv(file_path, parse_dates=['dtutc'])
    dcl_prices['date'] = dcl_prices['dtutc'].dt.date

    # Expectet hours for the 6 daily prices (4-hour intervals)
    expected_hours = [3, 7, 11, 15, 19, 23]

    dcl_by_date = {}

    for date, group in dcl_prices.groupby('date'):
        if len(group) < 6:
            print(f"Warning: Only {len(group)} prices available for {date}")
            continue

        group = group.sort_values('dtutc')
        # sortiraj spored cas i dobij ceni vo ocekuvani casa
        hours_available = group['dtutc'].dt.hour.tolist()

        if all(hour in hours_available for hour in expected_hours):
            prices = []
            for hour in expected_hours:
                hour_data = group[group['dtutc'].dt.hour == hour]
                prices.append(hour_data.iloc[0]['dcl_price'])
            dcl_by_date[date] = prices
        else:
            print(f"Warning: Not all expected hours available for {date}")
            prices = group.head(6)['dcl_price'].tolist()
            if len(prices) == 6:
                dcl_by_date[date] = prices
            else:
                print(f"Warning: Could not find 6 prices for {date}")

    return dcl_by_date


def evaluate_reference_days(reference_mapping_file, dcl_prices_file, start_month=None, end_month=None):
    # Load reference day mapping
    ref_mapping = pd.read_csv(reference_mapping_file, parse_dates=['delivery_date', 'reference_date'])

    # Filter by month if specified
    if start_month is not None:
        ref_mapping = ref_mapping[ref_mapping['delivery_date'].dt.month >= start_month]
    if end_month is not None:
        ref_mapping = ref_mapping[ref_mapping['delivery_date'].dt.month <= end_month]

    # Load DCL prices
    dcl_by_date = load_dcl_prices(dcl_prices_file)

    # Calculate RMSE for each delivery-reference pair
    rmse_values = []
    for _, row in ref_mapping.iterrows():
        delivery_date = row['delivery_date'].date()
        reference_date = row['reference_date'].date()

        # Get DCL prices for both dates
        delivery_prices = dcl_by_date.get(delivery_date)
        reference_prices = dcl_by_date.get(reference_date)

        # Calculate RMSE if both prices are available
        if delivery_prices and reference_prices:
            rmse = calculate_rmse(delivery_prices, reference_prices)
            rmse_values.append({
                'delivery_date': delivery_date,
                'reference_date': reference_date,
                'rmse': rmse
            })
        else:
            if not delivery_prices:
                print(f"Warning: No DCL prices available for delivery date {delivery_date}")
            if not reference_prices:
                print(f"Warning: No DCL prices available for reference date {reference_date}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(rmse_values)

    # Calculate average RMSE
    if not results_df.empty:
        total_rmse = results_df['rmse'].mean()
        print(f"Average RMSE across all delivery dates: {total_rmse:.4f}")

        # Calculate monthly RMSE
        monthly_rmse = results_df.assign(
            month=pd.to_datetime(results_df['delivery_date']).dt.month
        ).groupby('month')['rmse'].mean()

        print("\nMonthly Average RMSE:")
        for month, rmse in monthly_rmse.items():
            month_name = pd.to_datetime(f"2024-{month:02d}-01").strftime('%B')
            print(f"{month_name}: {rmse:.4f}")
    else:
        print("No valid delivery-reference pairs found for evaluation.")

    return results_df


def main():
    reference_mapping_file = '../src/data/processed/time_aware_reference_mapping.csv'
    dcl_prices_file = '../src/data/processed/gb_dcl_prices_gbp_per_mw_per_hour.csv'

    start_month = 2
    end_month = 12

    print(f"Evaluating reference days for months: {start_month} to {end_month} (2024)")

    results_df = evaluate_reference_days(
        reference_mapping_file,
        dcl_prices_file,
        start_month,
        end_month
    )

    output_file = f'../src/data/processed/reference_day_rmse_evaluation_{start_month}_to_{end_month}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")



    kmeans_mapping_file = '../src/data/processed/kmeans_reference_mapping_approach2.csv'
    print("\nComparing with non-time-aware approach...")

    kmeans_results_df = evaluate_reference_days(
        kmeans_mapping_file,
        dcl_prices_file,
        start_month,
        end_month
    )

    # Compare average RMSE
    time_aware_rmse = results_df['rmse'].mean()
    kmeans_rmse = kmeans_results_df['rmse'].mean()

    print(f"\nComparison of approaches:")
    print(f"Time-aware approach average RMSE: {time_aware_rmse:.4f}")
    print(f"Non-time-aware approach average RMSE: {kmeans_rmse:.4f}")
    print(f"Difference: {abs(time_aware_rmse - kmeans_rmse):.4f}")
    print(f"Percentage improvement: {(1 - time_aware_rmse / kmeans_rmse) * 100:.2f}%")


if __name__ == "__main__":
    main()

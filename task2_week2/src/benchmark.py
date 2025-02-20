import pandas as pd
import logging
from optimization_logic import day_ahead
from similarity_analysis import find_reference_days
from feature_engineering import calculate_daily_metrics

logging.basicConfig(level=logging.INFO)

battery = day_ahead(
    size=10,
    power_up=10,
    power_down=10,
    timestep=1,
    charge_eff=0.9,
    discharge_eff=0.9,
    manual_soc_max=10,
    manual_soc_min=0,
    init_soc=5,
    end_soc=5,
    end_soc_limit_type="equal_to",
)


def benchmark_model(data):

    daily_pnls = {}
    grouped = data.groupby("date")

    for day, group in grouped:
        logging.info(f"Processing day: {day}")


        schedule = battery.solving(dam_actual=group["da_price"], dam_fc=group["da_price"], plot=False)
        daily_pnls[day] = battery.pnl

        logging.info(f"Benchmark revenue for {day}: {battery.pnl}")

    total_pnl = sum(daily_pnls.values())
    logging.info(f"Total Benchmark Pnl: {total_pnl}")

    return daily_pnls, total_pnl


def reference_model(data, lookback=60):

    daily_ref_pnls = {}
    grouped = data.groupby("date")

    dates = sorted(grouped.groups.keys())

    for idx, delivery_day in enumerate(dates):
        if idx < lookback:
            continue


        delivery_day_data = grouped.get_group(delivery_day)


        current_metrics = calculate_daily_metrics(delivery_day_data).iloc[0].to_dict()


        historical_metrics = {}
        for hist_day in dates[idx - lookback: idx]:
            hist_data = grouped.get_group(hist_day)
            historical_metrics[hist_day] = calculate_daily_metrics(hist_data).iloc[0].to_dict()


        combined_metrics = pd.DataFrame(
            [{"date": delivery_day, **current_metrics}] +
            [{"date": d, **metrics} for d, metrics in historical_metrics.items()]
        )

        ref_df = find_reference_days(combined_metrics, lookback=lookback, top_n=1)
        ref_day = ref_df.loc[0, "closest_reference_days"]


        ref_day_data = grouped.get_group(ref_day)

        # Schedule based on reference day's prices and settle on delivery day's actual prices
        battery.solving(dam_actual=delivery_day_data["da_price"], dam_fc=ref_day_data["da_price"], plot=False)
        daily_ref_pnls[delivery_day] = battery.pnl

        logging.info(f"Delivery Day {delivery_day}: Using Reference Day {ref_day} Pnl: {battery.pnl}")

    total_ref_pnl = sum(daily_ref_pnls.values())
    logging.info(f"Total Reference Model PnL: {total_ref_pnl}")

    return daily_ref_pnls, total_ref_pnl


def prepare_reference_day_table(data, lookback=60):

    reference_day_table = []
    grouped = data.groupby("date")


    dates = sorted(grouped.groups.keys())

    for idx, delivery_day in enumerate(dates):
        if idx < lookback:
            continue


        delivery_day_data = grouped.get_group(delivery_day)


        current_metrics = calculate_daily_metrics(delivery_day_data).iloc[0].to_dict()


        historical_metrics = {}
        for hist_day in dates[idx - lookback: idx]:
            hist_data = grouped.get_group(hist_day)
            historical_metrics[hist_day] = calculate_daily_metrics(hist_data).iloc[0].to_dict()


        combined_metrics = pd.DataFrame(
            [{"date": delivery_day, **current_metrics}] +
            [{"date": d, **metrics} for d, metrics in historical_metrics.items()]
        )


        ref_df = find_reference_days(combined_metrics, lookback=lookback, top_n=1)
        ref_day = ref_df.loc[0, "closest_reference_days"]
        distance = ref_df.loc[0, "distance"]


        reference_day_table.append({
            "delivery_date": delivery_day,
            "reference_date": ref_day,
            "euclidean_distance": distance,
        })

        logging.info(f"Delivery Day {delivery_day}: Reference Day {ref_day} (Distance: {distance})")

    return pd.DataFrame(reference_day_table)


def main():
    try:

        data = pd.read_csv("data/processed/gb_epex_da_actuals.csv", parse_dates=["dtutc", "dtcet"])

    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        return

    logging.info("***** Runing Benchmark Model *****")
    benchmark_daily, benchmark_total = benchmark_model(data)

    logging.info("\n***** Runing Reference Day Model *****")
    reference_daily, reference_total = reference_model(data, lookback=60)

    logging.info("\n***** Preparing Reference Day Table *****")

    # ( Aug/Sep/Oct)
    last_three_months = ["2024-08", "2024-09", "2024-10"]
    filtered_data = data[data["date"].dt.strftime("%Y-%m").isin(last_three_months)]

    reference_table = prepare_reference_day_table(filtered_data, lookback=60)


    reference_table.to_csv("reference_day_table.csv", index=False)
    print(reference_table.head(10))
    logging.info("\n*********** Summary ***********")
    logging.info(f"Benchmark Total Pnl: {benchmark_total}")
    logging.info(f"Reference Model Total Pnl: {reference_total}")

    return benchmark_daily, benchmark_total, reference_daily, reference_total


if __name__ == "__main__":
    main()

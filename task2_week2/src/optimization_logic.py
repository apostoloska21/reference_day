import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pulp import *

settlement_freq = {0.25: "15min", 0.5: "30min", 1: "H"}
settlement_label = {0.25: "qH", 0.5: "HH", 1: "PH"}

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------


class day_ahead:

    def __init__(
        self,
        size,
        power_up,
        power_down,
        timestep,
        charge_eff,
        discharge_eff,
        manual_soc_max,
        manual_soc_min,
        init_soc=None,
        end_soc=None,
        daily_cycle_max=2.2,
        throughput_size=None,
        solution_time_limit=30,
        integer_solution=True,
        end_soc_limit_type = None
    ):

        self.size = size
        self.power_up = power_up
        self.power_down = power_down
        if throughput_size is None:
            self.throughput_size = self.size
        else:
            self.throughput_size = throughput_size
            
        self.daily_cycle_max = daily_cycle_max

        self.manual_soc_max = manual_soc_max
        self.manual_soc_min = manual_soc_min

        self.timestep = timestep
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff

        if init_soc is None:
            self.init_soc = self.size / 2
        else:
            self.init_soc = init_soc

        self.end_soc = end_soc
        if end_soc_limit_type not in ['equal_to', 'greater_than_or_equal_to', 'less_than_or_equal_to']:
            raise ValueError(f"param 'end_soc_limit_type' = {end_soc_limit_type} must be chosen from {['equal_to', 'greater_than_or_equal_to', 'less_than_or_equal_to']}")
        else:
            self.end_soc_limit_type = end_soc_limit_type

        self.end_soc_tolerance_mwh = (0.25 / 100) * self.size

        self.solution_time_limit = solution_time_limit
        self.integer_solution = integer_solution

    def solving(
        self,
        dam_actual=None,
        dam_fc=None,
        plot=True,
        title_text1="DAM Auction",
        title_text2="",
    ):

        price = dam_fc if isinstance(dam_fc, pd.Series) else dam_actual
        start_index = price.index.min()

        # Create the problem variable to contain the problem data
        model = LpProblem("bess", LpMinimize)

        # Decision variables
        charge = LpVariable.dicts("charge", price.index, 0, self.power_down, cat="Integer" if self.integer_solution else "Continuous")
        discharge = LpVariable.dicts("discharge", price.index, -self.power_up, 0, cat="Integer" if self.integer_solution else "Continuous")
        energy = LpVariable.dicts("soc", price.index)
        c_binary = LpVariable.dicts("c01", price.index, 0, 1, cat="Integer")
        d_binary = LpVariable.dicts("d01", price.index, 0, 1, cat="Integer")

        # Objective function: Minimize the total value/Maximize the PnL
        model += lpSum((discharge[i] + charge[i]) * price[i] for i in price.index)

        # Constraint: Cycles
        model += (
            -lpSum(discharge[i] for i in price.index)
            * self.timestep
            / (self.throughput_size)
            <= self.daily_cycle_max
        )

        smallM = 0.009
        bigM = 100 * max(self.power_up, self.power_down)
        for f in price.index:

            # Constraint: SOC
            model += energy[f] == self.init_soc + lpSum(
                charge[i] * self.charge_eff * self.timestep
                + discharge[i] * self.timestep * (1 / self.discharge_eff)
                for i in range(start_index, f + 1)
            )
            model += energy[f] <= self.manual_soc_max
            model += energy[f] >= self.manual_soc_min

            model += energy[f] <= self.size
            model += energy[f] >= 0

            # adding big-M constraints linking binary and continuous:
            model += charge[f] <= c_binary[f] * bigM
            model += discharge[f] >= -d_binary[f] * bigM
            
            # additional small-M constraints, to be used in case of leaky and/or sub-optimal results
            # model += charge[f] >= c_binary[f] * smallM
            # model += discharge[f] <= -d_binary[f] * smallM
            
            # Adding constraint for simultaneous charge/discharge
            model += c_binary[f] + d_binary[f] <= 1

        # SOC in last hour:
        if self.end_soc is not None:
            if self.end_soc_limit_type == "equal_to":
                model += (
                    energy[price.index[-1]] >= self.end_soc - self.end_soc_tolerance_mwh
                )
                model += (
                    energy[price.index[-1]] <= self.end_soc + self.end_soc_tolerance_mwh
                )
            elif self.end_soc_limit_type == "greater_than_or_equal_to":
                model += energy[price.index[-1]] >= self.end_soc
            elif self.end_soc_limit_type == "less_than_or_equal_to":
                model += energy[price.index[-1]] <= self.end_soc
            
        # Solve the problem
        solver = getSolver(
            "PULP_CBC_CMD", timeLimit=self.solution_time_limit, msg=False
        )
        model.solve(solver)

        # accessing the solution
        schedule = pd.DataFrame(index=price.index)
        for i in schedule.index:
            schedule.loc[i, "charge_mw"] = round(
                (charge[i].varValue) if c_binary[i].varValue == 1 else 0, 3
            )
            schedule.loc[i, "discharge_mw"] = round(
                (discharge[i].varValue) if d_binary[i].varValue == 1 else 0, 3
            )
            schedule.loc[i, "soc_end"] = round(
                self.init_soc
                + sum(
                    schedule.loc[f, "charge_mw"] * self.timestep * self.charge_eff
                    + schedule.loc[f, "discharge_mw"]
                    * self.timestep
                    / self.discharge_eff
                    for f in range(start_index, i + 1)
                ),
                3,
            )

        schedule['soc_start'] = schedule.soc_end.shift(1)
        schedule.loc[start_index, "soc_start"] = self.init_soc
        schedule["net"] = schedule["charge_mw"] + schedule["discharge_mw"]
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if schedule["net"].isnull().any():
            raise ValueError("The 'net' column in the schedule contains NaN or None values.")
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.pnl = -np.round(sum(schedule.net * dam_actual) * self.timestep, 2)
        self.charge_mwh = np.round(sum(schedule.charge_mw) * self.timestep, 2)
        self.discharge_mwh = -np.round(sum(schedule.discharge_mw) * self.timestep, 2)
        self.cycles = np.round(
            -sum(schedule.discharge_mw) * self.timestep / (self.throughput_size), 2
        )
        #  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if dam_actual is None:
            raise ValueError("The 'dam_actual' parameter is None. Please provide valid day-ahead market prices.")
        if dam_actual.isnull().any():
            raise ValueError("The 'dam_actual' Series contains NaN or None values.")
        if schedule["charge_mw"].isnull().any():
            raise ValueError("The 'charge_mw' column contains NaN or None values.")
        if schedule["discharge_mw"].isnull().any():
            raise ValueError("The 'discharge_mw' column contains NaN or None values.")
        if dam_actual.isnull().any():
            raise ValueError("The 'dam_actual' Series contains NaN or None values.")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------- Visualize ------- #
        if plot:

            fig, ax = plt.subplots(3, 1, sharex=True)
            ax[0].plot(schedule.soc_end/self.size * 100, label="SOC", c="g", marker="o")
            ax[1].bar(schedule.index, schedule.charge_mw, label="charge")
            ax[1].bar(schedule.index, schedule.discharge_mw, label="discharge")
            ax[2].plot(dam_actual, label="DAM price", c='k')
            if isinstance(dam_fc, pd.Series):
                ax[2].plot(dam_fc, label="DAM forecast price", c='gray')

            ax[0].axhline(0, c='k', ls = ':', alpha = 0.3)
            ax[0].axhline(100, c='k', ls = ':', alpha = 0.3)

            ax[0].set(ylim=(-3, 103), ylabel="%")
            
            ax[1].set(ylabel="MW")
            ax[2].set(ylabel="EUR/MWh", xlabel=f"{settlement_label[self.timestep]}")

            ax[0].legend()
            ax[1].legend()
            ax[2].legend()

            ax[0].grid(alpha=0.5)
            ax[1].grid(alpha=0.5)
            ax[2].grid(alpha=0.5)

            plt.suptitle(
                f"{title_text1}\n PnL = {self.pnl} GBP based on {title_text2},\nCycles = {self.cycles}"
            )
            plt.tight_layout()
            plt.show()

        return schedule[["charge_mw","discharge_mw","net","soc_start","soc_end",]]


# ------------------------------------------------------------------------------------------
# ----------------------------------- EOF --------------------------------------------------
# ------------------------------------------------------------------------------------------

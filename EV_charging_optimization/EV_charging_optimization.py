"""
Author: Lukas Elmiger, luki.elmiger@hotmail.com, luelmiger@ethz.ch, lukas.elmiger@ckw.ch
Date of first contribution: 07.03.2025

This file is used to optimize EV charging concurrency - based on real EV charging time series data, stored in
.parquet file and generated in EV_charging_session_identification or downloaded elsewhere.

Lots of plotting functionality is implemented as well to see the effect of different EV charging strategies from a
DSO perspective.

Remark: Very heavy CPU load - takes more than 1 day in this form on PSL server with 22 cores

Input: File path containing all the EV charging sessions in a .parquet file
"""
import os
import pickle
import re

import numpy as np
import pandas as pd
import datetime as dt
from datetime import time, timedelta
from time import sleep
from collections import defaultdict
import random

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

df = pd.read_parquet("//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/unlabelled_smd/results/"
                     "11kw_charging_timeseries/charging_timeseries.parquet")
ALL_SLOTTED_TIMESERIES = None
import scienceplots
plt.style.use('science')


def set_plot_params(num_figures=1, base_font_size=13, text_width=6.5, wide_plot=False):
    """
    Set global plotting parameters using rcParams for LaTeX-style figures.

    Parameters:
    - num_figures: Number of figures to handle for LaTeX layout (1 or 2).
    - base_font_size: Base font size to match LaTeX document.
    - text_width: Width of the LaTeX text area in inches (default 6.5 inches).
    - wide_plot: If True, increases width for long time series plots.
    """
    plt.style.use('science')

    if num_figures == 1:
        latex_width = text_width * (1.1 if wide_plot else 0.8)  # Make wider if needed
    elif num_figures == 2:
        latex_width = text_width * 0.95 / 2
    else:
        raise ValueError("Only 1 or 2 figures are supported for LaTeX formatting.")

    latex_height = latex_width / (2.0 if wide_plot else 1.618)  # Wider plots should be shorter

    SMALL_SIZE = base_font_size * 0.8
    MEDIUM_SIZE = base_font_size
    BIGGER_SIZE = base_font_size * 1.2

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": SMALL_SIZE,
        "axes.titlesize": BIGGER_SIZE,
        "axes.labelsize": MEDIUM_SIZE,
        "xtick.labelsize": MEDIUM_SIZE,
        "ytick.labelsize": MEDIUM_SIZE,
        "legend.fontsize": MEDIUM_SIZE,
        "figure.titlesize": BIGGER_SIZE,
        "figure.figsize": (latex_width, latex_height),
        "grid.color": "gray",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
    })


class ChargingSession:
    """
    Represents an uncontrolled charging session.

    Attributes:
        is_controlled (bool): Indicates if the session is controlled.
        original_start (datetime): Original start time.
        original_end (datetime): Original end time.
        original_magnitude (float): Original charging power in kW.
        original_energy (float): Total energy delivered in kWh, computed dynamically.
    """
    def __init__(self, original_start: dt.datetime, original_end: dt.datetime, original_magnitude: float):
        self.is_controlled = False
        self.original_start = original_start
        self.original_end = original_end
        self.original_magnitude = original_magnitude

        # Compute original energy (kWh)
        duration_hours = (self.original_end - self.original_start).total_seconds() / 3600
        self.original_energy = self.original_magnitude * duration_hours

    def __repr__(self):
        return (f"ChargingSession(start={self.original_start}, end={self.original_end}, "
                f"power={self.original_magnitude} kW, original_energy={self.original_energy:.2f} kWh)")


class ControlledChargingSession(ChargingSession):
    """
    Represents a controlled charging session with scheduled start, end, power, and energy constraints.

    Attributes:
        scheduled_start (datetime): New scheduled start time.
        scheduled_end (datetime): New scheduled end time.
        scheduled_magnitude (float): Scheduled charging power in kW.
        scheduled_energy (float): Scheduled energy delivered in kWh, computed dynamically.
        energy_scheduled_ok (bool): Indicates if scheduled energy matches the original energy.
    """

    def __init__(self, original_start: dt.datetime, original_end: dt.datetime, original_magnitude: float,
                 scheduled_start: dt.datetime, scheduled_end: dt.datetime, scheduled_magnitude: float = None):
        super().__init__(original_start, original_end, original_magnitude)
        self.is_controlled = True
        self.scheduled_start = scheduled_start
        self.scheduled_end = scheduled_end
        self.scheduled_magnitude = scheduled_magnitude if scheduled_magnitude is not None else original_magnitude

        # Compute scheduled energy (kWh)
        scheduled_duration_hours = (self.scheduled_end - self.scheduled_start).total_seconds() / 3600
        self.scheduled_energy = self.scheduled_magnitude * scheduled_duration_hours

        # Check if scheduled energy matches original energy
        self.energy_scheduled_ok = abs(
            self.scheduled_energy - self.original_energy) < 1e-3  # Tolerance for float errors

    def __repr__(self):
        return (f"ControlledChargingSession(start={self.original_start}, end={self.original_end}, "
                f"power={self.original_magnitude} kW, original_energy={self.original_energy:.2f} kWh, "
                f"scheduled_start={self.scheduled_start}, scheduled_end={self.scheduled_end}, "
                f"scheduled_power={self.scheduled_magnitude} kW, scheduled_energy={self.scheduled_energy:.2f} kWh, "
                f"energy_scheduled_ok={self.energy_scheduled_ok})")


class ScenarioResults:
    """
    Contains results for a Scenario, i.e., worst case coincidence of the aggregated time series, lists with number of
    failed charging sessions, total number of charging sessions, fraction of failed charging sessions, amount of energy
    that could not be delivered until the end of the flexible time frame and departure time (from pdf), the fraction of
    undelivered energy in relation to the session energy for all sessions.
    """
    def __init__(self, wcc, num_failed, num_failed_dep, num_tot, energy_failed, energy_failed_dep, energy_fraction,
                 energy_fraction_dep, num_modified, subset_size):
        """
        Initializing the ScenarioResults instance
        :param wcc: Worst case coincidence for aggregated signals and the given scenario
        :param num_failed: int, Number of failed charging session, where at the end of the flexible time frame or
                           departure time has not been delivered
        :param num_tot: int, Total number of charging sessions present in the investigated scenario
        :param energy_failed: List(float), containing the energy in kWh that was not delivered in time
        :param energy_fraction: List(float), containing the fractions of energy, that could not be delivered in time in
                                relation to the total session energy
        :param num_modified: Number of modified charging sessions
        :param subset_size: Number of CP (charging points) of the given scenario
        The _dep variables (dep standing for departure) measure the same metrics but assume a departure time from a
        probability density function which is called for each session still going on at the end of the flexible time
        frame. This is done, because typically not all cars immediately depart at the end of the flexible time range.
        """
        # Worst case coincidence of the given combination of time series
        self.worst_case_coincidence = wcc
        # Number of unfinished sessions (charged energy less than required) within flexible time (default: 1800-0600)
        self.num_failed_sessions = num_failed
        self.num_failed_sessions_dep = num_failed_dep

        # Total number of charging sessions within the scenario
        self.num_sessions_total = num_tot

        # Missing energy
        self.missing_energy = energy_failed
        self.missing_energy_dep = energy_failed_dep

        self.missing_energy_fraction = energy_fraction
        self.missing_energy_fraction_dep = energy_fraction_dep

        self.num_modified_sessions = num_modified
        self.subset_size = subset_size


def dump_plot_data(var_data, pyfun_name, plot_name, data_direc=None, var_name="data"):
    """
    Stores raw data used to generate plots to avoid recomputation when only visual settings change.
    Also creates a metafile storing the Python function name used to generate the plot.

    :param var_data: The actual variable data to be saved (e.g. list, dict, DataFrame)
    :param pyfun_name: Name of the Python function that generated the plot (for reference)
    :param plot_name: Name of the plot (used as subfolder inside data_direc and as file prefix)
    :param data_direc: Optional base directory to store data. Defaults to a standard path.
    :param var_name: Name of the variable (used as the pickle filename inside plot_name directory)
    """
    if data_direc is None:
        data_direc = "//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/optimization/"

    assert os.path.exists(data_direc), f"Data directory {data_direc} does not exist"

    # Create subdirectory for the specific plot
    plot_dir = os.path.join(data_direc, plot_name)
    os.makedirs(plot_dir, exist_ok=True)

    # Save the variable as a pickle
    var_path = os.path.join(plot_dir, f"{var_name}.pkl")
    with open(var_path, "wb") as f:
        pickle.dump(var_data, f)

    # Save the metadata with the python function name
    meta_path = os.path.join(plot_dir, "plot_info.txt")
    with open(meta_path, "w") as f:
        f.write(f"Generated by function: {pyfun_name}\n")


def load_plot_data_from_file(file_path):
    """
    Loads stored raw data for plotting from a given .pkl file path.
    Assumes a corresponding 'plot_info.txt' is in the same directory.

    :param file_path: Full path to the .pkl file containing the plot data.
    :return: Tuple (loaded_data, metadata_string)
    """
    assert os.path.exists(file_path), f"File does not exist: {file_path}"
    assert file_path.endswith(".pkl"), "File must be a .pkl file"

    # Load variable data
    with open(file_path, "rb") as f:
        var_data = pickle.load(f)

    # Load corresponding metadata
    plot_dir = os.path.dirname(file_path)
    meta_path = os.path.join(plot_dir, "plot_info.txt")

    assert os.path.exists(meta_path), f"No metadata file found at {meta_path}"
    with open(meta_path, "r") as f:
        metadata = f.read().strip()

    return var_data, metadata


def ID_property_generator(id_list: list, num_slots: list, participation_levels: list) -> pd.DataFrame:
    """
    Generator for id properties for monte carlo simulation of different EV charging strategies
    Properties:
    - Participation: Boolean - 0 or 1, one column for each participation level called participation_p, where p
      is the probability of participation, e.g., participation_0.2
    Slot properties
    - Slot_2: 1 or 2 (Slot assignment in the case of two slots)
    - Slot_4: 1, 2, 3, 4
    ...
    - Slot_x: 1, ..., x

    :param id_list: list containing all ids, for which the properties have to be generated
    :param num_slots: list containing the number of slots
    :param participation_levels: list containing the relative participation levels
    :return: pd.Dataframe with id properties
    """
    # Make sure that participation levels are smaller equal 1
    if np.sum([participation >= 1 for participation in participation_levels]) == len(participation_levels):
        participation_levels = [participation/100 for participation in participation_levels]

    # Initialize DataFrame
    df = pd.DataFrame({'ID': id_list})
    # Generate participation columns (Boolean 0 or 1 based on probability)
    for p in participation_levels:
        df[f'participation_{p}'] = np.random.rand(len(id_list)) < p  # Assign 1 with probability `p`
    # Generate slot assignments (random integer from 1 to `num_slots`)
    for slots in num_slots:
        df[f'slot_{slots}'] = np.random.randint(1, slots + 1, size=len(id_list))
    df = df.set_index('ID')
    return df


def fix_11kW_collection(comb_11kW_collection: dict) -> dict:
    """
    Removes entries where max power exceeds 11 kW.
    This fixes corrupt or misprocessed time series entries.

    :param comb_11kW_collection: dict {id: pd.Series}
    :return: dict with faulty entries removed
    """
    faulty_keys = [key for key, value in comb_11kW_collection.items() if value.max() > 12]

    for key in faulty_keys:
        del comb_11kW_collection[key]

    faulty_keys = [key for key, value in comb_11kW_collection.items() if value.max() == 0]

    for key in faulty_keys:
        del comb_11kW_collection[key]

    return comb_11kW_collection


def simulate_worst_case(args):
    """ Runs a subset of Monte Carlo simulations for a given subset size. """
    n, all_timeseries, simulations_per_worker = args
    results = []

    for _ in range(simulations_per_worker):
        # Randomly select `n` time series
        sampled_ts = random.sample(all_timeseries, n)
        # Compute normalization factor: sum of max values of selected time series
        normalization_factor = sum(ts.max() for ts in sampled_ts)
        # Compute worst-case coincident power (sum of max values at each time step)
        worst_case_power = max(sum(ts) for ts in zip(*sampled_ts))
        # Normalize the result
        normalized_worst_case = worst_case_power / normalization_factor
        results.append(normalized_worst_case)

    return n, results  # Return results for this subset size


def plot_worst_case_coincidence_parallel(load_data_dict, result_direc, file_type, num_simulations=10000,
                                         num_workers=None, power_scale=False):
    """
    Fully parallelized Monte Carlo simulation of worst-case coincident power.
    - Utilizes all CPU cores to run simulations efficiently.
    - Divides work into balanced chunks for optimal CPU utilization.

    :param load_data_dict: Dictionary where key = id, value = Pandas time series of load data.
    :param result_direc: Directory to save the resulting boxplot.
    :param file_type: File format for saving the plot (e.g., 'png', 'pdf').
    :param num_simulations: Total number of Monte Carlo simulations per subset size.
    :param num_workers: Number of CPU cores to use (default: all available).
    :param power_scale: Power per charging point (kW), to display concurrent power instead of normalized power. If this
                        is changed from its default value 1, concurrent power is displayed, basically just multiplying
                        the values by power_scale* number of charging points n.
    :return: DataFrame of worst-case results.
    """
    # Use all available cores if num_workers is not specified
    if num_workers is None:
        num_workers = mp.cpu_count()

    os.makedirs(result_direc, exist_ok=True)
    print(f"Using {num_workers} CPU cores for parallel execution.")

    # Extract time series from dictionary
    all_timeseries = list(load_data_dict.values())
    subset_sizes = [5, 10, 20, 30, 50, 100]

    # Determine simulations per worker
    simulations_per_worker = num_simulations // num_workers

    # Prepare arguments for multiprocessing
    task_args = [(n, all_timeseries, simulations_per_worker) for n in subset_sizes for _ in range(num_workers)]

    worst_case_results = {n: [] for n in subset_sizes}

    # Parallel execution with ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(task_args), desc="Running Monte Carlo Simulations", position=0) as pbar:
            for n, results in executor.map(simulate_worst_case, task_args, chunksize=1):
                worst_case_results[n].extend(results)
                pbar.update(1)

    # Convert results to DataFrame for plotting
    df_results = pd.DataFrame(dict([(f"{n}", worst_case_results[n]) for n in subset_sizes]))

    # If concurrent power is desired, scale the values
    if power_scale:
        for n in subset_sizes:
            df_results[str(n)] = np.array(df_results[str(n)]) * n * power_scale
        ylabel = "Worst Case Concurrent\nPower (kW)"
    else:
        ylabel = "Worst Case Normalized Power"

    # Plot boxplot
    fig, ax = plt.subplots()
    df_results.boxplot(ax=ax, showfliers=True, patch_artist=True,
                       flierprops=dict(marker='.', markersize=1.2, linestyle='none', alpha=0.6))  # Small outliers

    ax.set_xlabel("Number of CP")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save & Show Plot
    fig.tight_layout()
    fig.savefig(f"{result_direc}/worst_case_coincidence_parallel.{file_type}", dpi=500)
    fig.show()

    # Convert back to write concurrency in pickle file
    if power_scale:
        for n in subset_sizes:
            df_results[str(n)] = np.array(df_results[str(n)]) / n

    return df_results


def concurrency_boxplot(df_0, df_1, result_path, power_scale=False):
    """
    Concurrency boxplot for two cases:
    :param df_0: Reference data frame containing the wcc per number of charging points from the Monte Carlo simulation
                 for the case, when charging session starting times are left unchanged
    :param df_1: Reference data frame containing the wcc per number of charging points from the Monte Carlo simulation
                 for the case, when charging session starting times are adjusted according to arrival probability
                 density function
    :param power_scale: Variable to determine whether to plot concurrency distribution (False) or concurrent
                        power distribution (scale variable, e.g., 11 for 11 kW charging points)
    :param result_path: Path, to which result is stored.

    This boxplot should highlight the difference between EV charging with a biased starting time
    (bi-level, in this case) and 'natural' / unbiased charging behavior)
    :return:
    """
    # Copy to avoid modifying originals
    df_0 = df_0.copy()
    df_1 = df_1.copy()
    subset_sizes = df_0.columns.astype(int)

    # If concurrent power is desired, scale the values
    if power_scale:
        for n in subset_sizes:
            df_0[str(n)] *= n * power_scale
            df_1[str(n)] *= n * power_scale
        ylabel = "Worst Case Concurrent\n Power (kW)"
        leg_pos = 'upper left'
    else:
        ylabel = "Worst case normalized power"
        leg_pos = 'upper right'

    # Get viridis colors using modern syntax
    viridis = plt.colormaps['viridis']
    color_0 = viridis(0.25)
    color_1 = viridis(0.75)
    color_median = viridis(0.95)

    # Setup for boxplot
    labels = df_0.columns
    n = len(labels)
    width = 0.35
    positions_0 = np.arange(n)
    positions_1 = positions_0 + width

    fig, ax = plt.subplots()

    # Plot both boxplots
    ax.boxplot(
        [df_0[col].dropna() for col in labels],
        positions=positions_0,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor=color_0),
        flierprops=dict(marker='.', markersize=1, linestyle='none', alpha=0.5),
        medianprops=dict(color=color_median, linewidth=1.2)
    )

    ax.boxplot(
        [df_1[col].dropna() for col in labels],
        positions=positions_1,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor=color_1),
        flierprops=dict(marker='.', markersize=1, linestyle='none', alpha=0.5),
        medianprops=dict(color=color_median, linewidth=1.2)
    )

    # Format axes
    mid_positions = positions_0 + width / 2
    ax.set_xticks(mid_positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of CP")
    ax.set_ylabel(ylabel)
    #ax.set_title("Comparison of Concurrency Strategies")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Legend
    handles = [
        plt.Line2D([], [], color=color_0, marker='s', linestyle='None', label='Unbiased Plug-In'),
        plt.Line2D([], [], color=color_1, marker='s', linestyle='None', label='ToU Tariff')
    ]
    ax.legend(handles=handles, loc=leg_pos)

    # Save & show
    fig.tight_layout()
    fig.savefig(result_path, dpi=500)
    plt.show()


def round_dttime_to_prev_15(dt_obj: dt.datetime) -> dt.datetime:
    """Rounds datetime object down to previous 15-min interval."""
    minutes = (dt_obj.minute // 15) * 15
    return dt_obj.replace(minute=minutes, second=0, microsecond=0)


def generate_slot_config(slot_start: dt.time | str, slot_stop: dt.time | str, num_slot_list=None) -> dict:
    """
    Generates slot configuration to define, which scenarios have to be computed - evenly distributes slots in available
    time. A slot marks the earliest possible charging start time. Rounds slot start times to next 15-min interval
    The returned values in the dict.values() are of type timedelta() with respect to slot_start - this is done, to make
    handling more simple in the simulation, when slots start after midnight, one can simply add the timedelta to
    the reference time (slot_start) to get the date without handling the day swapping logic.
    :param slot_start: Earliest possible slot time
    :param slot_stop: Latest possible slot time
    :param num_slot_list: Number of slots within slot_start and slot_stop
    :return: Dictionary with scenario_name: [timedelta for each slot with respect to start time], first timedelta = 0
    """
    if num_slot_list is None:
        num_slot_list = [2, 4, 8]

    if isinstance(slot_start, str):
        slot_start = pd.to_datetime(slot_start).time()
    if isinstance(slot_stop, str):
        slot_stop = pd.to_datetime(slot_stop).time()

    # Convert times to datetime for easier calculations
    slot_start_dt = dt.datetime.combine(dt.date.today(), slot_start)

    # Handle case where slot_stop is after midnight (i.e., slot_stop < slot_start)
    if slot_stop < slot_start:
        slot_stop_dt = dt.datetime.combine(dt.date.today() + dt.timedelta(days=1), slot_stop)
    else:
        slot_stop_dt = dt.datetime.combine(dt.date.today(), slot_stop)

    assert slot_stop_dt > slot_start_dt, "slot_stop must be later than slot_start."

    slot_dict = {}

    for num_slot in num_slot_list:
        assert num_slot > 1, f"The number of slots must be larger than 1, input: {num_slot}"

        if num_slot == 2:
            # Special case: exactly two slots
            slot_dict[f"local_slot_{num_slot}"] = [dt.timedelta(0), slot_stop_dt - slot_start_dt]
        else:
            # Compute evenly spaced slots
            step = (slot_stop_dt - slot_start_dt) / (num_slot - 1)
            slots = [slot_start_dt + step * i for i in range(num_slot)]

            # Round times to next 15-minute interval - round down at end of slot window
            rounded_slots = []
            for s in slots:
                rounded = round_dttime_to_next_15(s)
                if rounded > slot_stop_dt:
                    # If rounding pushes past slot_stop_dt, round down instead
                    rounded = round_dttime_to_prev_15(s)
                rounded_slots.append(rounded)

            # Convert to timedelta with respect to `slot_start_dt`
            slot_dict[f"local_slot_{num_slot}"] = [t - slot_start_dt for t in rounded_slots]

            # Ensure first timedelta is always `0`
            slot_dict[f"local_slot_{num_slot}"][0] = dt.timedelta(0)

    return slot_dict


def preprocess_timeseries_for_slots(sampled_ts, arrival_col):
    """
    Preprocesses time series by shifting all sessions that start between 22:00-22:30
    to a new arrival time, selected randomly from `arrival_col`.

    Works with both:
    - A list of pd.Series (each representing one time series)
    - A dictionary {id: pd.Series}

    :param sampled_ts: List or dict of `pd.Series`, each representing a power time series.
    :param arrival_col: List of possible new arrival times.

    :return: List or dict of preprocessed time series with updated session start times.
    """
    is_dict_input = isinstance(sampled_ts, dict)
    ts_items = sampled_ts.items() if is_dict_input else enumerate(sampled_ts)

    adjusted_collection = {} if is_dict_input else []

    for key, ts in ts_items:
        ts_adj = ts.copy()
        timestamps = ts.index
        session_starts = (ts.shift(1, fill_value=0) == 0) & (ts > 0)
        start_indices = np.where(session_starts)[0]

        for idx in start_indices:
            session_start_time = timestamps[idx]

            # If session starts between 22:00 and 22:30
            if dt.time(22, 0) <= session_start_time.time() <= dt.time(22, 30):
                chosen_arrival = random.choice(arrival_col)
                new_start_dt = session_start_time.replace(
                    hour=int(chosen_arrival),
                    minute=int(60 * (chosen_arrival - int(chosen_arrival))),
                    second=0,
                    microsecond=0
                )

                # Make sure new_start_dt exists in the index
                if new_start_dt not in ts.index:
                    continue  # skip if not aligned with index

                new_start_idx = ts.index.get_loc(new_start_dt)

                # Determine original session length
                session_end_idx = idx
                while session_end_idx < len(ts) - 1 and ts.iloc[session_end_idx] > 0:
                    session_end_idx += 1

                session_length = session_end_idx - idx

                # Shift session
                ts_adj[new_start_idx:new_start_idx + session_length] = ts.iloc[idx]
                ts_adj[idx:session_end_idx] = 0  # Clear original session

        if is_dict_input:
            adjusted_collection[key] = ts_adj
        else:
            adjusted_collection.append(ts_adj)

    return adjusted_collection


def simulate_slot_single_core(comb_11kW, slot_config, num_slot_list, slot_start, participation_levels,
                              subset_sizes, flexible_start_time, flexible_end_time, num_sim, dtc):
    """
    Simulates local scenarios (micro - only a subset of entire time series data, e.g., 5 CP). For the sessions of ids
    that participate, all sessions starting between flexible_start_time and flexible_end_time are allowed to start
    earliest at their slot time. If they arrive after their slot time within the flexible window, they just start
    charging upon arrival.

    Total number of iterations: num_sim * len(subset_sizes) * len(participation_levels) * len(num_slot_list)

    :param comb_11kW: dictionary containing key: id (str), value: charging time series (pd.Series), where the charging
                      session start times happening between 22:00 and 22:30 were moved before according to some arrival
                      probability density function.
    :param slot_config: contains timedeltas for each number of slots in dictionary - slot_start is the reference time
    :param num_slot_list: list of number of slots, e.g., [2, 4, 8] -> 3 scenarios
    :param slot_start: First slot start time, which serves as reference to timedeltas in slot_config
    :param participation_levels: Levels of participation, e.g.: [0.25, 0.5, 0.75, 1] -> combining with the 3 scenarios
                                 from the number of slots, this results in 12 scenarios per iteration
    :param subset_sizes: list of sizes of subsets, i.e., how many CP time series are combined, a subset size of 5 means,
                         that the EV charging data of 5 charging points is added to each other. In this context, this
                         is repeated for num_scenarios, to get an understanding of coincidence of EV charging using
                         different strategies.
    :param flexible_start_time: Start time of the flexible window, within which charge sessions strategies are applied.
                                It is assumed, that the majority of customers are fully flexible within this time range.
    :param flexible_end_time: End time of the flexible window, at which the energy should ideally have been completely
                              charged.
    :param num_sim: Number of scenarios for each configuration
    :param dtc: List of departure times according to probabilty density function from Giorgi et al. 2021
    :return:
    """
    results = {}

    if sum([participation >= 1 for participation in participation_levels]) == len(participation_levels):
        participation_levels = [p/100 for p in participation_levels]

    # Initialize result dictionary
    for num_slot in num_slot_list:
        for participation_level in participation_levels:
            results[f"local_slot_s{num_slot}_p{participation_level}"] = []

    all_timeseries = list(comb_11kW.values())
    id_list = list(comb_11kW.keys())

    # Iterate over all subset sizes, e.g. 5 CP, 10 CP, ...
    for current_subset_size in subset_sizes:
        # Repeat for num_sim times
        for _ in tqdm(range(num_sim), desc=f'Simulation slots {current_subset_size} charging points 11 kW'):
            """
            Get properties for all scenarios for all ids - has to be called for all ids before selecting a subset
            to prevent getting a skewed distribution in the subset, especially for small subsets
            """
            property_df = ID_property_generator(id_list, num_slot_list, participation_levels)

            # Randomly select `n` time series for which all scenarios are calculated
            sampled_ts = random.sample(all_timeseries, current_subset_size)
            # Get id names of the selected sample
            selected_names = [ts.name for ts in sampled_ts]
            # Step 1: Filter property_df to only include indices in names_list
            property_df_filtered = property_df.loc[property_df.index.intersection(selected_names)]
            names_by_participation = {
                level: property_df_filtered.index[property_df_filtered[f'participation_{level}'] == True].tolist()
                for level in participation_levels
            }
            # Compute normalization factor: sum of max values of the selected time series
            normalization_factor = sum([ts.max() for ts in sampled_ts])

            # Start calculation of local slot scenario
            for participation in participation_levels:
                # Get list of all participating ids, the ones that are not contained do not participate
                participation_id_list = names_by_participation[participation]

                # Saves some time for low number of CP, when it is possible, that no id participates
                if len(participation_id_list) == 0:
                    continue

                for num_slot in num_slot_list:
                    # Create dictionary with id as key and slot as value for all ids in participation_id_list
                    slot_column = f"slot_{num_slot}"
                    slot_key = f"local_slot_{num_slot}"

                    assert slot_column in property_df_filtered.columns, f"'{slot_column}' not found in DataFrame"
                    assert slot_key in slot_config, f"'{slot_key}' not found in slot_config"

                    slot_assignment = {
                        # -1 to account for fact that numbering starts with 1, instead of zero
                        # If someone sees this, feel free to fix this, don't have time now.
                        id_: slot_config[slot_key][property_df_filtered.loc[id_, slot_column]-1]
                        for id_ in participation_id_list
                        if id_ in property_df_filtered.index
                    }
                    scenario_result = scen_slot(sampled_ts, normalization_factor, flexible_start_time,
                                                flexible_end_time, dtc, slot_assignment, slot_start)
                    results[f"local_slot_s{num_slot}_p{participation}"].append(scenario_result)

    return results


def scen_slot(sampled_ts, normalization_factor, flexible_start_time, flexible_end_time, dtc, slot_assignment,
              slot_start):
    """
    Run individual slot scenarios - e.g., 4 slots, 50 % participating, 50 ids

    Calculates result metrics according to the slot configuration and participation of the timeseries data in sampled_ts
    :param sampled_ts: List(pd.Series) - list with the concerned ev charging time series
    :param normalization_factor: Sum of aggregated theoretically possible charging magnitude - is calculated out of loop
                                 for efficiency
    :param flexible_start_time: Start time of the flexible window, within which start times can be moved for
                                participating ids.
    :param flexible_end_time: End time of the flexible window
    :param dtc: List containing departure values according to departure time probability density function
    :param slot_assignment: Dictionary with timeseries id as key, timedelta as value. Only contains the keys, which
                            are participating. The timedelta is referenced to slot_start.
    :param slot_start: Start time of slot charging, aka earliest possible start time.
    :return: instance of ScenarioResults, containing all the necessary metrics
    """
    num_modified = 0
    adjusted_series = []
    num_failed, num_failed_dep, total_sessions = 0, 0, 0
    energy_failed, energy_failed_dep, energy_fraction, energy_fraction_dep = [], [], [], []

    participation_id_list = slot_assignment.keys()

    for ts in sampled_ts:
        ts_adj = ts.copy()
        timestamps = ts.index
        session_starts = (ts.shift(1, fill_value=0) == 0) & (ts > 0)
        start_indices = np.where(session_starts)[0]
        session_id = ts.name

        # Skip if id is not participating
        if session_id not in participation_id_list:
            total_sessions += len(start_indices)
            adjusted_series.append(ts_adj)
            continue

        assigned_timedelta = slot_assignment[session_id]

        for idx in start_indices:
            total_sessions += 1
            session_start_time = timestamps[idx]
            session_end_idx = idx

            # Skip sessions that don't fall in the flexible window
            if not (
                (flexible_start_time <= session_start_time.time() <= dt.time(23, 59)) or
                (dt.time(0, 0) <= session_start_time.time() < flexible_end_time)
            ):
                continue

            while session_end_idx < len(ts) - 1 and ts.iloc[session_end_idx] > 0:
                session_end_idx += 1

            # Original end time of charging session
            session_end_dt = timestamps[session_end_idx]
            original_magnitude = ts.iloc[idx]
            original_energy = (session_end_idx - idx) * original_magnitude * 0.25

            # ref date is the earliest possible date in the flexible time range
            if session_start_time.time() < flexible_end_time:
                # If session starts after midnight
                ref_date = session_start_time.date() - timedelta(days=1)
            else:
                ref_date = session_start_time.date()

            flexible_end_dt = dt.datetime.combine(ref_date + timedelta(days=1), flexible_end_time)

            slot_start_dt = dt.datetime.combine(ref_date, slot_start)
            assigned_slot_time = slot_start_dt + assigned_timedelta

            # Compute max(original_start_time, assigned_slot_time), e.g. when arrival after slot start time
            new_start_dt = max(session_start_time, assigned_slot_time)

            # Valid for sessions that started very late in the night, e.g. 05:45 - shouldn't be counted.
            if session_end_dt > flexible_end_dt and assigned_slot_time < new_start_dt:
                continue

            # Ensure new_start_dt is in the index (neglect time shift/Zeitumstellung)
            if new_start_dt not in timestamps:
                continue  # Skip if rounded slot time doesn't exist in time index

            new_start_idx = timestamps.get_loc(new_start_dt)
            original_duration_intervals = session_end_idx - idx

            # Apply session shift
            ts_adj[idx:session_end_idx] = 0
            ts_adj[new_start_idx:new_start_idx + original_duration_intervals] = original_magnitude
            num_modified += 1

            # Check for failures
            new_end_time = timestamps[min(new_start_idx + original_duration_intervals, len(ts_adj) - 1)]

            # If the session ends after the flexible window
            if new_end_time > flexible_end_dt:
                interval_length = 0.25  # 15 min
                intervals_after_flex_end = 1 + sum(
                    1 for t in timestamps[new_start_idx:new_start_idx + original_duration_intervals]
                    if t > flexible_end_dt
                )

                failed_e = round(intervals_after_flex_end * original_magnitude * interval_length, 2)
                energy_failed.append(failed_e)
                energy_fraction.append(failed_e / original_energy)
                num_failed += 1
            # No missing energy - dep variables will be 0 for this case --> jump to next session
            else:
                continue

            dep_hour, dep_minute = divmod(int(random.choice(dtc) * 60), 60)
            dep_time_obj = session_end_dt.replace(hour=dep_hour, minute=dep_minute)

            if new_end_time > dep_time_obj:
                num_failed_dep += 1
                failed_e_dep = round(
                    max(0, (new_end_time.hour + new_end_time.minute / 60) -
                        (dep_time_obj.hour + dep_time_obj.minute / 60)) * original_magnitude, 2
                )
                energy_failed_dep.append(failed_e_dep)
                energy_fraction_dep.append(failed_e_dep / original_energy)

        adjusted_series.append(ts_adj)

    # Compute WCC
    aggregate_power = pd.DataFrame(adjusted_series).sum(axis=0)
    wcc = aggregate_power.max() / normalization_factor

    return ScenarioResults(wcc, num_failed, num_failed_dep, total_sessions,
                           energy_failed, energy_failed_dep, energy_fraction, energy_fraction_dep,
                           num_modified, len(sampled_ts))


def simulate_slot_worker(task):
    (
        sampled_indices, sampled_ts_names, normalization_factor,
        participation_id_list, num_slot, slot_config, slot_start,
        flexible_start_time, flexible_end_time, dtc,
        property_df_filtered, scenario_key
    ) = task

    sampled_ts = [ALL_SLOTTED_TIMESERIES[i] for i in sampled_indices]

    slot_column = f"slot_{num_slot}"
    slot_key = f"local_slot_{num_slot}"

    slot_assignment = {
        id_: slot_config[slot_key][property_df_filtered.loc[id_, slot_column] - 1]
        for id_ in participation_id_list
        if id_ in property_df_filtered.index
    }

    result = scen_slot(sampled_ts, normalization_factor, flexible_start_time,
                       flexible_end_time, dtc, slot_assignment, slot_start)

    return scenario_key, result


def simulate_slot_parallel(comb_11kW, slot_config, num_slot_list, slot_start, participation_levels,
                           subset_sizes, flexible_start_time, flexible_end_time, num_sim, dtc):
    results = {f"local_slot_s{n}_p{p}": [] for n in num_slot_list for p in participation_levels}
    #all_timeseries = list(comb_11kW.values())
    id_list = list(comb_11kW.keys())
    tasks = []

    total_iterations = len(subset_sizes) * num_sim
    progress_bar = tqdm(total=total_iterations, desc="Initializing slot simulation tasks")

    for current_subset_size in subset_sizes:
        for _ in range(num_sim):
            property_df = ID_property_generator(id_list, num_slot_list, participation_levels)

            sampled_indices = random.sample(range(len(ALL_SLOTTED_TIMESERIES)), current_subset_size)
            sampled_ts = [ALL_SLOTTED_TIMESERIES[i] for i in sampled_indices]
            sampled_ts_names = [ts.name for ts in sampled_ts]
            normalization_factor = sum(ts.max() for ts in sampled_ts)

            selected_names = [ALL_SLOTTED_TIMESERIES[i].name for i in sampled_indices]
            property_df_filtered = property_df.loc[property_df.index.intersection(selected_names)]

            for participation in participation_levels:
                participation_key = f'participation_{participation / 100}'
                if participation_key not in property_df_filtered.columns:
                    print(participation_key)
                    continue

                participation_id_list = property_df_filtered.index[
                    property_df_filtered[participation_key] == True
                    ].tolist()

                for num_slot in num_slot_list:
                    scenario_key = f"local_slot_s{num_slot}_p{participation}"
                    task = (
                        sampled_indices,
                        sampled_ts_names,
                        normalization_factor,
                        participation_id_list,
                        num_slot,
                        slot_config,
                        slot_start,
                        flexible_start_time,
                        flexible_end_time,
                        dtc,
                        property_df_filtered,
                        scenario_key
                    )
                    tasks.append(task)

            progress_bar.update(1)

    progress_bar.close()

    num_cores = max(1, mp.cpu_count())
    print(f"Initialization complete - Start Simulation using {num_cores}/{mp.cpu_count()} Cores")

    with ProcessPoolExecutor(initializer=initialize_slot_worker, initargs=(ALL_SLOTTED_TIMESERIES,)) as executor:
        futures = [executor.submit(simulate_slot_worker, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulating Slot Scenarios"):
            scenario_key, scenario_result = future.result()
            results[scenario_key].append(scenario_result)

    return results


def scenario37(sampled_ts, new_magnitude, normalization_factor, start_time, end_time, arr_time, dep_time,
               version_b=False, version_c=False, version_d=False, fraction_d_list=None):
    """
    Adjusts charging session power levels and evaluates their impact on worst-case coincidence (WCC),
    session failures, and energy delivery.

    This function modifies power time series by adjusting the power magnitude of sessions that start
    between `start_time` and `end_time`. The duration of each modified session is recalculated to ensure
    the same total energy is delivered. The function then analyzes the impact of this modification by:
    - Computing the Worst-Case Coincidence (WCC) for aggregated charging power.
    - Identifying failed charging sessions (sessions that can no longer finish on time).
    - Computing missing energy for failed sessions.
    - Determining the fraction of undelivered energy relative to the total session energy.

    If all version_ variables are False, the following is done: For all charging activity that starts in the flexible
    time range (start_time - end_time), the magnitude is reduced to new_magnitude and the session is prolonged to
    keep the energy of the original session delivered. The other versions are described in the argument description:

    :param sampled_ts: List of `pd.Series`, each representing a power time series in 15-minute resolution.
    :param new_magnitude: The new power value (kW) assigned to affected charging sessions.
    :param normalization_factor: Value used to normalize WCC.
    :param start_time: `datetime.time` object indicating the start time of the flexible charging window
                        (default = 18:00). - is fixed for all charging sessions
    :param end_time: `datetime.time` object indicating the end time of the flexible charging window (default = 06:00
                      next day) - is fixed for all charging sessions
    :param arr_time: Collection of arrival times in a list (e.g. 15.5 = 15:30), earliest possible time to start charging
    :param dep_time: Latest possible departure time for vehicles, is a large list of integers (e.g. 6.25 = 06:15), for
                     all sessions that start their charging between start_time and end_time (aka the flexible time
                     range)
    :param version_b: If True, only adjust sessions that are still able to finish with the new magnitude and therefore
                      longer session duration before the end of the flexible time frame, i.e., end_time.
    :param version_c: If True: only adjust magnitude for the sessions that start between 22:00-22:30, also move their
                      starting timestamp according to random timestamp in arr_time. Move end timestamp, such that energy
                      remains identical
    :param version_d: If True, runs scenario, in which each id gets participation boolean from the fraction_d df. If
                      the id participates, the charging sessions which start in the flexible time range have their
                      magnitudes and start/end times changed.  - first all charging activity starting between
                      22:00-22:30 have their start time adjusted according to arrival time pdf, regardless of
                      participation.
    :param fraction_d_list: list of ids, that are participating (correspond to ids in sampled_ts, sampled_ts[x].name)

    :return: `ScenarioResults` object containing:
             - Worst-case coincidence (`wcc`)
             - Number of failed sessions (`num_failed`, `num_failed_dep`)
             - Total number of charging sessions (`num_tot`)
             - Undelivered energy amounts (`energy_failed`, `energy_failed_dep`)
             - Fractions of undelivered energy (`energy_fraction`, `energy_fraction_dep`)
    """
    assert sum([version_b, version_c, version_d]) <= 1, "Only one version flag can be True at a time."

    # insert variable to count number of modified sessions:
    num_modified = 0

    adjusted_series = []
    num_failed, num_failed_dep, total_sessions = 0, 0, 0
    energy_failed, energy_failed_dep, energy_fraction, energy_fraction_dep = [], [], [], []

    for ts in sampled_ts:
        ts_adj = ts.copy()
        timestamps = ts.index
        session_starts = (ts.shift(1, fill_value=0) == 0) & (ts > 0)
        start_indices = np.where(session_starts)[0]
        session_id = ts.name  # Get session ID from Series name

        for idx in start_indices:
            total_sessions += 1
            session_start_time = timestamps[idx]
            session_end_idx = idx

            while session_end_idx < len(ts) - 1 and ts.iloc[session_end_idx] > 0:
                session_end_idx += 1

            # End of the charging session according to the timeseries data
            session_end_dt = timestamps[session_end_idx]
            session_duration = session_end_idx - idx
            original_magnitude = ts.iloc[idx]
            original_energy = session_duration * original_magnitude * 0.25

            # Max. energy in flexible window with new magnitude - skip these sessions
            if original_energy > abs(end_time.hour - start_time.hour)*new_magnitude:
                continue

            modify_session = False

            # Set end time as dt object - set to the next day, if over midnight charging
            if (session_end_dt.day != session_start_time.day or
                    dt.time(0, 0) <= session_start_time.time() < end_time):
                end_time_dt = session_end_dt.replace(hour=end_time.hour, minute=end_time.minute)
            # Session start before midnight, should also end before midnight, otherwise caught in if-cond just above
            elif start_time <= session_start_time.time() <= dt.time(23, 45):
                end_time_dt = session_start_time + timedelta(days=1)
                end_time_dt = end_time_dt.replace(hour=end_time.hour, minute=end_time.minute)
            # Go to next sessions, if they do not start between start_time and end_time
            else:
                continue

            # Also attach the date to the time of the flexible range for the given session:
            start_time_dt = session_start_time.replace(hour=start_time.hour, minute=start_time.minute)
            # if session starts in second half of the night
            if dt.time(0, 0) <= session_start_time.time() < end_time_dt.time():
                start_time_dt = start_time_dt - timedelta(days=1)

            # if previous end time was way later than end_time, continue with next session
            if session_end_dt - end_time_dt > timedelta(hours=2):
                continue

            if version_d or version_c:
                if time(22, 0) <= session_start_time.time() <= time(22, 30):
                    chosen_arrival = random.choice(arr_time)

                    new_start_dt = session_start_time.replace(hour=int(chosen_arrival),
                                                              minute=int(60*(chosen_arrival-int(chosen_arrival))))
                    new_start_idx = timestamps.get_loc(new_start_dt)

                    modify_session = True
                    # Participation is assigned out of loop
                    if version_d and session_id not in fraction_d_list:
                        modify_session = False
                else:
                    new_start_idx = idx
            else:
                new_start_idx = idx

            # version b: Reduce to new_magnitude, if able to finish with new magnitude
            if version_b and not modify_session:
                new_duration_intervals = int(np.ceil(original_energy / (new_magnitude * 0.25)))
                # Calculate new end-time with new magnitude
                new_end_time = timestamps[min(new_start_idx + new_duration_intervals, len(ts_adj) - 1)]

                if new_end_time <= end_time_dt:
                    modify_session = True

            # Second or condition is useless - can be tested later on
            if modify_session or (((start_time_dt <= session_start_time) or
                                   (session_start_time < end_time_dt < start_time_dt)) and
                                  not version_d and not version_c and not version_b):
                new_duration_intervals = int(np.ceil(original_energy / (new_magnitude * 0.25)))
                ts_adj[new_start_idx:new_start_idx + new_duration_intervals] = new_magnitude
                ts_adj[new_start_idx + new_duration_intervals:session_end_idx] = 0

                new_end_time = timestamps[min(new_start_idx + new_duration_intervals, len(ts_adj) - 1)]
                num_modified += 1

                if new_end_time > end_time_dt:
                    num_failed += 1
                    failed_e = round(max(0, (new_end_time.hour + new_end_time.minute / 60) -
                                         (end_time_dt.hour + end_time_dt.minute / 60)) * new_magnitude, 2)
                    energy_failed.append(failed_e)
                    energy_fraction.append(failed_e / original_energy)
                else:
                    # If this time is sufficient, later departure time does not need to be checked
                    continue

                dep_hour, dep_minute = divmod(int(random.choice(dep_time) * 60), 60)
                dep_time_obj = end_time_dt.replace(hour=dep_hour, minute=dep_minute)

                if new_end_time > dep_time_obj:
                    num_failed_dep += 1
                    failed_e_dep = round(max(0, (new_end_time.hour + new_end_time.minute / 60) -
                                             (dep_time_obj.hour + dep_time_obj.minute / 60)) * new_magnitude, 2)
                    energy_failed_dep.append(failed_e_dep)
                    energy_fraction_dep.append(failed_e_dep / original_energy)

        adjusted_series.append(ts_adj)

    aggregate_power = pd.DataFrame(adjusted_series).sum(axis=0)
    wcc = aggregate_power.max() / normalization_factor

    return ScenarioResults(wcc, num_failed, num_failed_dep, total_sessions,
                           energy_failed, energy_failed_dep, energy_fraction, energy_fraction_dep, num_modified,
                           len(sampled_ts))


def plot_37_results(result_dict, participation_levels, subset_sizes, res_direc, metric='worst_case_coincidence',
                    normalize_by=None, plot_37abcd25=False, presentation_mode=False):
    """
    Plots results from 37d simulation OR compares multiple 37x scenarios (37a, 37b, 37c, 37d_100).

    :param result_dict: Dictionary containing the list of ScenarioResults instances for each scenario.
    :param participation_levels: List with all the participation levels from the simulation, e.g. [25,50,75,100].
    :param subset_sizes: List with the subset sizes from the simulation, e.g. [5,10,20,30,50,100].
    :param res_direc: Directory where results should be saved.
    :param metric: Select metric from ScenarioResults to be plotted.
    :param normalize_by: If given, normalizes metric values by this attribute of ScenarioResults.
    :param plot_37abcd25: If True, compares 37a, 37b, 37c, and 37d_25 instead of different participation levels.
    :param presentation_mode: For pptx --> sets fig-size.
    """
    # Ensure the result directory exists
    if res_direc:
        os.makedirs(res_direc, exist_ok=True)

    # Select appropriate labels (either participation levels or 37x scenarios)
    if plot_37abcd25:
        labels = ["37a", "37b", "37c", "37d_25"]
        legend_labels = labels  # Legend shows scenario names
    else:
        labels = [f"local_37d_{level}" for level in participation_levels]
        legend_labels = [str(level) for level in participation_levels]  # Legend shows just participation levels

    # Assign colors based on a fixed order, ensuring scenarios remain visually consistent
    color_map = {label: color for label, color in
                 zip(labels, plt.cm.viridis(np.linspace(0, 1, len(labels) + 2)))}
    median_color = plt.cm.viridis(0.95)
    #plt.cm.viridis((len(labels) + 1) / (len(labels) + 2))  # Select next color in the viridis colormap

    # Collect data for plotting
    boxplot_data = {size: {label: [] for label in labels} for size in subset_sizes}

    # Extract metric data while handling lists correctly
    for size in subset_sizes:
        for label in labels:
            key = f"local_{label}" if plot_37abcd25 else label  # Ensure correct key for 37d scenarios
            if key in result_dict.keys():
                values = []
                for element in result_dict[key]:
                    if hasattr(element, metric) and element.subset_size == size:
                        metric_value = getattr(element, metric, None)
                        norm_value = getattr(element, normalize_by, None) if normalize_by else None

                        # Normalize if applicable and avoid division by zero
                        if normalize_by and norm_value and norm_value != 0:
                            if isinstance(metric_value, list):
                                values.extend([v / norm_value for v in metric_value])
                            elif metric_value is not None:
                                values.append(metric_value / norm_value)
                        else:
                            if isinstance(metric_value, list):
                                values.extend(metric_value)
                            elif metric_value is not None:
                                values.append(metric_value)

                # Store only non-empty lists
                boxplot_data[size][label] = values if values else []

    if presentation_mode:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig, ax = plt.subplots()

    box_data = []
    valid_labels = []  # Store labels that have valid data
    valid_positions = []  # Store positions for valid (non-empty) data
    width = 0.17  # Width of each box

    # Generate positions and filter out empty lists
    for i, size in enumerate(subset_sizes):
        for j, label in enumerate(labels):
            if len(boxplot_data[size][label]) > 0:  # Ensure non-empty lists
                box_data.append(boxplot_data[size][label])
                valid_positions.append(i + j * width - (1.5 * width))  # Offset within each group
                if label not in valid_labels:
                    valid_labels.append(label)  # Track only those scenarios that have data

    # Create boxplot with colors and smaller outlier markers
    if len(box_data) > 0:  # Only plot if there is data available
        bp = ax.boxplot(box_data, positions=valid_positions, widths=width, patch_artist=True,
                        flierprops=dict(marker='.', markersize=1.4, linestyle='none', alpha=0.6))  # Smaller outliers

        # Apply colors from the fixed color_map, ensuring consistency
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(color_map[valid_labels[i % len(valid_labels)]])  # Keep colors aligned with fixed order

        # Change median line color to the preselected distinct color
        for median in bp['medians']:
            median.set_color(median_color)
            median.set_linewidth(1.2)

    # Set x-ticks and labels
    xticks = np.arange(len(subset_sizes))
    ax.set_xticks(xticks)
    ax.set_xticklabels(subset_sizes)

    # Labels and title
    y_label_dict = {
        'worst_case_coincidence': 'Worst Case Normalized Power',
        'num_failed_sessions': 'Number of Failed Sessions',
        'num_failed_sessions_dep': 'Number of Failed Sessions',
        'missing_energy': 'Missing Energy per Session (kWh)',
        'missing_energy_dep': 'Missing Energy per Session (kWh)',
        'missing_energy_fraction': 'Missing/Original Energy per Session',
        'missing_energy_frac_dep': 'Missing/Original Energy per Session',
        'num_modified_sessions': 'Number of Modified Sessions'
    }

    y_label = y_label_dict.get(metric, metric.replace("_", " ").title())
    if normalize_by:
        y_label += f" (Normalized by {normalize_by.replace('_', ' ').title()})"
    y_label = 'Worst Case Normalized Power'
    ax.set_xlabel("Number of CP")
    ax.set_ylabel(y_label)

    # Grid
    ax.grid(True, linestyle="--", alpha=0.6)

    # Add legend (only showing valid labels that have data)
    handles = [plt.Line2D([0], [0], color=color_map[label], lw=4, label=legend_labels[labels.index(label)])
               for label in valid_labels]
    ax.legend(handles, legend_labels, loc="upper right", fancybox=True, borderaxespad=0.2)

    # Adjust layout and save
    plt.tight_layout()
    if res_direc:
        filename_suffix = "scenario_comparison" if plot_37abcd25 else "37d"
        fig.savefig(os.path.join(res_direc, f'{filename_suffix}_{metric}_normalized_by_{normalize_by}.png'), dpi=500)
        fig.savefig(os.path.join(res_direc, f'{filename_suffix}_{metric}_normalized_by_{normalize_by}.eps'),
                    format='eps')

    # Show plot
    plt.show()


def plot_slot_results_by_slot_and_participation(result_dict, participation_levels, subset_sizes, num_slot_list,
                                                res_direc, metric='worst_case_coincidence', normalize_by=None,
                                                presentation_mode=False):
    """
    Plots results of slot scenarios in boxplots grouped by slot, participation and number of charging points.
    :param result_dict: Contains ScenarioResult instances
    :param participation_levels:
    :param subset_sizes:
    :param num_slot_list: list with the number of slots
    :param res_direc: result directory where the plots + data for the plots is stored
    :param metric: select metric to plot
    :param normalize_by: select metric, by which the plot data is normalized (e.g. total number of sessions)
    :param presentation_mode: For pptx set to true
    :return:
    """
    if res_direc:
        os.makedirs(res_direc, exist_ok=True)

    # SLOT-BASED GROUPING (group by number of slots, multiple participations)
    for slot_count in num_slot_list:
        # Otherwise HTTP error
        sleep(2)
        if presentation_mode:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig, ax = plt.subplots()

        labels = [f"local_slot_s{slot_count}_p{p}" for p in participation_levels]
        legend_labels = [f"{p}%" for p in participation_levels]
        color_map = {label: color for label, color in zip(labels, plt.cm.viridis(np.linspace(0, 1,
                                                                                             len(labels) + 2)))}
        median_color = plt.cm.viridis(0.95)#plt.cm.viridis((len(labels) + 1) / (len(labels) + 2))

        box_data, valid_positions, valid_labels = [], [], []
        width = 0.22

        for i, size in enumerate(subset_sizes):
            for j, label in enumerate(labels):
                if label not in result_dict:
                    continue
                values = []
                for el in result_dict[label]:
                    if el.subset_size != size:
                        continue
                    val = getattr(el, metric, None)
                    norm = getattr(el, normalize_by, None) if normalize_by else None
                    if isinstance(val, list):
                        vals = [(v / norm if norm and norm != 0 else v) for v in val]
                        values.extend(vals)
                    elif val is not None:
                        values.append(val / norm if norm and norm != 0 else val)
                if values:
                    box_data.append(values)
                    valid_positions.append(i + j * width - (1.5 * width))
                    valid_labels.append(label)

        if box_data:
            bp = ax.boxplot(box_data, positions=valid_positions, widths=width, patch_artist=True,
                            flierprops=dict(marker='.', markersize=1.2, linestyle='none', alpha=0.6))
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(color_map[valid_labels[i]])
            for median in bp['medians']:
                median.set_color(median_color)
                median.set_linewidth(1.2)

        ax.set_xticks(np.arange(len(subset_sizes)))
        ax.set_xticklabels(subset_sizes)
        ax.set_xlabel("Number of CP (Charging Points)")

        y_label_dict = {
            'worst_case_coincidence': 'Worst Case Normalized Power',
            'num_failed_sessions': 'Number of Failed Sessions',
            'num_failed_sessions_dep': 'Number of Failed Sessions',
            'missing_energy': 'Missing Energy per Session (kWh)',
            'missing_energy_dep': 'Missing Energy per Session (kWh)',
            'missing_energy_fraction': 'Missing/Original Energy per Session',
            'missing_energy_frac_dep': 'Missing/Original Energy per Session',
            'num_modified_sessions': 'Number of Modified Sessions'
        }
        y_label = y_label_dict.get(metric, metric.replace("_", " ").title())
        if normalize_by:
            y_label += f" (/\n{normalize_by.replace('_', ' ').title()})"
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.6)

        handles = [Line2D([0], [0], color=color_map[label], lw=4, label=legend_labels[i])
                   for i, label in enumerate(labels) if label in valid_labels]
        ax.legend(handles=handles, loc="upper right", title='Participation\n in percent')
        fig.tight_layout()
        fig.savefig(os.path.join(res_direc, f"slot{slot_count}_{metric}_by_participation.png"), dpi=500)
        fig.savefig(os.path.join(res_direc, f"slot{slot_count}_{metric}_by_participation.eps"), format='eps')
        fig.show()
        plt.close(fig)

    # PARTICIPATION-BASED GROUPING (group by participation, multiple slot counts)
    for participation in participation_levels:
        # Otherwise HTTP error
        sleep(2)

        if presentation_mode:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig, ax = plt.subplots()

        labels = [f"local_slot_s{s}_p{participation}" for s in num_slot_list]
        legend_labels = [f"{s} Slots" for s in num_slot_list]
        color_map = {label: color for label, color in zip(labels, plt.cm.viridis(np.linspace(0, 1,
                                                                                             len(labels) + 2)))}
        median_color = plt.cm.viridis(0.95) #plt.cm.viridis((len(labels) + 1) / (len(labels) + 2))

        box_data, valid_positions, valid_labels = [], [], []
        width = 0.25

        for i, size in enumerate(subset_sizes):
            for j, label in enumerate(labels):
                if label not in result_dict:
                    continue
                values = []
                for el in result_dict[label]:
                    if el.subset_size != size:
                        continue
                    val = getattr(el, metric, None)
                    norm = getattr(el, normalize_by, None) if normalize_by else None
                    if isinstance(val, list):
                        vals = [(v / norm if norm and norm != 0 else v) for v in val]
                        values.extend(vals)
                    elif val is not None:
                        values.append(val / norm if norm and norm != 0 else val)
                if values:
                    box_data.append(values)
                    valid_positions.append(i + j * width - (1.5 * width))
                    valid_labels.append(label)

        if box_data:
            bp = ax.boxplot(box_data, positions=valid_positions, widths=width, patch_artist=True,
                            flierprops=dict(marker='.', markersize=1.2, linestyle='none', alpha=0.6))
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(color_map[valid_labels[i]])
            for median in bp['medians']:
                median.set_color(median_color)
                median.set_linewidth(1.2)

        ax.set_xticks(np.arange(len(subset_sizes)))
        ax.set_xticklabels(subset_sizes)
        ax.set_xlabel("Number of CP (Charging Points)")

        y_label = y_label_dict.get(metric, metric.replace("_", " ").title())
        if normalize_by:
            y_label += f" (/\n{normalize_by.replace('_', ' ').title()})"
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.6)

        handles = [Line2D([0], [0], color=color_map[label], lw=4, label=legend_labels[i])
                   for i, label in enumerate(labels) if label in valid_labels]
        ax.legend(handles=handles, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(res_direc, f"participation{participation}_{metric}_by_slots.png"), dpi=500)
        fig.savefig(os.path.join(res_direc, f"participation{participation}_{metric}_by_slots.eps"), format='eps')
        fig.show()
        plt.close(fig)


def simulate_local(comb_11kW, result_dict, subset_sizes, start_flex, end_flex, num_sim, arrival_col, departure_col):
    """
    Simulates local scenarios (micro - only a subset of entire time series data, e.g. 5 CP):
    Scenario 1: 3.7 kW charging
    1a: All session between 18:00 and 06:00 get 3.7 kW charging
    1b: All session between 18:00 and 06:00 which are able to charge the required (=original energy requirement) energy
        charge with 3.7 kW, the ones towards 06:00 are charging with 11 kW
    1c: Change start times of the sessions that start between 22:00 and 22:30 according to some probability density
        function or arrival times - this is done, because in the used data, people were mostly charging from 22:00,
        when the energy tariff switched to a cheaper tariff. The cars most likely arrived earlier.
    1d-x: Changed start times for 2200-2230 sessions, reduce from 11 kW to 3.7 kW for fraction of ids. E.g., of all
          meters, 20% are chosen, for which participation is assumed, the other ids charge without any changes

    :param comb_11kW: collection of real 11kW EV charging time series
    :param result_dict: key: scenario_name, value: list of ScenarioResults instances - name determines logic to process
                             data in comb_11kW
    :param subset_sizes: list of sizes of subsets, i.e., how many CP time series are combined, a subset size of 5 means,
                         that the EV charging data of 5 charging points is added to each other. In this context, this
                         is repeated for num_scenarios, to get an understanding of coincidence of EV charging using
                         different strategies.
    :param start_flex: Start time of flexible window
    :param end_flex: End time of flexible window
    :param num_sim: Number of scenarios, since we have a large number of charging time series, we can randomly
                          combine them to get an understanding, how local coincidence may really look like.
    :param arrival_col: List of arrival times generated out of loop to quickly get arrival times (i.e., potential
                        charging session start times) according to probability density function of previous work.
    :param departure_col: List of departure times generated out of loop to quickly get departure times (i.e., potential
                          charging session end times) according to probability density function of previous work.
    :return: result dictionary with key: scenario name, value: list of ScenarioResults instances
    """
    # Determines participation levels of local_37d scenario, same for local_slot scenario
    local_37d_participation = []
    local_slot_participation = []
    num_slot_collection = []

    # First we check scenario names:
    for key in result_dict.keys():
        assert 'local' in key, "Check scenario names, 'local' not found."

        if 'local_37d' in key:
            participation = float(key.replace('local_37d_', ''))  # Use replace instead of strip
            if participation > 1:
                participation = participation / 100
                print(f"INFO: Converting participation of scenario {key} to rel. value: {participation}")
                local_37d_participation.append(participation)

        if 'local_slot' in key:
            # Get number of slots:
            match = re.search(r"slot(\d+)", key)
            num_slots = int(match.group(1))
            assert 1 < num_slots <= 100, f"ERROR: Scenario {key} returned {num_slots} slots, out of reasonable range"
            num_slot_collection.append(num_slots)

            participation = float(key.replace(f"local_slot{num_slots}_", ""))
            if participation > 1:
                participation = participation / 100
                print(f"INFO: Converting participation of scenario {key} to rel. value: {participation}")
                local_slot_participation.append(participation)

    local_slot_participation = sorted(set(local_slot_participation))
    num_slot_collection = sorted(set(num_slot_collection))

    all_timeseries = list(comb_11kW.values())
    id_list = list(comb_11kW.keys())

    # Iterate over all subset sizes, e.g. 5 CP, 10 CP, ...
    for current_subset_size in subset_sizes:
        # Repeat for num_sim times
        for _ in tqdm(range(num_sim), desc=f'Simulation {current_subset_size} charging points 11 kW'):
            """
            Get properties for all scenarios for all ids - has to be called for all ids before selecting a subset
            to prevent getting a skewed distribution in the subset, especially for small subsets
            """
            property_df = ID_property_generator(id_list, num_slot_collection, local_slot_participation)

            """
            Calculate some scenario independent reference values
            """
            # Randomly select `n` time series
            sampled_ts = random.sample(all_timeseries, current_subset_size)
            # Get id names of the selected sample
            selected_names = [ts.name for ts in sampled_ts]
            # Step 1: Filter property_df to only include indices in names_list
            property_df_filtered = property_df.loc[property_df.index.intersection(selected_names)]
            names_by_participation = {
                level: property_df_filtered.index[property_df_filtered[f'participation_{level}'] == True].tolist()
                for level in local_37d_participation
            }
            # Compute normalization factor: sum of max values of the selected time series
            normalization_factor = sum([ts.max() for ts in sampled_ts])

            """
            Start calculation of local / micro scenarios
            """
            # Scenario 37a: Reduce all charging that starts in flexible window to 3.7 kW charging
            scenario_result_37a = scenario37(sampled_ts, 3.7, normalization_factor, start_flex, end_flex,
                                             arrival_col, departure_col)
            result_dict['local_37a'].append(scenario_result_37a)
            # Scenario 37b: Reduce all charging to 3.7 kW, if able to finish before end_flex.
            scenario_result_37b = scenario37(sampled_ts, 3.7, normalization_factor, start_flex, end_flex,
                                             arrival_col, departure_col, version_b=True)
            result_dict['local_37b'].append(scenario_result_37b)
            # Scenario 37c: Change start times of the sessions that started between 22:00 and 22:30 (those customers
            # seem to be willing to provide flexibility) according to arrival pdf, change magnitudes
            scenario_result_37c = scenario37(sampled_ts, 3.7, normalization_factor, start_flex, end_flex,
                                             arrival_col, departure_col, version_c=True)
            result_dict['local_37c'].append(scenario_result_37c)

            for par in local_37d_participation:
                # Get list of all participating ids, the ones that are not contained do not participate
                participation_id_list = names_by_participation[par]
                scenario_result_37d_par = scenario37(sampled_ts, 3.7, normalization_factor, start_flex,
                                                     end_flex, arrival_col, departure_col, version_d=True,
                                                     fraction_d_list=participation_id_list)
                result_dict[f"local_37d_{int(par*100)}"] = scenario_result_37d_par

    return result_dict


def plot_wcc_scenario_comparison(ref_data, perf_e_data, slot4_p50_data, local_37c_data, subset_sizes, result_directory,
                                 pres_mode=False):
    """
    Data can be read from pkl files.
    Compare WCC values across four scenarios:
    - ref_data: DataFrame [runs x subset size]
    - perf_e_data: Dict {subset_size: list of WCC values}
    - slot2_p50_data, local_37c_data: List of ScenarioResults (with .worst_case_coincidence and .subset_size)

    :param subset_sizes: list of subset sizes to compare (e.g., [5, 10, 20, 50])
    :param result_directory: directory to save result PNG
    :param pres_mode: if True, use wider layout
    """
    # --- Prepare data ---
    scenario_labels = ['Unbiased Plug-In', 'Perfect E', 'Slot2 P50', 'Local 37c']
    # Custom label names
    label_map = {
        'Unbiased Plug-In': 'Reference',
        'Perfect E': 'Dynamic low power',
        'Slot2 P50': 'Staggered 4, 50%',
        'Local 37c': 'Fixed low power'
    }
    viridis = plt.colormaps['viridis']
    colors = [viridis(0.2), viridis(0.45), viridis(0.65), viridis(0.85)]
    median_color = viridis(0.95)

    scenario_data = {label: {size: [] for size in subset_sizes} for label in scenario_labels}

    # 1. Reference Data (DataFrame)
    for size in subset_sizes:
        col = str(size)
        if col in ref_data.columns:
            scenario_data['Unbiased Plug-In'][size] = ref_data[col].dropna().tolist()

    # 2. Perfect E (dict of lists)
    for size in subset_sizes:
        if size in perf_e_data:
            scenario_data['Perfect E'][size] = perf_e_data[size]

    # 3. Slot2 P50 (list of ScenarioResults)
    for res in slot4_p50_data:
        if hasattr(res, 'subset_size') and hasattr(res, 'worst_case_coincidence'):
            scenario_data['Slot2 P50'][res.subset_size].append(res.worst_case_coincidence)

    # 4. Local 37c (list of ScenarioResults)
    for res in local_37c_data:
        if hasattr(res, 'subset_size') and hasattr(res, 'worst_case_coincidence'):
            scenario_data['Local 37c'][res.subset_size].append(res.worst_case_coincidence)

    # --- Plot setup ---
    num_scenarios = len(scenario_labels)
    width = 0.17

    if pres_mode:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig, ax = plt.subplots()

    box_data = []
    positions = []
    tick_positions = []
    tick_labels = []

    for i, size in enumerate(subset_sizes):
        for j, label in enumerate(scenario_labels):
            values = scenario_data[label][size]
            if len(values) > 0:
                pos = i + j * width - (width * (num_scenarios - 1) / 2)
                box_data.append(values)
                positions.append(pos)
        tick_positions.append(i)
        tick_labels.append(str(size))

    # --- Boxplot drawing ---
    bp = ax.boxplot(box_data, positions=positions, widths=width, patch_artist=True,
                    flierprops=dict(marker='.', markersize=1.1, linestyle='none', alpha=0.6),
                    medianprops=dict(color=median_color, linewidth=1.2))

    # Apply colors
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % num_scenarios])

    # X-ticks & labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Number of CP")
    ax.set_ylabel("Worst Case Normalized Power")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Legend
    handles = [plt.Line2D([], [], color=colors[i], lw=4, label=label_map[scenario_labels[i]])
               for i in range(num_scenarios)
    ]
    ax.legend(handles=handles, loc="upper right")

    # Save
    fig.tight_layout()
    output_path = os.path.join(result_directory, "wcc_scenario_comparison.png")
    fig.savefig(output_path, dpi=500)
    plt.show()



def round_to_nearest_15(minutes: float):
    """Round time (decimal hours) to the nearest 15-minute interval."""
    quarter_hours = np.round(minutes * 4) / 4  # Convert to nearest quarter-hour
    return quarter_hours


def round_dttime_to_next_15(time_obj: dt.datetime):
    """
    Rounds a datetime object up to the next 15-minute interval.
    """
    # Compute the next multiple of 15 minutes
    minute = ((time_obj.minute // 15) + 1) * 15

    if minute == 60:  # If rounding pushes to the next hour
        time_obj = time_obj.replace(hour=(time_obj.hour + 1) % 24, minute=0)
    else:
        time_obj = time_obj.replace(minute=minute)

    # Ensure seconds and microseconds are set to zero
    time_obj = time_obj.replace(second=0, microsecond=0)

    return time_obj


def times_from_pdf(csv_file, num_samples, mode):
    """
    Is used to generate arrival/departure times from pdf
    Samples `num_samples` values from a probability density function (PDF) given in a CSV file.
    Ensures all sampled values are within the time range 15:00 - 22:30.

    This is done like this to prevent a lot of function calls, since the optimization will require lots of time already

    :param csv_file: Path to the CSV file containing the PDF.
                     Assumes the first column contains time values (X-axis),
                     and the second column contains PDF values (Y-axis).
    :param num_samples: Number of samples to generate.
    :param mode: 'arrival' or 'departure'
    :return: Numpy array of sampled values.
    """
    assert mode == 'arrival' or mode == 'departure', \
        f"mode: {mode} not supported - choose either 'arrival' or 'departure'"

    # Read the CSV file
    df = pd.read_csv(csv_file, delimiter=';', header=None)

    # Extract time values (X) and corresponding probabilities (Y)
    x_values = df.iloc[:, 0].values  # Time of day
    pdf_values = df.iloc[:, 1].values  # Probability density values

    # Normalize PDF values
    pdf_values = pdf_values / np.sum(pdf_values)

    # Filter valid times (15:00 to 22:30)
    if mode == 'arrival':
        valid_mask = (x_values >= 15.00) & (x_values <= 22.50)  # 22:30 = 22.50 in decimal format
    else:
        valid_mask = (x_values >= 6.00) & (x_values <= 10.00)
    x_valid = x_values[valid_mask]
    pdf_valid = pdf_values[valid_mask]

    # Normalize again after filtering
    pdf_valid = pdf_valid / np.sum(pdf_valid)

    # Generate samples, re-sampling if needed
    samples = np.random.choice(x_valid, size=num_samples, p=pdf_valid)

    rounded_samples = round_to_nearest_15(samples)

    return np.round(rounded_samples, 2)


def simulate_local_worker(args):
    """
    Worker function for parallel execution of local simulations.
    """
    (sampled_ts, normalization_factor, start_flex, end_flex, arrival_col, departure_col, local_37d_participation,
     names_by_participation) = args

    results = {'local_37a': scenario37(sampled_ts, 3.7, normalization_factor, start_flex, end_flex,
                                       arrival_col, departure_col),
               'local_37b': scenario37(sampled_ts, 3.7, normalization_factor, start_flex, end_flex,
                                       arrival_col, departure_col, version_b=True),
               'local_37c': scenario37(sampled_ts, 3.7, normalization_factor, start_flex, end_flex,
                                       arrival_col, departure_col, version_c=True)}

    # Scenario 37d: Different participation levels
    for par in local_37d_participation:
        participation_id_list = names_by_participation[par]
        results[f"local_37d_{int(par * 100)}"] = scenario37(
            sampled_ts, 3.7, normalization_factor, start_flex, end_flex, arrival_col, departure_col,
            version_d=True, fraction_d_list=participation_id_list
        )

    return results


# Needed for parallelization
ALL_TIMESERIES = None


def initializer(timeseries_data):
    global ALL_TIMESERIES
    ALL_TIMESERIES = timeseries_data


def initialize_slot_worker(ts_data):
    global ALL_SLOTTED_TIMESERIES
    ALL_SLOTTED_TIMESERIES = ts_data


def simulate_one_run_wrapper(current_subset_size, flexible_start_time, flexible_end_time, dtc, p_threshold):
    return simulate_one_run_e(
        ALL_TIMESERIES,
        current_subset_size,
        flexible_start_time,
        flexible_end_time,
        dtc,
        p_threshold
    )


def simulate_local_parallel(comb_11kW, result_dict, subset_sizes, start_flex, end_flex, num_sim, arrival_col,
                            departure_col):
    """
    Parallelized version of simulate_local using ProcessPoolExecutor.
    """
    local_37d_participation = []
    num_slot_collection = []

    for key in result_dict.keys():
        if 'local_37d' in key:
            participation = float(key.replace('local_37d_', '')) / 100
            local_37d_participation.append(participation)

    local_37d_participation = sorted(set(local_37d_participation))

    all_timeseries = list(comb_11kW.values())
    id_list = list(comb_11kW.keys())

    tasks = []

    for current_subset_size in subset_sizes:
        for _ in tqdm(range(num_sim), desc=f"Initializing simulation for {current_subset_size} CP"):
            sampled_ts = random.sample(all_timeseries, current_subset_size)
            selected_names = [ts.name for ts in sampled_ts]

            property_df = ID_property_generator(id_list, num_slot_collection, local_37d_participation)
            property_df_filtered = property_df.loc[property_df.index.intersection(selected_names)]

            names_by_participation = {
                level: property_df_filtered.index[property_df_filtered[f'participation_{level}'] == True].tolist()
                for level in local_37d_participation
            }

            normalization_factor = sum([ts.max() for ts in sampled_ts])

            tasks.append((sampled_ts, normalization_factor, start_flex, end_flex, arrival_col, departure_col,
                          local_37d_participation, names_by_participation))

    # Get infos on core for parallelization
    num_cores = os.cpu_count()
    workers = max(1, num_cores - 1)

    # Run simulations in parallel
    """
    If this causes problems, set break point before this and execute the following code manually. 
    """
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results_list = list(
            tqdm(executor.map(simulate_local_worker, tasks), total=len(tasks),
                 desc=f"Simulating Local Scenarios using {workers}/{num_cores} cores"))

    # Aggregate results
    for scenario_results in results_list:
        for key, value in scenario_results.items():
            result_dict[key].append(value)

    return result_dict


def simulate_perfect_e(comb_11_new_arrival, subset_sizes, flexible_start_time, flexible_end_time, num_sim, dtc,
                       p_threshold=3.7):
    """
    Parallelization of simulation with perfect knowledge on session energy consumption.
    :param comb_11_new_arrival: time series data with new arrival times according to arrival distribution
    :param subset_sizes: number of charging points (:= CP)
    :param flexible_start_time: start time of the flexible time range
    :param flexible_end_time: end time of the flexible time range
    :param num_sim: number of iterations for each number of CP
    :param dtc: list containing departure times according to departure time distribution
    :param p_threshold: minimum charging speed in kW
    :return:
    """

    flex_dict = defaultdict(list)
    dep_dict = defaultdict(list)
    all_timeseries = list(comb_11_new_arrival.values())

    with ProcessPoolExecutor(initializer=initializer, initargs=(all_timeseries,)) as executor:
        for current_subset_size in subset_sizes:
            tasks = [
                executor.submit(
                    simulate_one_run_wrapper,
                    current_subset_size,
                    flexible_start_time,
                    flexible_end_time,
                    dtc,
                    p_threshold
                )
                for _ in range(num_sim)
            ]

            for future in tqdm(as_completed(tasks), total=num_sim, desc=f"Subset size {current_subset_size}"):
                wcc_flex, wcc_dep = future.result()
                flex_dict[current_subset_size].append(wcc_flex)
                dep_dict[current_subset_size].append(wcc_dep)

    return flex_dict, dep_dict


def simulate_one_run_e(all_timeseries, current_subset_size, flexible_start_time, flexible_end_time, dtc, p_threshold):
    """
    Running one iteration of a specific subset size - needed for parallelization

    For this scenario, flexibility is assumed for all sessions starting between flexible_start_time and
    flexible_end_time and ending before flexible_end_time. For each of these sessions, the session energy is calculated
    and the power of the session is adjusted such that it is able to finish until flexible_end_time / its assigned
    departure time. So the magnitude is reduced to its minimum for all sessions.

    :param all_timeseries: Collection of 11kW EV charging time series (dict with key: id (str) , value: time
                           series data (pd.Series)
    :param current_subset_size: denoting the number of charging points, e.g. [5,10,30,100]
    :param flexible_start_time: The start time of the flexible charging window
    :param flexible_end_time: The end time of the flexible charging window
    :param dtc: list of departure times following a probability distribution form another publication
    :param p_threshold: Lower boundary of the charging speed - a lot of onboard chargers require a certain minimum
                        charging current/power.
    :return: result dictionary with worst case coincidence for 'flex' and 'departure' as keys, list with the
             corresponding values for all scenarios. In this scenario no failed sessions are possible
    """

    sampled_ts = random.sample(all_timeseries, current_subset_size)
    normalization_factor = sum(ts.max() for ts in sampled_ts)

    # For flexible time range
    adjusted_flex = []
    # For departure based time range (departure > flexible_end_time)
    adjusted_dep = []

    for ts in sampled_ts:
        ts_flex = ts.copy()
        ts_dep = ts.copy()
        timestamps = ts.index
        session_starts = (ts.shift(1, fill_value=0) == 0) & (ts > 0)
        start_indices = np.where(session_starts)[0]

        for idx in start_indices:
            session_start_time = timestamps[idx]
            session_end_idx = idx
            while session_end_idx < len(ts) - 1 and ts.iloc[session_end_idx] > 0:
                session_end_idx += 1

            session_end_dt = timestamps[session_end_idx]
            original_magnitude = ts.iloc[idx]
            original_energy = (session_end_idx - idx) * original_magnitude * 0.25

            # Skip sessions outside flexible window
            if not (
                (flexible_start_time <= session_start_time.time() <= dt.time(23, 59)) or
                (dt.time(0, 0) <= session_start_time.time() < flexible_end_time)
            ):
                continue
            # Skip late night sessions that naturally go beyond flexible_end_time
            if session_end_dt.time() >= flexible_end_time:
                continue

            # Handle overmidnight
            if session_start_time.time() < flexible_end_time:
                ref_date = session_start_time.date() - timedelta(days=1)
            else:
                ref_date = session_start_time.date()

            flex_end_dt = dt.datetime.combine(ref_date + timedelta(days=1), flexible_end_time)
            dep_hour, dep_minute = divmod(int(random.choice(dtc) * 60), 60)
            dep_dt = flex_end_dt.replace(hour=dep_hour, minute=dep_minute)

            def adjust_ts(ts_adj, deadline_dt):
                delivery_window_h = (deadline_dt - session_start_time).total_seconds() / 3600
                if delivery_window_h <= 0:
                    return ts_adj
                adjusted_power = max(original_energy / delivery_window_h, p_threshold)
                adjusted_duration_intervals = int(np.ceil(original_energy / (adjusted_power * 0.25)))
                ts_adj[idx:session_end_idx] = 0
                ts_adj[idx:idx + adjusted_duration_intervals] = adjusted_power
                return ts_adj

            ts_flex = adjust_ts(ts_flex, flex_end_dt)
            ts_dep = adjust_ts(ts_dep, dep_dt)

        adjusted_flex.append(ts_flex)
        adjusted_dep.append(ts_dep)

    wcc_flex = pd.DataFrame(adjusted_flex).sum(axis=0).max() / normalization_factor
    wcc_dep = pd.DataFrame(adjusted_dep).sum(axis=0).max() / normalization_factor

    return wcc_flex, wcc_dep


def plot_perfect_e_wcc(flex_dict, dep_dict, res_direc=None, presentation_mode=False):
    """
    Plots side-by-side boxplots for WCC of 'flex' and 'dep' scenarios across subset sizes. Stores the underlying
    data for reproducibility

    :param flex_dict: Dictionary with subset size as keys and list of WCC values for flexible_end_time deadlines.
    :param dep_dict: Dictionary with subset size as keys and list of WCC values for departure time deadlines.
    :param res_direc: Directory to save plots.
    :param presentation_mode: If True, sets larger figure size.
    """
    subset_sizes = sorted(flex_dict.keys())
    labels = ['fix', 'dep']
    width = 0.3

    if presentation_mode:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig, ax = plt.subplots()

    positions = []
    box_data = []

    for i, size in enumerate(subset_sizes):
        flex_vals = flex_dict.get(size, [])
        dep_vals = dep_dict.get(size, [])

        box_data.extend([flex_vals, dep_vals])
        positions.extend([i - width/2, i + width/2])  # Flex left, dep right

    colors = [plt.cm.viridis(0.25), plt.cm.viridis(0.75)]
    median_color = plt.cm.viridis(0.95)

    # Plot boxplots
    bp = ax.boxplot(box_data, positions=positions, widths=width, patch_artist=True,
                    flierprops=dict(marker='.', markersize=2, linestyle='none', alpha=0.6))

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % 2])
    for median in bp['medians']:
        median.set_color(median_color)
        median.set_linewidth(1.2)

    ax.set_xticks(np.arange(len(subset_sizes)))
    ax.set_xticklabels(subset_sizes)
    ax.set_xlabel("Number of CP (Charging Points)")
    ax.set_ylabel("Worst Case Normalized Power")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend([plt.Line2D([0], [0], color=c, lw=4) for c in colors], labels, loc='upper right')

    plt.tight_layout()

    if res_direc:
        os.makedirs(res_direc, exist_ok=True)
        filename = 'wcc_perfect_e_comparison'
        fig.savefig(os.path.join(res_direc, f"{filename}.png"), dpi=500)
        fig.savefig(os.path.join(res_direc, f"{filename}.eps"), format='eps')

        # Dump data for reproducibility
        dump_plot_data(flex_dict, "plot_perfect_e_wcc", filename, var_name="flex_dict", data_direc=res_direc)
        dump_plot_data(dep_dict, "plot_perfect_e_wcc", filename, var_name="dep_dict", data_direc=res_direc)

    plt.show()


def main():
    ALL_TIMESERIES = None
    global ALL_SLOTTED_TIMESERIES

    # Path to .parquet file with 11kW charging time series, change as needed
    fpath = ("//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/unlabelled_smd/results/11kw_charging_timeseries/"
             "charging_timeseries.parquet")
    assert os.path.exists(fpath), ("Change fpath in main function of this py file to loc of 11 kW charging time series "
                                   ".parquet file")
    # result directory, change as needed
    result_direc = "//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/optimization/results/"
    # path to pdf of weekday arrival times
    arrival_times_path = ("//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/optimization/"
                          "pdf_arrival_time_weekdays_giorgi_et_al_2021.csv")
    # Arrival Times Collection
    atc = times_from_pdf(arrival_times_path, 10_000, 'arrival')
    departure_times_path = ("//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/optimization/"
                            "pdf_departure_distribution_Romano_et_al_2024.csv")
    # Departure times collection
    dtc = times_from_pdf(departure_times_path, 10_000, 'departure')

    # Set to True to start Monte-Carlo Simulation of the 11kW charging behavior
    sim_local_coincidence = True

    # Read 11 kW charging time series data
    df = pd.read_parquet(fpath)
    comb_11kW_collection = {col: df[col].dropna() for col in df.columns}
    # Something went wrong in data saving for some ids - those get excluded here:
    comb_11kW_collection = fix_11kW_collection(comb_11kW_collection)

    """
    For the slot scenario, all the charging from 22:00-22:30 gets moved to new arrival times. Theoretically one would 
    have to assign new arrival time for each session and each iteration - it is assumed that due to the large number
    of sessions, this is a simplification that should not affect the results. 
    """
    comb_11_new_arrival = preprocess_timeseries_for_slots(comb_11kW_collection, atc)
    ALL_SLOTTED_TIMESERIES = list(comb_11_new_arrival.values())

    # 100'000 takes approximately 2 hours
    if sim_local_coincidence:
        print(f"\n{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Start simulation of wcc for reference scenario / "
              f"Uncontrolled without arrival reassignment\n")
        wc_collection = plot_worst_case_coincidence_parallel(comb_11kW_collection, result_direc + 'reference',
                                                             'eps', num_simulations=100000)
        dump_plot_data(wc_collection, 'plot_worst_case_coincidence_parallel',
                       'results/reference_scenario')

        print(f"\n{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Start simulation of wcc for reference scenario / "
              f"with arrival reassignment\n")
        wc_collection_new_arrival = plot_worst_case_coincidence_parallel(comb_11_new_arrival, result_direc +
                                                                         'reference_new_arrival', 'eps',
                                                                         num_simulations=100000)
        dump_plot_data(wc_collection_new_arrival, 'plot_worst_case_coincidence_parallel',
                       'results/reference_scenario_new_arrival')

        concurrency_boxplot(wc_collection_new_arrival, wc_collection, result_direc+'wcc_boxplot_comparison.png')
        concurrency_boxplot(wc_collection_new_arrival, wc_collection, result_direc+'wccp_boxplot_comparison.png',
                            power_scale=11)

    """
    Simulation configuration:
    """
    # Define flexible time-range, default: 18:00-06:00
    flexible_start_time = dt.time(18, 00)
    flexible_end_time = dt.time(6, 0)

    # Define earliest possible starting slot:
    slot_start = dt.time(20, 0)
    slot_stop = dt.time(1, 0)
    num_slot_list = [2, 4, 8]

    # Scenario configuration
    participation_levels = [25, 50, 75, 100] # for 37d and slot scenarios
    subset_sizes = [5, 10, 20, 30, 50, 100]
    slot_config = generate_slot_config(slot_start, slot_stop, num_slot_list)
    # Local and global should not be misunderstood --> micro vs macro analysis of charging strategies
    scenario_types = ["local", "global"]
    scenario_variants = ["37a", "37b", "37c", "37d", "slot2", "slot4", "slot8"]

    # Separate dictionaries for local and global scenarios
    local_results = {}
    global_results = {}
    pres_mode = False

    num_sim = 20000
    print(f"\n{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Start Simulation scenario perfect energy information\n")

    # Run perfect energy knowledge (energy for all charging sessions is known at beginning of session) simulation
    flex_dict, dep_dict = simulate_perfect_e(comb_11_new_arrival, subset_sizes, flexible_start_time, flexible_end_time,
                                             num_sim, dtc)
    # Data dumping happens inside plot_perfect_e_wcc
    plot_perfect_e_wcc(flex_dict, dep_dict, res_direc=result_direc + 'perfect_e/', presentation_mode=pres_mode)

    # Initialize results - for each scenario empty list, which contains ScenarioResults instances
    for scenario in scenario_types:
        scenario_dict = local_results if scenario == "local" else global_results  # Choose the right dictionary

        for variant in scenario_variants:
            if variant in ["37a", "37b", "37c"]:  # Scenarios without participation levels
                scenario_dict[f"{scenario}_{variant}"] = []
            else:  # Scenarios with participation levels
                for level in participation_levels:
                    scenario_dict[f"{scenario}_{variant}_{level}"] = []

    single_plotting_metrics = ['worst_case_coincidence', 'missing_energy_fraction', 'missing_energy_fraction_dep']
    # Metrics to be plotted, second column denotes the normalization metric
    normalized_plotting_metrics = [['num_failed_sessions', 'num_sessions_total'],
                                   ['num_failed_sessions_dep', 'num_sessions_total'],
                                   ['num_modified_sessions', 'num_sessions_total'],
                                   ['num_failed_sessions', 'num_modified_sessions']]

    print(f"\n{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Starting simulation for local 37 scenarios\n")
    local_results = simulate_local_parallel(comb_11kW_collection, local_results, subset_sizes, flexible_start_time,
                                            flexible_end_time, num_sim, atc, dtc)
    dump_plot_data(local_results, 'simulate_local_parallel',
                   'results/local_37scenarios')

    for metr in single_plotting_metrics:
        plot_37_results(local_results, participation_levels, subset_sizes, res_direc=result_direc, metric=metr,
                        plot_37abcd25=True, presentation_mode=pres_mode)
        plot_37_results(local_results, participation_levels, subset_sizes, res_direc=result_direc, metric=metr,
                        plot_37abcd25=False, presentation_mode=pres_mode)
    for metr in normalized_plotting_metrics:
        first, reference = metr
        plot_37_results(local_results, participation_levels, subset_sizes, res_direc=result_direc, metric=first,
                        plot_37abcd25=True, presentation_mode=pres_mode, normalize_by=reference)
        plot_37_results(local_results, participation_levels, subset_sizes, res_direc=result_direc, metric=first,
                        plot_37abcd25=False, presentation_mode=pres_mode, normalize_by=reference)

    num_sim = 20000
    print(f"\n{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}Start simulation of slot scenarios\n")
    slot_results = simulate_slot_parallel(comb_11_new_arrival, slot_config, num_slot_list, slot_start,
                                          participation_levels, subset_sizes, flexible_start_time, flexible_end_time,
                                          num_sim, dtc)
    dump_plot_data(slot_results, 'plot_slot_results_by_slot_and_participation',
                   'results/slot_results')
    """
    Single core implementation
    slot_results = simulate_slot_single_core(comb_11_new_arrival, slot_config, num_slot_list, slot_start,
                                             participation_levels, subset_sizes, flexible_start_time, flexible_end_time,
                                             num_sim, dtc)
    """
    # Plotting slot scenarios
    for metr in single_plotting_metrics:
        plot_slot_results_by_slot_and_participation(
            result_dict=slot_results,
            participation_levels=participation_levels,
            subset_sizes=subset_sizes,
            num_slot_list=num_slot_list,
            res_direc=result_direc,
            metric=metr,
            presentation_mode=pres_mode
        )

    for metr, norm in normalized_plotting_metrics:
        plot_slot_results_by_slot_and_participation(
            result_dict=slot_results,
            participation_levels=participation_levels,
            subset_sizes=subset_sizes,
            num_slot_list=num_slot_list,
            res_direc=result_direc,
            metric=metr,
            normalize_by=norm,
            presentation_mode=pres_mode
        )

    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Done")


if __name__ == "__main__":
    main()

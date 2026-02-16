"""
Author: Lukas Elmiger, luki.elmiger@hotmail.com, luelmiger@ethz.ch, lukas.elmiger@ckw.ch
Date of first contribution: 12.02.2025

This file is used to identify EV-charging activity in yearly SMD using optimized parameters.

Also offers the possibility to plot certain metrics like annual EV charging energy consumption for sanity checking of
the extracted charging sessions. Another main output are the plots of important EV-charging modelling metrics:
- EV charging power distribution
- Energy consumption per charging session, per charging power (if there is data for each charging power) distribution
- Number of days between consecutive charging sessions distribution
- Charging start time distribution

Input: File path containing all the preprocessed SMD (using private_smd_preprocessing.py)
Output: Result folder in the same directory containing all the extracted information.
"""
import pickle
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

from datetime import timedelta
import datetime as dt

import os
import time
import statistics
import random
import copy
import concurrent.futures
import multiprocessing

from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import STL
import pywt

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scienceplots
plt.style.use('science')

import warnings
warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency")


"""
temp - delete this
"""
path = ("//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/unlabelled_smd/sorted_by_id/"
        "CH1003601234500000000000000006601.csv")

"""
Helper methods
"""


def ask_directory(window_string: str = 'Select directory'):
    """
    Ask user for a directory
    """
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 2.0)

    directory = filedialog.askdirectory(title=window_string)

    root.withdraw()
    return directory


def read_15min_energy_smd(fpath: str) -> pd.Series:
    """
    Read 15 min energy smart meter data. Return pd.Series with timestamp as datetime object and the energy value.
    :param fpath: Path to the .csv file
    :return: pd.Series of the energy data, indices are timestamps (datetime objects)
    """
    df = pd.read_csv(fpath)
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except KeyError:
        df['timestamp'] = pd.to_datetime(df['index'])
    df.set_index('timestamp', inplace=True)
    series = df['value_kwh']
    return series


def create_result_dir(input_directory):
    """
    Creates a directory named 'extraction_fig_plots' inside the 'results' folder,
    which is located in the parent directory of the given input directory.

    Parameters:
    - input_directory (str): The path to the input directory.

    Returns:
    - str: The path to the newly created 'extraction_fig_plots' directory.
    """
    # Handle input file_paths that have a slash at the end
    if os.path.basename(input_directory) == '':
        input_directory = os.path.dirname(input_directory)

    # Get the parent directory of the input directory
    parent_directory = os.path.dirname(input_directory)

    # Define the path for the 'results' folder
    results_directory = os.path.join(parent_directory, "results")

    # Define the path for 'extraction_fig_plots' inside 'results'
    extraction_fig_plots_directory = os.path.join(results_directory, "extraction_fig_plots")

    # Create the directories if they do not exist
    os.makedirs(extraction_fig_plots_directory, exist_ok=True)


def shift_time_dict(input_dict):
    """
    Converts summer-time dictionary to winter time dictionary -> keys: datetime.time get one hour added
    :param input_dict:
    :return:
    """
    shifted_dict = {}
    for t, value in input_dict.items():
        # Add 1 hour and handle overflow
        new_time = (dt.datetime.combine(dt.date.today(), t) + timedelta(hours=1)).time()

        # If the new time already exists, sum the values
        if new_time in shifted_dict:
            shifted_dict[new_time] += value
        else:
            shifted_dict[new_time] = value

    return shifted_dict


def get_extraction_fig_plots_path(input_directory) -> str:
    """
    Returns directory for plots of charging session extractions.
    :param input_directory: Input directory, in which the .csv files are located.
    :return: directory for plots, creates directories, if they do not exist yet.
    """
    while True:
        if not os.path.exists(input_directory):
            ask_directory("Directory not found.")
        else:
            break

    if os.path.basename(input_directory) == '':
        input_directory = os.path.dirname(input_directory)

    ret_dir = os.path.dirname(input_directory) + "/results/extraction_fig_plots/"
    if not os.path.exists(ret_dir):
        create_result_dir(input_directory)

    return ret_dir


"""
Charging session identification
"""


class IDecomp:
    """
    Object for each ID which contains the needed data for the extraction of charging profiles.
    """
    def __init__(self, filepath):
        discrete_power_level = [3.7, 5.5, 7.4, 11.09, 17.6, 22.17]

        assert os.path.exists(filepath)
        self.data_file_path = filepath

        self.id = os.path.basename(filepath).strip('.csv')

        self.session_data = None
        self.pairs = None

        # Read 15-min average time series energy data from .csv files
        data = read_15min_energy_smd(filepath)

        # Convert 15-min energy consumption to 15-min avg power values
        self.df_time_series_kw = 4 * data

        # If there are somehow still duplicates, they are handled here
        self.df_time_series_kw = self.df_time_series_kw.groupby(self.df_time_series_kw.index).max()

        # Timeseries with the EXTRACTED charging sessions - series of zeros as default
        self.extracted_session_timeseries_kW = self.df_time_series_kw - self.df_time_series_kw

        self.start_timestamp = self.df_time_series_kw.index.min()
        self.end_timestamp = self.df_time_series_kw.index.max()

        # Make sure to only use one year of data here - other params needed for other timeframes.
        self.detected_peak = find_peak_in_df(self.df_time_series_kw)

        # tuned magnitude - is equal to initially detected peak until tuning
        self.tuned_magnitude = self.detected_peak

        if self.detected_peak != 0:
            self.closest_discrete_value = min(discrete_power_level, key=lambda x: abs(x - self.detected_peak))
        else:
            self.closest_discrete_value = 0

        self.stl_trend_kw = None
        self.filtered_signal_kW = None

    def compute_stl(self, stl_period: int, stl_seasonal: int):
        """
        computes stl trend component of the time series power signal
        """
        # Set stl_trend time series to original load timeseries minus base load (could be further refined with local
        # base load estimation)
        power_timeseries = self.df_time_series_kw - self.df_time_series_kw.min()

        # Yes, I haven't renamed the column, but trust me, its kW - if you got here, you'll figure it out ;)
        stl = STL(power_timeseries, seasonal=stl_seasonal, period=stl_period)
        result = stl.fit()
        trend = result.trend
        trend_df = pd.DataFrame(trend)
        self.stl_trend_kw = trend_df

    def compute_dwt(self, dwt_level: int, dwt_threshold: float):
        """
        applies dwt to the stl-trend component, stores filtered signal to self.filtered_signal_kW.
        After this step, the signal is ready for self.edge_detection
        """
        assert self.stl_trend_kw is not None, "stl_trend_kw is None, have it be calculated before."

        # The stl-trend component is the input signal to this function
        signal = self.stl_trend_kw['trend']

        wvlt = 'haar'
        coeffs = pywt.wavedec(signal, wavelet=wvlt, level=dwt_level)
        cA = coeffs[0]
        cD = coeffs[1:]

        # Apply thresholding to detail coefficients
        for i in range(len(cD)):
            cD[i] = pywt.threshold(cD[i], dwt_threshold * max(cD[i]))

        # Reconstruct the signal from the thresholded coefficients
        reconstructed_signal = pywt.waverec([cA] + cD, wvlt)

        # Create a DataFrame with the reconstructed signal and timestamps

        try:
            reconstructed_df = pd.DataFrame(reconstructed_signal, index=signal.index, columns=['filtered_signal_kW'])
        except ValueError:
            reconstructed_signal = reconstructed_signal[:len(signal)]
            reconstructed_df = pd.DataFrame(reconstructed_signal, index=signal.index, columns=['filtered_signal_kW'])
        self.filtered_signal_kW = reconstructed_df

    def edge_detection(self, edge_nu, edge_eps, edge_d_min_minutes, edge_d_max_minutes,
                       edge_min_minutes_btw_sessions, edge_theta_min, edge_theta_max):
        """
        This method is used to detect the edge-pairs of the charging windows. For this, the difference of the dwt
        power value between all timestamps is computed. Then we analyse this by the following metrics:
        - Nu: The magnitude of the difference between two timestamps has to be larger than edge_nu
        - Eps: Start_change_magnitude = - end_change_magnitude + eps
        - D_min: Minimum duration between t+ (start time) and t- (end time)
        - D_max: Maximum duration between t+ and t-
        - D_min_btw_sessions: minimum duration between charging sessions
        - Theta: Total number of charging events

        All of these metrics are dependent on the detected charging magnitude, since e.g. a 3.7 kW charging customer
        tends to charge more often (due to smaller battery size). On the other hand 11 kW charging customers usually
        have a BEV with a larger battery, charging less frequently, but longer and with higher magnitudes. This
        differentiation leads to an increased detection precision.

        Constraints to find edge-pairs:
            Constraint 1: The difference of the magnitudes of the values are smaller than edge_eps
            Constraint 2: The timestamps have a minimum and a maximum timedelta (timedelta_min, timedelta_max) between
                          each other
            Constraint 3: The positive timestamp comes before the negative timestamp (switch on, before switch off)
            Constraint 4: The time between two paired timestamps is called session. It is not allowed to have
                          overlapping sessions.
            Constraint 5: The total number of sessions (=number of pairs) is smaller equal than edge_theta

        :param edge_nu:
        :param edge_eps:
        :param edge_d_min_minutes:
        :param edge_d_max_minutes:
        :param edge_min_minutes_btw_sessions: Minimum number of minutes between adjacent sessions
        :param edge_theta_min: Minimum number of sessions
        :param edge_theta_max: Maximum number of sessions
        :return:
        """
        pairs = []
        sessions = []

        assert self.filtered_signal_kW is not None, "Filtered signal is None, apply two filter stages first."

        diff = self.filtered_signal_kW['filtered_signal_kW'].diff().fillna(0)

        t_on_candidates = diff[diff > edge_nu]
        t_off_candidates = diff[diff < - edge_nu]
        if len(t_on_candidates) == 0 or len(t_off_candidates) == 0:
            self.pairs = pairs
            return pairs

        if len(t_on_candidates) < edge_theta_min or len(t_off_candidates) < edge_theta_min:
            self.pairs = pairs
            return pairs

        neg_times = t_off_candidates.index
        neg_values = t_off_candidates.values

        edge_d_min = pd.Timedelta(minutes=edge_d_min_minutes)
        edge_d_max = pd.Timedelta(minutes=edge_d_max_minutes)
        d_between_min = pd.Timedelta(minutes=edge_min_minutes_btw_sessions)

        for pos_time, pos_value in t_on_candidates.items():
            # Limit the search space, not all timestamp combinations need to be checked
            start_idx = neg_times.searchsorted(pos_time + edge_d_min)
            end_idx = neg_times.searchsorted(pos_time + edge_d_max, side='right')

            for neg_time, neg_value in zip(neg_times[start_idx:end_idx], neg_values[start_idx:end_idx]):
                # Constraint 1
                if abs(pos_value + neg_value) < edge_eps:
                    # Constraints 2-4
                    if all(not (session[0] < pos_time < session[1] or session[0] < neg_time < session[1]) and
                           (session[1] + d_between_min <= pos_time or neg_time + d_between_min <= session[0]) for
                           session in sessions):
                        pairs.append((pos_time, neg_time))
                        sessions.append((pos_time, neg_time))
                        # Constraint 5 - returns full list --> the rest of the sessions will not be delivered and hence
                        # result in a larger error
                        if len(pairs) >= edge_theta_max:
                            self.pairs = pairs
                            return pairs
        self.pairs = pairs

        return pairs

    def tune_amplitude(self):
        """
        Having gotten potential pairs of start-/end timestamps of charging sessions, the magnitudes need to be tuned,
        since the peak detection method returns the maximum possible magnitude, e.g. 11.09 kW. The real charging speed
        depends on the onboard charger and might be slightly smaller (up to 5 % in the synthetic dataset).

        We assume that the magnitude is constant throughout the year. If the customer has a variable charging profile,
        they probably use an energy management system or dynamic tariff, which usually translates to a grid friendly
        consumption behavior that does not necessarily need to be detected.

        In this method we have a look at the timeseries data between and adjacent to the potential sessions. Taking the
        power distribution of adjacent time steps into consideration, we can deduce a charging magnitude that minimizes
        the difference in variance between the charging and in a window before/after charging. This is done for all
        potential sessions, and we take the minimal power value as the presumed charging magnitude, if it lies within
        certain boundaries.
        :return:

        NOTE CURRENT IMPLEMENTATION: For now simple solution: Analyze baseload around sessions, make sure, that in flat
        top region of charging, the aggregated signal - charging signal - baseload > 0. Return 97.5 % of known charging
        magnitude, if no negative values can be found in agg - charging - baseload
        """
        assert self.pairs is not None, f"Run edge detection before amplitude tuning., ID: {self.id}"

        initial_magnitude = self.detected_peak

        pairs = self.pairs

        amplitude_estimation = []

        for pair in pairs:
            start, stop = pair[0], pair[1]

            start_window = max(self.start_timestamp, start - pd.Timedelta(hours=6))
            stop_window = min(self.end_timestamp, stop + pd.Timedelta(hours=6))

            # We only take sessions longer than 6h to make sure that the edges of the sessions do not lead to
            # false conclusions
            if stop - start < pd.Timedelta(hours=6):
                continue

            # Assume that here there is real charging activity
            mid_window_start = start + pd.Timedelta(hours=2)
            mid_window_stop = stop - pd.Timedelta(hours=2)

            local_timeseries = self.df_time_series_kw.loc[start_window:stop_window]
            # Subtract baseload
            assert (local_timeseries.min() >= 0), f"Encountered baseload < 0 for id: {self.id}"

            local_timeseries -= local_timeseries.min()

            local_timeseries = local_timeseries.loc[mid_window_start:mid_window_stop]

            """
            Check descending amplitudes: If original timeseries - charging time series yields negative values, charging
            amplitude needs to be decreased. Done until we find a value, for which no negative values result (limit:
            90 % of known charging level). If no charging level can be found, that fulfills this condition, 95 % of the
            known charging level is returned.
            """
            delta_candidates = [self.detected_peak]

            sub_df = local_timeseries - self.detected_peak

            # If initial amplitude leads to neg values, decrease charging magnitude
            if (sub_df < 0).any().any():
                # Check negativity for decreasing magnitudes
                for delta in np.arange(self.detected_peak, 0.89*self.detected_peak, -0.01*self.detected_peak):
                    sub_df = local_timeseries - delta
                    # if no negative values with this power value, go to next pair
                    if not (sub_df < 0).any().any():
                        delta_candidates.append(delta)
                        break
                    else:
                        continue

            amplitude_estimation.append(min(delta_candidates))
        try:
            ret_value = statistics.mean(amplitude_estimation)
        # this is for the ids without detected sessions longer than 6 hours (also 22 kW and 17.6 kW ids)
        except statistics.StatisticsError:
            ret_value = self.detected_peak * 0.975

        if ret_value < 0:
            print(f"Value smaller zero for id: {self.id}")

        if ret_value == initial_magnitude:
            self.tuned_magnitude = 0.975*initial_magnitude
            return self.tuned_magnitude
        else:
            self.tuned_magnitude = ret_value
            return self.tuned_magnitude

    def edge_tuning(self, shape='rect'):
        """
        This function can be used to tune the detected edges (15 min -resolution) - might be able to get more precision
        using squared gradient and variance of baseload before and after charging event. To be tuned with a separate
        optimization routine.

        The current version identifies the largest change in power between adjacent timestamps within a window around
        the identified charging pairs. It first identifies the largest peak and then checks within +- 30 min, whether
        the aggregated signal is larger than the detected charging peak.

        NOTE: not used atm, worsens the overall identification performance

        :param shape: shape type, for now only rect is implemented
        :return:
        """
        if shape != 'rect':
            raise NotImplementedError("Edge tuning only implemented for rectangular charging shapes.")

        pairs = self.pairs
        updated_pairs = []

        for pair in pairs:
            # Find improved start stop for each pair
            start, stop = pair[0], pair[1]

            # For each start and stop timestamp, we define a time window _0, _1 within which we expect the switching
            start_0 = min(max(self.start_timestamp, start - pd.Timedelta(hours=2)), self.end_timestamp)
            start_1 = min(max(self.start_timestamp, start + pd.Timedelta(hours=2)), self.end_timestamp)

            stop_0 = min(max(self.start_timestamp, stop - pd.Timedelta(hours=2)), self.end_timestamp)
            stop_1 = min(max(self.start_timestamp, stop + pd.Timedelta(hours=2)), self.end_timestamp)

            if start_0 == start_1 or stop_0 == stop_1:
                print(f"Triggered edge_tuning start-stop")
                continue

            # Power time series
            start_timeseries = self.df_time_series_kw.loc[start_0:start_1].diff().fillna(0)
            on_peak = start_timeseries.max().value_kwh
            # Find the index of the first value greater than 0.5 of the known charging power
            on_timestamp = start_timeseries[start_timeseries['value_kwh'] > 0.8*on_peak].index[0]

            start_range = pd.date_range(on_timestamp-pd.Timedelta(minutes=30), on_timestamp+pd.Timedelta(minutes=30),
                                        freq='15min')
            for timestamp in start_range:
                if self.df_time_series_kw.loc[timestamp]['value_kwh'] >= 0.95*self.detected_peak:
                    on_timestamp = timestamp
                    break

            if start != on_timestamp:
                start_candidate = on_timestamp
                #print(f"{self.id} Adjusted charging start time from {start} to {on_timestamp}")
            else:
                start_candidate = start
                #print(f"{self.id} No adjustment necessary, start time {start}")

            stop_timeseries = self.df_time_series_kw.loc[stop_0:stop_1].diff().fillna(0)
            off_peak = stop_timeseries.min().value_kwh
            off_timestamp = stop_timeseries[stop_timeseries['value_kwh'] < 0.95*off_peak].index[0]

            stop_range = pd.date_range(off_timestamp-pd.Timedelta(minutes=30), off_timestamp+pd.Timedelta(minutes=30),
                                       freq='15min')

            for timestamp in stop_range:
                if self.df_time_series_kw.loc[timestamp]['value_kwh'] < 0.8*self.detected_peak:
                    off_timestamp = timestamp
                    break

            if stop != off_timestamp:
                stop_candidate = off_timestamp
                #print(f"{self.id} Adjusted charging stop time from {stop} to {off_timestamp}")
            else:
                stop_candidate = stop
                #print(f"{self.id} No adjustment necessary, stop time {stop}")

            updated_pairs.append((start_candidate, stop_candidate))

        self.pairs = updated_pairs
        return updated_pairs


def find_peak_in_df(df_smd_kwh, num_bins=1150, p_distance=10.62, p_prom_min=15.9, p_prom_max=1728, plot=False,
                    sliding_window=True):
    """
    Finds peaks in histogram of power timeseries, which can be an indication of the used charging power. If multiple
    peaks in proximity (+- 10% of known charging levels) are found, the maximum value of the selection is returned.
    Optimization params: num_bins, p_distance, p_prominence
    :param df_smd_kwh: data frame with time series 15-min energy consumption
    :param num_bins: number of bins in the power histogram
    :param p_distance: find_peaks optimization parameter
    :param p_prom_min: Param for peak prominence optimization
                       - Assuming at least 10 h of charging per timeseries --> 40 datapoints
                       - Optimum determined: 15.9
    :param p_prom_max: Param for peak prominence optimization
                       - Assuming max of 2 MWh p.a. with 3.7 kW charger at this charging level (more charging with
                         other superpositioned appliances possible --> ~550 h = 2200 datapoints
                       - Determined optimum: 1728 datapoints
    :param plot: If true, the plot is shown
    :param sliding_window: Sliding window taking minimum within 3 steps activated (leads to better results)
    :return: Peak power value, 0 if no value could be found
    """
    discrete_power_level = [3.7, 5.5, 7.4, 11.09, 17.6, 22.17]

    p_prominence = np.array([p_prom_min, p_prom_max])

    df = df_smd_kwh.copy()

    # Convert from 15-min energy consumption to 15-min avg power values
    df *= 4

    # Get rid of baseload
    df -= df.min()

    if sliding_window:
        df = df.rolling(3).min().round(2)
        df = df.fillna(0)

    hist, bin_edges = np.histogram(df, bins=num_bins)

    peaks, peak_properties = find_peaks(hist, distance=p_distance, prominence=p_prominence)

    fig = None

    if plot:
        set_plot_params()
        # Plot the histogram and highlight the peaks
        plt.figure()
        plt.bar(bin_edges[:-1]/4, hist/4, width=np.diff(bin_edges), alpha=0.7)
        plt.scatter(bin_edges[peaks]/4, hist[peaks]/4, color='red', label='Peaks')
        plt.xlabel('Power (kW)')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.grid()
        #plt.title('Histogram with Peaks on Logarithmic Occurrence Scale')
        plt.legend()
        plt.tight_layout()

        fig = plt.gcf()
        fig.show()

    power_candidates = bin_edges[peaks]/4
    selection = []

    # Only select the candidates, that are close to the predefined power levels
    for value in power_candidates:
        for closest_value in discrete_power_level:
            if abs(value - closest_value)/closest_value < 0.1:
                selection.append(closest_value)

    if len(selection) == 0:
        #print(f"\nWARNING: Could not find peaks.")
        return 0

    elif len(set(selection)) > 1:
        #print(f"\nWARNING: More than one peak found : {str(selection)}. Returning {max(selection)} kW.")
        pass

    if plot:
        return max(selection), fig

    return max(selection)


def extract_sessions_for_file(fpath, tuning=False, tune_magnitude=True, post_process=True) -> (IDecomp, pd.Series):
    """
    This function is used to extract the charging sessions for a particular id. Returns charging time series as a
    pd.Series object. Also return IDecomp object.
    :param fpath: file path to the 15-min energy data csv
    :param tuning: Edge/Magnitude tuning - to compare results -> currently worsens overall identification performance
    :param tune_magnitude: Tunes charging magnitudes
    :param post_process: Boolean parameter to determine, whether post-processing is turned on or off
    :return:
    """
    # Optimal identification params
    edge_nu_dict = {3.7: 0.9087666233050595, 5.5: 1.2529006625402606, 7.4: 1.7662053525591945, 11.09: 2.384150955897031,
                    17.6: 3.5982392833766355, 22.17: 3.511158218022613}
    edge_eps_dict = {3.7: 0.6333124752667946, 5.5: 0.45973731358036346, 7.4: 0.6784923477730571,
                     11.09: 1.204210010943867, 17.6: 0.5089605006438872, 22.17: 0.959193706459211}
    edge_d_min_minutes_dict = {3.7: 41.93545751774573, 5.5: 44.31894595346649, 7.4: 43.96890636447061,
                               11.09: 38.95405826568586, 17.6: 40.42763970277018, 22.17: 85.82529699707163}
    edge_d_max_minutes_dict = {3.7: 1138.1057061257814, 5.5: 1135.7423103447127, 7.4: 954.847742625771,
                               11.09: 537.7434792265013, 17.6: 213.00519387090492, 22.17: 490.9465539062792}
    edge_theta_max_dict = {3.7: 165.82448919832638, 5.5: 148.6223461534799, 7.4: 197.3555844369173,
                           11.09: 171.28911557693834, 17.6: 169.77519967916228, 22.17: 142.05372758628522}

    stl_period = 4
    stl_seasonal = 69
    dwt_level = 3
    dwt_threshold = 0.00020884150733906787
    edge_min_minutes_btw_sessions = 540.7070447161025
    edge_theta_min = 47.19516743100579

    idecomp = IDecomp(fpath)
    idecomp.compute_stl(stl_period, stl_seasonal)
    idecomp.compute_dwt(dwt_level, dwt_threshold)

    # Get charging speed to determine power dependent charging identification parameters
    estimated_peak = idecomp.detected_peak
    #print(f"Estimated peak: {estimated_peak}")
    closest_discrete_value = idecomp.closest_discrete_value

    # In the case that no peak was identified
    if closest_discrete_value == 0:
        idecomp.pairs = []
        return idecomp, idecomp.df_time_series_kw-idecomp.df_time_series_kw

    # Get charging power dependent optimal charging identification parameters
    edge_nu = edge_nu_dict[closest_discrete_value]
    edge_eps = edge_eps_dict[closest_discrete_value]
    edge_d_min_minutes = edge_d_min_minutes_dict[closest_discrete_value]
    edge_d_max_minutes = edge_d_max_minutes_dict[closest_discrete_value]
    # Scale number of maximal occurrences proportionally by length of the time series
    edge_theta_max = edge_theta_max_dict[closest_discrete_value] * ((idecomp.end_timestamp -
                                                                     idecomp.start_timestamp).days / 365)

    pairs = idecomp.edge_detection(edge_nu, edge_eps,
                                   edge_d_min_minutes, edge_d_max_minutes, edge_min_minutes_btw_sessions,
                                   edge_theta_min, edge_theta_max)
    if closest_discrete_value >= 5.5 and not pairs:
        pairs = idecomp.edge_detection(edge_nu, edge_eps,
                                       edge_d_min_minutes, edge_d_max_minutes, edge_min_minutes_btw_sessions,
                                       15, edge_theta_max)
    if len(pairs) <= 5:
        pairs = []

    updated_pairs = pairs

    if tuning:
        new_amplitude = 0
    else:
        new_amplitude = estimated_peak

    # Flag for very noisy signals which lead to many succeeding false detections
    hyper_detection_flag = False
    # Flag if very long periods without sessions and very short intervals between adjacent sessions for same id
    unlikely_charging_break_flag = False
    # Counter for intervals shorter than a day
    short_interval_count = 0
    long_interval_found = False
    # Count sessions shorter equal 1 hour
    short_duration_counter = 0
    short_duration_flag = False

    if post_process:
        # check pairs - if 5 or more timestamps with exactly 1 day between adjacent extracted sessions discard of
        # this id
        if len(pairs) > 1:
            start_timestamps = sorted([start for start, stop in updated_pairs])
            # Track adjacent intervals of exactly 1 day
            adjacent_sequence_count = 0
            for i in range(1, len(start_timestamps)):
                interval = start_timestamps[i] - start_timestamps[i - 1]
                if interval == timedelta(days=1):
                    adjacent_sequence_count += 1
                    if adjacent_sequence_count == 5:
                        hyper_detection_flag = True
                else:
                    adjacent_sequence_count = 0

                # Find very long intervals between sessions
                if interval >= timedelta(days=60):
                    long_interval_found = True

                # Count very short intervals
                if interval < timedelta(days=1):
                    short_interval_count += 1

            # If more than 50 % of the intervals are shorter than a day + there is a very long interval:
            if long_interval_found and short_interval_count >= 0.5 * len(pairs):
                print(f"{idecomp.id} unlikely distribution of intervals between sessions - charge timeseries set "
                      f"to zero")
                unlikely_charging_break_flag = True

            for start, stop in pairs:
                if stop-start <= pd.Timedelta(hours=1):
                    short_duration_counter += 1

            if short_duration_counter/len(pairs) > 0.5:
                short_duration_flag = True

    # If there are pairs, i.e. charging sessions
    if len(pairs) > 0 and tuning:
        new_amplitude = idecomp.tune_amplitude()
        idecomp.detected_peak = new_amplitude
        updated_pairs = idecomp.edge_tuning()

    if len(pairs) and tune_magnitude:
        new_amplitude = idecomp.tune_amplitude()
        idecomp.detected_peak = new_amplitude

    if post_process:
        if hyper_detection_flag or unlikely_charging_break_flag:
            if estimated_peak <= 5.5:
                print(f"{idecomp.id} Sequence of 5 sessions with 1 day interval - setting id charging timeseries "
                      f"to zero")
                new_amplitude = 0

        if short_duration_flag:
            print(f"{idecomp.id} unlikely amounts of very short sessions (more than 50% of the extracted sessions "
                  f"take less than 1 hour) - charge timeseries set to zero")
            new_amplitude = 0

    # Create extracted charging power timeseries:
    date_range = pd.date_range(start=idecomp.start_timestamp, end=idecomp.end_timestamp, freq='15min')
    timeseries = pd.Series(0.0, index=date_range)
    # Set the value x for all timestamps from start to stop (including start and stop)
    for start, stop in updated_pairs:
        timeseries[start:stop] = new_amplitude

    idecomp.extracted_session_timeseries_kW = timeseries

    return idecomp, timeseries


"""
Plotting
"""


def set_plot_params(num_figures=1, base_font_size=14, text_width=6.5, wide_plot=False):
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


def plot_extraction(id, start_plot=None, end_plot=None, kind='step', source_path=None, save_name=None, store_all=False,
                    ftype: str = 'png', post_processing=True):
    """
    Plots the extracted charging sessions vs ground truth. If start_plot and/or end_plot are equal to None, all sessions
    are plotted. Automatically stores plots in source_path results directory.
    :param start_plot: Start of the charging session, default: first timestamp
    :param end_plot: End of the charging session, default: last timestamp
    :param id: id of the meter, or directly processed idecomp object
    :param kind: "normal" or "step" --> determines the kind of plot we want to generate. "normal" interpolates linearly
                 between two x-ticks, "step" remains at the value and jumps to the next value at the end of the current
                 x-tick
    :param source_path: Path to the directory containing the .csv files, in parent directory, plotting result directory
                        is created and contains the plots.
    :param save_name: Only used as name for the extraction of the entire date range
    :param store_all: If true, plots for all pairs are generated and stored at fig_path
    :param ftype: file type for the figure, e.g., 'png', 'pgf', 'jpeg', ...
    :param post_processing: Boolean parameter to determine, whether post processing is turned on or off for plotting
    :return:
    """
    if source_path is None:
        source_path = "//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/unlabelled_smd/sorted_by_id/"

    while True:
        if not os.path.exists(source_path):
            source_path = ask_directory("Select directory with the single id .csv files")
        else:
            break

    fpath = source_path + f"{id}.csv"

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File for id {id} could not be found here:\n{source_path}\n.")

    # Create result directory structure
    fig_path = get_extraction_fig_plots_path(source_path)

    # Extract sessions
    idecomp, extracted_timeseries = extract_sessions_for_file(fpath, post_process=post_processing)

    # Plot the entire series
    if start_plot is None and end_plot is None:
        plot_extraction(id,
                        start_plot=idecomp.df_time_series_kw.index.min(),
                        end_plot=idecomp.df_time_series_kw.index.max(),
                        save_name=f'_all.{ftype}', post_processing=post_processing)
        return

    else:
        if start_plot is None:
            start_plot = idecomp.df_time_series_kw.index.min()
        if end_plot is None:
            end_plot = idecomp.df_time_series_kw.index.max()

    if store_all:
        i = 0
        for start, stop in idecomp.pairs:
            start_plot = start - pd.Timedelta(hours=4)
            end_plot = stop + pd.Timedelta(hours=4)

            # Aggregated data
            df_agg = idecomp.df_time_series_kw.loc[start_plot:end_plot]
            # Extracted charging timeseries:
            extracted_charging = extracted_timeseries.loc[start_plot:end_plot]

            fig, ax = plt.subplots()

            if kind == 'normal':
                ax.plot(df_agg, label="Aggregated load")
                ax.plot(extracted_charging, label="Extracted charging\ntimeseries", lw=1, ls='dashed')
            else:
                ax.step(df_agg.index, df_agg, label="Aggregated\nload", where='post')
                ax.step(extracted_charging.index, extracted_charging, label='Extracted\ncharging\ntimeseries',
                        where='post', lw=0.75, ls='dashed')

            ax.set_xlabel('Time')
            # Dynamically adjust x-ticks for long timeseries
            num_hours = (end_plot - start_plot).total_seconds() // 3600  # Compute total hours

            if num_hours <= 10:
                interval = 2  # Tick every 1 hour for short ranges
            elif num_hours <= 20:
                interval = 4  # Tick every 2 hours for one-day plots
            elif num_hours <= 72:
                interval = 12  # Tick every 6 hours for multi-day plots
            else:
                interval = 24  # Tick every 12 hours for long-range plots

            ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Minor ticks at hour intervals

            ax.set_xlim(start_plot, end_plot)

            ax.set_ylabel('15-min\navg power\n(kW)')
            ax.grid()

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)

            fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(fig_path + f"/{idecomp.id}_{i}_{start.strftime('%Y-%m-%d_%H-%M-%S')}.{ftype}")
            fig.show()
            i += 1
            # To prevent HTTP 429 - Too many requests
            time.sleep(1.5)

    else:
        # Aggregated data
        df_agg = idecomp.df_time_series_kw.loc[start_plot:end_plot]
        # Extracted charging timeseries:
        extracted_charging = extracted_timeseries.loc[start_plot:end_plot]

        fig, ax = plt.subplots()

        if kind == 'normal':
            ax.plot(df_agg, label="Aggregated load")
            ax.plot(extracted_charging, label="Extracted charging\ntimeseries", lw=1, ls='dashed')
        else:
            ax.step(df_agg.index, df_agg, label="Aggregated\nload", where='post')
            ax.step(extracted_charging.index, extracted_charging, label='Extracted\ncharging\ntimeseries',
                    where='post', lw=0.75, ls='dashed')

        ax.set_xlabel('Date')

        # Dynamically adjust x-ticks for long timeseries
        num_days = (end_plot - start_plot).days
        if num_days > 180:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Tick every 2 months
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        elif num_days >= 1:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.set_xlabel('Time')

        if num_days == 1 and (end_plot-start_plot).seconds/3600 >=5:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.set_xlabel('Time')

        #if not num_days <= 1:
        #    ax.xaxis.set_tick_params(rotation=15)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

        ax.set_xlim(start_plot, end_plot)

        ax.set_ylabel('15-min\navg power\n(kW)')
        ax.grid()

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)

        fig = plt.gcf()
        fig.tight_layout()
        fig.show()

        if save_name:
            fig.savefig(fig_path + f"/{save_name}")
        else:
            date_string = str(start_plot).replace(':', '_')
            fig.savefig(fig_path + f"/{idecomp.id}_{date_string}.{ftype}")


def plot_coincidence(agg_charging_timeseries_kW, agg_power, result_directory: str, ftype='png'):
    """
    Plot histogram of coincident power normalized by total charging capacity (sum of all estimated power ratings)
    :param agg_charging_timeseries_kW: Time series summing up all the identified charging sessions
    :param agg_power: Sum of all estimated power ratings in kW
    :param result_directory: Directory to save the figure
    :param ftype: file type for the figure, e.g., 'png', 'pgf', 'jpeg', ...
    :return:
    """
    coincidence_time_series = (agg_charging_timeseries_kW/agg_power).fillna(0)

    fig, ax = plt.subplots()

    # Plot histogram
    ax.hist(coincidence_time_series.values, bins=96)
    ax.set_xlabel(f"Coincidence factor\n Total power: {round(agg_power*1e-3,2)} MW")
    ax.set_ylabel("Number of occurrences")

    ax.grid()

    fig.tight_layout()
    fig.show()
    fig.savefig(result_directory + f"/concidence_histo.{ftype}")


def plot_coincidence_boxplot_per_power(agg_timeseries_kw_per_power, agg_power_by_powerrating, total_agg_kW,
                                       results_direc, ftype='png'):
    """
    Plots boxplots of coincidence factors for all charging levels. Also includes a boxplot for total coincidence.
    A bar plot of total power for each discrete value is included.

    Parameters:
    - agg_timeseries_kw_per_power (dict): Key = charging power (closest discrete value) in kW,
                                          Value = summed timeseries of extracted charging power of all IDs with
                                          this charging power.
    - agg_power_by_powerrating (dict): Key = charging power level (kW),
                                       Value = total aggregated power of all IDs at that power level.
    - total_agg_kW (pd.Series): Sum of all time series of extracted charging.
    - results_direc (str): Directory in which the plot is saved.
    - ftype (str): file type for the figure, e.g., 'png', 'pgf', 'jpeg', ...
    """

    # Compute total power for coincidence factor calculation
    total_power = sum(agg_power_by_powerrating.values())

    # Compute total coincidence series
    total_coincidence = total_agg_kW / total_power
    total_coincidence = total_coincidence.dropna()  # Remove NaN values

    # Define valid charging levels (excluding 0)
    charging_levels = sorted(agg_timeseries_kw_per_power.keys())
    valid_levels = [level for level in charging_levels if level != 0 and agg_power_by_powerrating[level] >= 10 * level]

    # Compute coincidence factors per power level
    coincidence_data = {
        level: (agg_timeseries_kw_per_power[level] / agg_power_by_powerrating[level]).dropna()
        for level in valid_levels
    }

    # Compute number of charging points per charging level
    charging_points = [int(agg_power_by_powerrating[level] // level) for level in valid_levels]

    # Initialize lists for labels and data
    valid_data = []
    custom_labels = []

    # Add 'All' only if it contains valid data
    if not total_coincidence.empty:
        valid_data.append(total_coincidence)  # Insert 'All' data at the start
        custom_labels.append("All")  # Insert 'All' label at the start

    # Add valid power levels
    valid_data.extend([coincidence_data[level] for level in valid_levels])
    custom_labels.extend([str(level) for level in valid_levels])

    # Convert to arrays for plotting
    x_positions = np.arange(len(custom_labels))

    # Create the figure
    fig, ax1 = plt.subplots()

    # Boxplot styling
    boxprops = dict(marker='.', markersize=1)  # Reduce outlier marker size
    capprops = dict(linewidth=1)  # Thicker whisker caps

    # Ensure valid_data and custom_labels match
    assert len(valid_data) == len(custom_labels), "Mismatch between labels and data!"

    # Plot boxplot
    ax1.boxplot(
        valid_data,
        positions=x_positions,
        vert=True,
        patch_artist=True,
        labels=custom_labels,
        flierprops=boxprops,  # Apply small marker size to outliers
        capprops=capprops  # Thicker whisker caps
    )

    ax1.set_ylabel("Coincidence Factor", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(custom_labels)
    ax1.set_xlabel("Charging Power (kW)")
    ax1.grid()

    # Plot charging points as a bar plot (right y-axis) for all levels except 'All'
    ax2 = ax1.twinx()
    ax2.bar(x_positions[1:], charging_points, alpha=0.6, color="gray", width=0.7, label="Number of Charging Points")
    ax2.set_ylabel("Number of Charging Points", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    fig.tight_layout()
    fig.show()
    fig.savefig(results_direc + f"/coincidence_boxplot_per_power.{ftype}", dpi=500)
    fig.savefig(results_direc + "/coincidence_boxplot_per_power.pgf", dpi=500)


def plot_coincidence_per_month(agg_charging_timeseries, agg_power, result_direc, file_type):
    """
    Plots aggregated charging time series
    :param agg_charging_timeseries: directory with key: id, value: charging time series
    :param agg_power: aggregated power peak
    :param result_direc: directory, to which results are written
    :param file_type: file type for result plot
    :return:
    """
    normalized_power = agg_charging_timeseries / agg_power

    # Compute absolute power (monthly averages) for labeling purposes
    monthly_avg_abs = agg_charging_timeseries.resample('M').mean()

    # Extract month from index and group data
    normalized_power_df = pd.DataFrame({'Month': normalized_power.index.month, 'Value': normalized_power})

    # Create figure and primary axis (Left Y-Axis: Normalized Power)
    fig, ax1 = plt.subplots()

    flierprops = dict(marker='.', markersize=1, linestyle='none', alpha=0.6)
    # Boxplot
    normalized_power_df.boxplot(column='Value', by='Month', ax=ax1, showfliers=True, patch_artist=True,
                                flierprops=flierprops)

    ax1.set_ylabel('Normalized Power', color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlabel('')

    # Right Y-Axis (Only Labels, No Plot)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Absolute Power (kW)', color="gray")
    ax2.tick_params(axis='y', labelcolor="gray")

    # Scale the right Y-axis to match absolute power values
    ax2.set_ylim(ax1.get_ylim()[0] * agg_power, ax1.get_ylim()[1] * agg_power)

    # Formatting
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                        rotation=55)

    ax1.set_title('')
    fig.suptitle("")
    fig.tight_layout()
    fig.show()
    # Save figure
    fig.savefig(f"{result_direc}/coinciding_power_per_month.{file_type}", dpi=500)


def plot_coincidence_per_15min_IQR(agg_charging_timeseries, agg_power, result_direc, file_type):
    """
    Plots the distribution of charging power over a 24-hour period, separated by:
    1. Weekdays (Mon-Fri)
    2. Weekends (Sat-Sun)
    3. Separate days (Mon-Sun)

    Maintains 15-minute resolution.

    :param agg_charging_timeseries: Pandas Series with 15-min interval time series data
    :param agg_power: Aggregated peak power
    :param result_direc: Directory where results are saved
    :param file_type: File type for the saved plot
    :return: None
    """
    # Compute normalized power
    normalized_power = agg_charging_timeseries / agg_power

    # Create DataFrame with time-related features
    df = pd.DataFrame({
        'Hour': agg_charging_timeseries.index.hour + agg_charging_timeseries.index.minute / 60.0,
        # Keep 15-min resolution
        'DayOfWeek': agg_charging_timeseries.index.dayofweek,  # 0=Mon, 6=Sun
        'NormalizedPower': normalized_power,
        'AbsolutePower': agg_charging_timeseries
    })

    # ----------- Plot 1: Weekday vs. Weekend (15-min resolution) -----------
    fig, ax1 = plt.subplots()

    # Weekday (Mon-Fri) & Weekend (Sat-Sun) Data
    weekday_avg = df[df['DayOfWeek'] < 5].groupby('Hour').mean()
    weekend_avg = df[df['DayOfWeek'] >= 5].groupby('Hour').mean()

    ax1.plot(weekday_avg.index, weekday_avg['NormalizedPower'], label="Weekday", color='blue', linestyle='-')
    ax1.plot(weekend_avg.index, weekend_avg['NormalizedPower'], label="Weekend", color='red', linestyle='--')

    ax1.set_ylabel('Normalized Power', color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.set_xlabel('Hour of Day')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Right Y-Axis: Absolute Power (No Plot, Only Labels)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Absolute Power (kW)', color="gray")
    ax2.tick_params(axis='y', labelcolor="gray")
    ax2.set_ylim(ax1.get_ylim()[0] * agg_power, ax1.get_ylim()[1] * agg_power)

    ax1.legend()
    fig.tight_layout()
    fig.savefig(f"{result_direc}/coinciding_power_weekday_vs_weekend.{file_type}", dpi=500)
    fig.show()

    # ----------- Plot 2: Daily Trends (Mon-Sun, 15-min resolution) -----------
    fig, ax1 = plt.subplots()

    for day in range(7):  # Monday–Sunday
        daily_avg = df[df['DayOfWeek'] == day].groupby('Hour').mean()
        ax1.plot(daily_avg.index, daily_avg['NormalizedPower'],
                 label=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day])

    ax1.set_ylabel('Normalized Power', color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.set_xlabel('Hour of Day')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Right Y-Axis: Absolute Power (No Plot, Only Labels)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Abs. Power (kW)', color="gray")
    ax2.tick_params(axis='y', labelcolor="gray")
    ax2.set_ylim(ax1.get_ylim()[0] * agg_power, ax1.get_ylim()[1] * agg_power)

    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(f"{result_direc}/coinciding_power_per_day.{file_type}", dpi=500)
    fig.show()

    normalized_power = agg_charging_timeseries / agg_power

    # Create DataFrame with time-related features
    df = pd.DataFrame({
        'Hour': agg_charging_timeseries.index.hour + agg_charging_timeseries.index.minute / 60.0,
        # Preserve 15-min resolution
        'DayOfWeek': agg_charging_timeseries.index.dayofweek,  # 0=Mon, 6=Sun
        'NormalizedPower': normalized_power,
        'AbsolutePower': agg_charging_timeseries
    })

    # Separate weekday (Mon-Fri) and weekend (Sat-Sun) data
    weekday_df = df[df['DayOfWeek'] < 5]
    weekend_df = df[df['DayOfWeek'] >= 5]

    def compute_stats(data):
        """ Compute median, IQR, and mean for each 15-min bin """
        grouped = data.groupby('Hour')
        median = grouped['NormalizedPower'].median()
        q25 = grouped['NormalizedPower'].quantile(0.25)
        q75 = grouped['NormalizedPower'].quantile(0.75)
        mean = grouped['NormalizedPower'].mean()
        return median, q25, q75, mean

    # Compute stats for weekday & weekend
    weekday_median, weekday_q25, weekday_q75, weekday_mean = compute_stats(weekday_df)
    weekend_median, weekend_q25, weekend_q75, weekend_mean = compute_stats(weekend_df)

    # ----------- Plot 1: Weekday Variation -----------
    fig, ax1 = plt.subplots()

    # Shaded region for IQR
    ax1.fill_between(weekday_median.index, weekday_q25, weekday_q75, color='blue', alpha=0.2,
                     label="IQR")

    # Median line
    ax1.plot(weekday_median.index, weekday_median, color='blue', linestyle='-', label="Median")

    # Mean line
    ax1.plot(weekday_median.index, weekday_mean, color='blue', linestyle='dashed', label="Mean")

    ax1.set_ylabel('Normalized Power', color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.set_xlabel('Hour of Day')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc="best")

    # Right Y-Axis: Absolute Power (No Plot, Only Labels)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Absolute Power (kW)', color="gray")
    ax2.tick_params(axis='y', labelcolor="gray")
    ax2.set_ylim(ax1.get_ylim()[0] * agg_power, ax1.get_ylim()[1] * agg_power)

    fig.tight_layout()
    fig.savefig(f"{result_direc}/coincidence_weekday_variation.{file_type}", dpi=500)
    fig.show()

    # ----------- Plot 2: Weekend Variation -----------
    fig, ax1 = plt.subplots()

    # Shaded region for IQR
    ax1.fill_between(weekend_median.index, weekend_q25, weekend_q75, color='red', alpha=0.2,
                     label="IQR")

    # Median line
    ax1.plot(weekend_median.index, weekend_median, color='red', linestyle='-', label="Median")

    # Mean line
    ax1.plot(weekend_median.index, weekend_mean, color='red', linestyle='dashed', label="Mean")

    ax1.set_ylabel('Normalized Power', color="red")
    ax1.tick_params(axis='y', labelcolor="red")
    ax1.set_xlabel('Hour of Day')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc="best")

    # Right Y-Axis: Absolute Power (No Plot, Only Labels)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Absolute Power (kW)', color="gray")
    ax2.tick_params(axis='y', labelcolor="gray")
    ax2.set_ylim(ax1.get_ylim()[0] * agg_power, ax1.get_ylim()[1] * agg_power)

    fig.tight_layout()
    fig.savefig(f"{result_direc}/coincidence_weekend_variation.{file_type}", dpi=500)
    fig.show()


def plot_worst_case_coincidence(load_data_dict, result_direc, file_type, num_simulations=1000):
    """
    Simulates worst-case coincident power using Monte Carlo sampling. Plots worst-case coincidence factor distribution
    for different number of charging points.

    - Selects random subsets of time series (5, 10, 20, 30, 50, 100).
    - Computes worst-case coincident power by summing them up.
    - Repeats 10,000 times per subset size.
    - Normalizes using the sum of max values of all time series.
    - Plots a boxplot showing the distribution of worst-case coincident power.

    :param load_data_dict: Dictionary where key = id, value = Pandas time series of load data.
    :param result_direc: Directory to save the resulting boxplot.
    :param file_type: File format for saving the plot (e.g., 'png', 'pdf').
    :param num_simulations: Number of worst case combinations (i.e., per box 1000 combinations of the subset size are
                            calculated)
    :return: None
    """

    # Extract time series from dictionary
    all_timeseries = list(load_data_dict.values())

    # Define subset sizes
    subset_sizes = [5, 10, 20, 30, 50, 100]

    # Store worst-case results
    worst_case_results = {n: [] for n in subset_sizes}

    for n in subset_sizes:
        for _ in tqdm(range(num_simulations), desc=f'Simulation {n} charging points 11 kW'):
            # Randomly select `n` time series
            sampled_ts = random.sample(all_timeseries, n)

            # Compute normalization factor: sum of max values of the selected time series
            normalization_factor = sum([ts.max() for ts in sampled_ts])

            # Compute the worst-case coincident power (sum of max values at each time step)
            worst_case_power = max(sum(ts) for ts in zip(*sampled_ts))

            # Normalize the result using only the selected time series
            normalized_worst_case = worst_case_power / normalization_factor

            # Store result
            worst_case_results[n].append(normalized_worst_case)

    # Convert results to DataFrame for plotting
    df_results = pd.DataFrame(dict([(f"{n}", worst_case_results[n]) for n in subset_sizes]))

    # Plot boxplot
    fig, ax = plt.subplots()

    df_results.boxplot(ax=ax, showfliers=True, patch_artist=True,
                       flierprops=dict(marker='o', markersize=2, linestyle='none', alpha=0.6))  # Small outliers

    ax.set_xlabel("Number of CP")
    ax.set_ylabel("Worst case normalized power")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save & Show Plot
    fig.tight_layout()
    fig.savefig(f"{result_direc}/worst_case_coincidence.{file_type}", dpi=500)
    fig.show()
    return df_results


def plot_power_distribution(power_collection_kW: dict, result_directory: str, ftype: str = 'png'):
    """
    Plots distribution of all charging power values != 0. Lower power might be a bit misleading, since such users
    are charging longer for the same energy requirement and thus have a higher occurrence. This is somewhat compensated
    by the fact that these users are more likely to have a plug-in hybrid or a car with a small battery size.

    On the second axis, the normalized distribution is displayed.

    The plot is stored in the results folder
    :param power_collection_kW: Generated in main() --> key, value - key: charging power, value: number of occurrences
    :param result_directory: Directory, where the plot is saved
    :param ftype: file type for the figure, e.g., 'png', 'pgf', 'jpeg', ...
    :return:
    """
    keys = np.array(list(power_collection_kW.keys()))
    values = np.array(list(power_collection_kW.values()))

    # Define custom x-tick labels
    custom_labels = [3.7, 5.5, 7.4, 11, 17.6, 22]

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot histogram
    ax.bar(keys, values)
    ax.set_xlabel("Charging Power (kW)")
    ax.set_ylabel("Number of occurrences")
    ax.tick_params(axis="y")
    ax.set_xlim(2.5, 23.5)
    ax.set_xticks(custom_labels)
    ax.set_xticklabels(custom_labels)
    ax.grid()

    fig.tight_layout()
    fig.show()
    fig.savefig(result_directory + f"/charging_power.{ftype}")


def plot_energy_per_session_distribution(energy_collection_kWh: dict, result_directory: str, ftype: str = 'png'):
    """
    Plots distribution of charged energy per session in a histogram. The power rating specific energy per session
    distributions are also plotted separately, if more than 10 sessions are present in the energy_collection_kWh
    dictionary.

    :param energy_collection_kWh: Energy per charging session in kWh, differentiated by power - key: power, value:
                                  charged energy in kWh.
    :param result_directory: Directory to which the plots are saved.
    :param ftype: File type for the figure, e.g., 'png', 'pgf', 'jpeg', ...
    :return: None
    """
    e_col = copy.deepcopy(energy_collection_kWh)

    # Flatten all energy values into a single list for combined histogram
    e_combined_collection = [e for energies in e_col.values() for e in energies]

    # Define bin edges (starting from 0, increasing by 5 kWh)
    max_energy = max(e_combined_collection) if e_combined_collection else 50
    bins = np.arange(0, max_energy + 5, 5)

    # Create figure and axis for combined distribution
    fig, ax = plt.subplots()
    ax.hist(e_combined_collection, bins=bins, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Energy consumption per\ncharging session (kWh)")
    ax.set_ylabel("Number of occurrences")
    ax.grid()
    ax.set_axisbelow(True) # Set lines of grid in background

    fig.tight_layout()
    fig.show()
    fig.savefig(f"{result_directory}/charging_energy_all.{ftype}")

    # Plot individual histograms for power levels with 10+ sessions
    for power, energy in e_col.items():
        if len(energy) < 10:
            continue

        fig, ax = plt.subplots()
        ax.hist(energy, bins=bins, edgecolor='black', linewidth=0.5)
        ax.set_xlabel("Energy consumption per\ncharging session (kWh)")
        ax.set_ylabel("Number of occurrences")
        ax.grid()
        ax.set_axisbelow(True)

        fig.tight_layout()
        fig.show()
        fig.savefig(f"{result_directory}/charging_energy_{power}kW.{ftype}")


def plot_energy_per_session_box_plot(energy_per_session_collection_kWh, result_direc, file_type):
    """
    Creates a box plot for charged energy per session, ordered by charging speed, with an additional bar plot
    for the number of occurrences for specific power levels.

    :param energy_per_session_collection_kWh: Energy per charging session in kWh, differentiated by power - key: power,
                                              value: charged energy in kWh.
    :param result_direc: Directory to which the plots are saved.
    :param file_type: File type for the figure, e.g., 'png', 'pgf', 'jpeg', ...
    :return: None
    """
    e_col = {}
    for power, energy in energy_per_session_collection_kWh.items():
        if power == 11.09:
            e_col[11] = energy
        elif power == 22.17:
            e_col[22] = energy
        else:
            e_col[power] = energy

    # Flatten all energy values for the combined boxplot
    e_combined_collection = [e for energies in e_col.values() for e in energies]

    # Define valid power levels (Only include if they have at least 10 sessions)
    valid_levels = [power for power, energy in e_col.items() if len(energy) >= 10]

    # Prepare data for boxplot
    valid_data = []
    custom_labels = []

    # Add 'All' (combined) if it contains valid data
    if e_combined_collection:
        valid_data.append(e_combined_collection)
        custom_labels.append("All")

    # Add valid power levels
    for power in sorted(valid_levels):
        valid_data.append(e_col[power])
        custom_labels.append(f"{power}")

    # Define power levels for the bar plot
    bar_plot_levels = {3.7, 5.5, 7.4, 11, 17.6, 22}
    bar_values = [len(e_col[power]) if power in e_col else 0 for power in bar_plot_levels]

    # Convert to arrays for plotting
    x_positions = range(len(custom_labels))

    # Create the figure and primary axis (boxplot)
    fig, ax1 = plt.subplots()

    # Boxplot styling
    boxprops = dict(marker='.', markersize=3)  # Small markers for outliers
    capprops = dict(linewidth=1)  # Thicker whisker caps

    # Plot boxplot
    ax1.boxplot(
        valid_data,
        positions=x_positions,
        vert=True,
        patch_artist=True,
        labels=custom_labels,
        flierprops=boxprops,  # Small outlier markers
        capprops=capprops  # Thicker whisker caps
    )

    ax1.set_ylabel("Energy per session (kWh)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(custom_labels)
    ax1.set_xlabel("Charging Power Level (kW)")
    ax1.grid()
    ax1.set_axisbelow(True)  # Ensure grid lines are in the background

    # Secondary axis for occurrences bar plot
    ax2 = ax1.twinx()
    ax2.bar(
        [custom_labels.index(str(power)) for power in bar_plot_levels if str(power) in custom_labels],
        [bar_values[i] for i, power in enumerate(bar_plot_levels) if str(power) in custom_labels],
        alpha=0.6,
        color="gray",
        width=0.7,
        label="Number of Occurrences"
    )

    ax2.set_ylabel("Number of sessions", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    fig.tight_layout()
    fig.show()
    fig.savefig(f"{result_direc}/charging_energy_boxplot.{file_type}", dpi=500)
    fig.savefig(f"{result_direc}/charging_energy_boxplot.pgf", dpi=500)


def plot_days_between_sessions_distribution(d_btw_sessions: dict, result_directory: str, ftype: str = 'png'):
    """
    Plots the distribution of days between consecutive sessions
    :param d_btw_sessions: dictionary with number of days between sessions by charging power
    :param result_directory: directory to which the figures are saved to
    :param ftype: file type for the figure, e.g., 'png', 'pgf', 'jpeg', ...
    :return:
    """
    d_col = copy.copy(d_btw_sessions)

    # Plot the combined distribution of energy per session:
    d_combined_collection = []
    for power, days in d_col.items():
        d_combined_collection.extend(days)

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot histogram
    ax.hist(d_combined_collection, bins=96)
    ax.set_xlabel("Days between consecutive\ncharging sessions")
    ax.set_xlim(0, 50)
    ax.set_ylabel("Number of occurrences")
    ax.grid()

    fig.tight_layout()
    fig.show()
    fig.savefig(result_directory + f"/d_btw_sessions_all.{ftype}")

    # Get rid of data, if less than 10 sessions for the charging power have been detected
    for power, days in d_col.items():
        if len(days) < 10:
            continue
        else:
            fig, ax = plt.subplots()

            # Plot histogram
            ax.hist(days, bins=96)
            ax.set_xlabel(f"Days between consecutive\ncharging sessions {power} kW")
            ax.set_xlim(0, 50)
            ax.set_ylabel("Number of occurrences")
            ax.grid()

            fig.tight_layout()
            fig.show()
            fig.savefig(result_directory + f"/d_btw_sessions_{power}kW.{ftype}")


def plot_starting_time_distribution(time_collection: dict, result_directory: str, ftype: str = "png",
                                    fname=''):
    """
    Plots the distribution of charging start time in a histogram
    :param time_collection: contains keys: 'time' as datetime.time instances and value: 'number of occurrences'
    :param result_directory: Directory to store the plots in
    :param ftype: file type for the figure, e.g., 'png', 'pgf', 'jpeg', ...
    :param fname: File name suffix, used to plot summer/winter time separately
    :return:
    """
    # Convert time to hours for better visualization
    time_hours = np.array([t.hour + t.minute / 60 for t in time_collection.keys()])
    occurrences = np.array(list(time_collection.values()))

    # Create figure and axis
    fig, ax = plt.subplots()

    # Create histogram with 15-minute bins (0.25 hour intervals)
    bins = np.arange(0, 24.25, 0.25)  # 0 to 24 in 15-minute increments
    ax.hist(time_hours, bins=bins, weights=occurrences)

    ax.set_xlabel("Charging Start Time (Hours)")
    ax.set_ylabel("Number of Occurrences")

    # Adjust x-ticks to display every 3 hours for readability
    ax.set_xticks(np.arange(0, 25, 3))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.set_xlim(0, 24)  # Ensure full 24-hour coverage
    ax.grid()

    # Show the plot
    fig.tight_layout()
    fig.show()

    if fname == '':
        fig.savefig(result_directory + f"/starting_time_distribution.{ftype}")
    else:
        fig.savefig(result_directory + f"/starting_time_distribution_{fname}.{ftype}")


def plot_annual_energy_consumption(annual_e: dict, result_direc: str, ftype: str = 'png'):
    """
    Plots annual energy consumption distribution per charging point. Also plots the histograms for individual charging
    powers, if enough data is present.

    :param annual_e: Dictionary with keys: Charging power (kW), values: List of annual energy consumption in Wh.
    :param result_direc: Directory to which the plots are saved.
    :param ftype: File type for the figure, e.g., 'png', 'pgf', 'jpeg', ...
    :return: None
    """
    e_col = {}

    for power, val in annual_e.items():
        if power == 11.09 or power == 22.17:
            e_col[int(power)] = val
        else:
            e_col[power] = val

    # Flatten all energy values for the combined histogram
    e_combined_collection = [e * 1e-6 for energies in e_col.values() for e in energies]  # Convert Wh to MWh

    # Determine the global max energy consumption (to keep x-axis consistent)
    max_energy_global = max(e_combined_collection) if e_combined_collection else 10  # Default max for safety
    bins = np.arange(0, max_energy_global + 0.5, 0.5)  # Bins of size 0.5 MWh

    # Create figure for combined distribution
    fig, ax = plt.subplots()

    # Plot histogram
    ax.hist(e_combined_collection, bins=bins, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Annual energy consumption\nper charging point (MWh)")
    ax.set_ylabel("Number of occurrences")
    ax.grid()
    ax.set_axisbelow(True)  # Grid in background
    ax.set_xlim(0, max_energy_global)  # Keep x-axis consistent

    fig.tight_layout()
    fig.show()
    fig.savefig(f"{result_direc}/e_annual_all.{ftype}")

    # Plot individual histograms for power levels with 10+ sessions
    for power, energy in e_col.items():
        if len(energy) < 10:
            continue

        energy_mwh = [e * 1e-6 for e in energy]  # Convert Wh to MWh

        fig, ax = plt.subplots()

        # Plot histogram
        ax.hist(energy_mwh, bins=bins, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(f"Annual energy consumption per\n{power} kW charging point (MWh)")
        ax.set_ylabel("Number of occurrences")
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_xlim(0, max_energy_global)  # Keep x-axis consistent

        fig.tight_layout()
        fig.show()
        fig.savefig(f"{result_direc}/e_annual_{power}kW.{ftype}")


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
                                         num_workers=None):
    """
    Fully parallelized Monte Carlo simulation of worst-case coincident power.
    - Utilizes all CPU cores to run simulations efficiently.
    - Divides work into balanced chunks for optimal CPU utilization.

    :param load_data_dict: Dictionary where key = id, value = Pandas time series of load data.
    :param result_direc: Directory to save the resulting boxplot.
    :param file_type: File format for saving the plot (e.g., 'png', 'pdf').
    :param num_simulations: Total number of Monte Carlo simulations per subset size.
    :param num_workers: Number of CPU cores to use (default: all available).
    :return: DataFrame of worst-case results.
    """

    # Use all available cores if num_workers is not specified
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

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

    # Plot boxplot
    fig, ax = plt.subplots()
    df_results.boxplot(ax=ax, showfliers=True, patch_artist=True,
                       flierprops=dict(marker='o', markersize=2, linestyle='none', alpha=0.6))  # Small outliers

    ax.set_xlabel("Number of CP")
    ax.set_ylabel("Worst case normalized power")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save & Show Plot
    fig.tight_layout()
    fig.savefig(f"{result_direc}/worst_case_coincidence_parallel.{file_type}", dpi=500)
    fig.show()

    return df_results  # Return DataFrame for further analysis


def convert_to_winter_time(agg_charging_timeseries_kW, summer_start, summer_end):
    """
    Converts time series to winter time - adds an hour to all data between summer_start and summer_end, adds NaN for
    overlapping 2 timestamps
    :param agg_charging_timeseries_kW: pd.Series with aggregated charging load
    :param summer_start: datetime object, denoting start datetime in investigated data
    :param summer_end: datetime object, denoting end datetime in investigated data
    :return:
    """
    # Select summer time data
    summer_data = agg_charging_timeseries_kW.loc[summer_start:summer_end].copy()

    # Shift timestamps by +1 hour
    summer_data.index = summer_data.index + pd.Timedelta(hours=1)

    # Merge summer-shifted data with original dataset
    adjusted_series = pd.concat([
        agg_charging_timeseries_kW.loc[:summer_start - pd.Timedelta(minutes=1)],  # Before summer time
        summer_data,  # Shifted summer data
        agg_charging_timeseries_kW.loc[summer_end + pd.Timedelta(minutes=1):]  # After summer time
    ])

    # Identify and handle duplicate timestamps (caused by fall-back time shift)
    duplicated_mask = adjusted_series.index.duplicated(keep=False)

    # Insert NaN at duplicated timestamps
    adjusted_series.loc[duplicated_mask] = None

    # Drop exact duplicates (keep first occurrence)
    adjusted_series = adjusted_series[~adjusted_series.index.duplicated(keep='first')]

    return adjusted_series


def write_11kw_to_csv(comb_11kW_collection, result_direc):
    """
    Writes 11 kW charging time series to a single parquet file to be used later on for optimization purposes
    :param comb_11kW_collection: dictionary with 11 kw charging time series
    :param result_direc: directory, where to store the file
    :return:
    """
    # Ensure the target directory exists
    save_path = os.path.join(result_direc, "11kw_charging_timeseries")
    os.makedirs(save_path, exist_ok=True)

    # Convert dictionary to DataFrame (Each Series becomes a column)
    df = pd.DataFrame(comb_11kW_collection)

    # Save to Parquet file
    parquet_file = os.path.join(save_path, "charging_timeseries.parquet")
    df.to_parquet(parquet_file, engine="pyarrow")


def main():
    #set_plot_params()
    plt.rcParams['figure.dpi'] = 300

    # Initialize some containers for results
    # Power distribution (key=round(power,2), value=number of occurrences) for all 15-min intervals:
    power_collection_kW = {}
    energy_per_session_collection_kWh = {3.7: [], 5.5: [], 7.4: [], 11.09: [], 17.6: [], 22.17: []}
    days_between_sessions_collection_d = {3.7: [], 5.5: [], 7.4: [], 11.09: [], 17.6: [], 22.17: []}
    # Charging start time collection (key=start time of the 15-min interval, value=number of occurrences)
    starting_time_collection_datetime = {}

    # Also store the datetime, depending on winter/summer time for debugging of peaks in charging start time plot
    winter_time_collection = {dt.time(hour, minute): 0 for hour in range(24) for minute in range(0, 60, 15)}
    summer_time_collection = {dt.time(hour, minute): 0 for hour in range(24) for minute in range(0, 60, 15)}
    summer_start_datetime = pd.to_datetime('2024-03-31 00:00:00')
    summer_end_datetime = pd.to_datetime('2024-10-27 00:00:00')
    # Collection with the two dicts added, summer corrected by 1 hour
    combined_collection = {}

    # Summed charging timeseries - all charging summed up - agg stands for aggregated/aggregation
    agg_charging_timeseries_kW = None
    # Summing up the timeseries per charging power
    agg_timeseries_kW_per_power = {3.7: None, 5.5: None, 7.4: None, 11.09: None, 17.6: None, 22.17: None}
    # Summing up extracted power sorted by power rating
    agg_power_by_powerrating = {3.7: 0, 5.5: 0, 7.4: 0, 11.09: 0, 17.6: 0, 22.17: 0}
    # Summing up all power ratings
    agg_power = 0
    annual_energy_consumption_kWh = {3.7: [], 5.5: [], 7.4: [], 11.09: [], 17.6: [], 22.17: []}
    # Dict with id: annual energy consumption (kWh) to find the reason for very large / small values
    id_annual_energy_consumption_kWh = {}

    # Dictionary with key: id, value: charging time series from 11 kW charging
    comb_11kW_collection = {}

    # Time series collection for export -> Don't have to rerun each time for plotting / further analyses
    corrected_timeseries_by_id = {}

    # Ask for directory with all the preprocessed smd
    source_path = "//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/unlabelled_smd/sorted_by_id/"
    result_direc = "//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/unlabelled_smd/results"
    while not os.path.exists(source_path):
        source_path = ask_directory("Select the directory containing all the preprocessed yearly smart meter data.")
    create_result_dir(source_path)

    # Get the path to all .csv files in the directory containing the 15-min energy values
    filenames = [os.path.join(source_path, file) for file in os.listdir(source_path)
                 if not file.endswith("_missing_times.csv")]

    """
    plot_extraction('CH1003601234500000000000005231743',
                    start_plot=pd.to_datetime('2024-01-01 00:00:00'), end_plot=pd.to_datetime('2024-01-10 08:00:00'))
    """

    pickle_output_path = os.path.join(result_direc, "all_identified_charging_ts.pkl")
    new_pickle = True
    # Extract charging timeseries
    for file in tqdm(filenames, desc=f"Processing files:"):
        # Check if analysis has been done before and break loop, if this is the case
        if os.path.exists(pickle_output_path):
            # Count number of ids with no detections
            zero_count = 0

            new_pickle = False
            with open(pickle_output_path, "rb") as f:
                corrected_timeseries_by_id = pickle.load(f)
            print(f"Read identified sessions from pickle file: {pickle_output_path}")

            for key, value in corrected_timeseries_by_id.items():
                # Kicking out outliers - there once was a timeseries with > 100 kW somehow...
                if value.max() > 23:
                    print(f"Skipping {key} -- max ts value: {value.max()}")
                    continue
                if value.max() == 0:
                    zero_count += 1
                agg_power += value.max()
                if agg_charging_timeseries_kW is None:
                    agg_charging_timeseries_kW = value
                else:
                    agg_charging_timeseries_kW += value

            print(f"Number of ids with no detections: {zero_count}")
            print(f"Number of ids with detections: {len(corrected_timeseries_by_id) - zero_count}")
            print(f"Total number of ids: {len(corrected_timeseries_by_id)}")
            break

        tqdm.write(os.path.basename(file).split('.csv')[0])
        # Get extraction
        idecomp, charging_timeseries = extract_sessions_for_file(file)

        corrected_ts = convert_to_winter_time(charging_timeseries, summer_start_datetime, summer_end_datetime)
        # Store corrected (converted to winter time) time series by ID for export to pickle
        corrected_timeseries_by_id[idecomp.id] = corrected_ts

        # plot_extraction('CH1003601234500000000000000038871', store_all=True)

        # Add time series to total charging time series
        if agg_charging_timeseries_kW is None:
            agg_charging_timeseries_kW = charging_timeseries
        else:
            agg_charging_timeseries_kW += charging_timeseries
        agg_power += idecomp.closest_discrete_value

        # Store results in a usable format for further processing:
        power = round(idecomp.tuned_magnitude, 2)
        number_of_charging_values = (charging_timeseries != 0).sum()
        try:
            # Append number of sessions / 4 (=number of hours) * power (kW)
            annual_energy_consumption_kWh[idecomp.closest_discrete_value].append(number_of_charging_values/4*power*1e3)
            id_annual_energy_consumption_kWh[idecomp.id] = number_of_charging_values/4*power*1e3

            # Add extracted time series to charging time series of the discrete power value
            if agg_timeseries_kW_per_power[idecomp.closest_discrete_value] is None:
                agg_timeseries_kW_per_power[idecomp.closest_discrete_value] = charging_timeseries
            else:
                agg_timeseries_kW_per_power[idecomp.closest_discrete_value] += charging_timeseries
            agg_power_by_powerrating[idecomp.closest_discrete_value] += power

        # When closest_discrete_value == 0 -> no annual consumption
        except KeyError:
            pass
        power_collection_kW[power] = number_of_charging_values

        if idecomp.closest_discrete_value == 11.09:
            comb_11kW_collection[idecomp.id] = charging_timeseries

        # Collect session start timestamps to calculate days between sessions and store in starting time collection
        start_collection = []
        for start, stop in idecomp.pairs:
            current_session_energy_kWh = round(power * (stop-start).total_seconds()/3600, 2)
            energy_per_session_collection_kWh[idecomp.closest_discrete_value].append(current_session_energy_kWh)
            start_collection.append(start)
            # Write start times into start time collection dictionary
            try:
                starting_time_collection_datetime[start.time()] += 1
                if summer_start_datetime <= start <= summer_end_datetime:
                    summer_time_collection[start.time()] += 1
                else:
                    winter_time_collection[start.time()] += 1
            except KeyError:
                starting_time_collection_datetime[start.time()] = 1
                if summer_start_datetime <= start <= summer_end_datetime:
                    summer_time_collection[start.time()] = 1
                else:
                    winter_time_collection[start.time()] = 1

            # Write start times into summer/winter time collection dictionaries
            try:
                if summer_start_datetime <= start <= summer_end_datetime:
                    summer_time_collection[start.time()] += 1
                else:
                    winter_time_collection[start.time()] += 1
            except KeyError:
                if summer_start_datetime <= start <= summer_end_datetime:
                    summer_time_collection[start.time()] = 1
                else:
                    winter_time_collection[start.time()] = 1

        # Summer time is 1 hour shifted (too early) and the keys get an hour added to correspond to winter time
        shifted_summer_time_collection = shift_time_dict(summer_time_collection)
        for timestamp, count in shifted_summer_time_collection.items():
            combined_collection[timestamp] = (shifted_summer_time_collection[timestamp] +
                                              winter_time_collection[timestamp])

        d_btw_sessions = [(start_collection[i] - start_collection[i - 1]).days for i in range(1, len(start_collection))]
        try:
            days_between_sessions_collection_d[idecomp.closest_discrete_value].extend(d_btw_sessions)
        except KeyError:
            pass

    # Save corrected timeseries to pickle file
    if new_pickle:
        with open(pickle_output_path, "wb") as f:
            pickle.dump(corrected_timeseries_by_id, f)
        print(f"Corrected time series dumped to: {pickle_output_path}")

    # Set False if summer/winter time is already handled before
    convert_to_winter = True

    # Statistical analysis / Plotting - take .eps for latex plots
    file_type = 'eps'

    # Histogram
    plot_coincidence(agg_charging_timeseries_kW, agg_power, result_direc, file_type)

    plot_coincidence_boxplot_per_power(agg_timeseries_kW_per_power, agg_power_by_powerrating,
                                       agg_charging_timeseries_kW, result_direc, file_type)

    if convert_to_winter:
        agg_converted = convert_to_winter_time(agg_charging_timeseries_kW, summer_start_datetime, summer_end_datetime)
        plot_coincidence_per_15min_IQR(agg_converted, agg_power, result_direc, file_type)
    else:
        plot_coincidence_per_15min_IQR(agg_charging_timeseries_kW, agg_power, result_direc, file_type)

    plot_coincidence_per_month(agg_charging_timeseries_kW, agg_power, result_direc, file_type)

    plot_power_distribution(power_collection_kW, result_direc, file_type)

    plot_energy_per_session_distribution(energy_per_session_collection_kWh, result_direc, file_type)
    plot_energy_per_session_box_plot(energy_per_session_collection_kWh, result_direc, file_type)

    plot_days_between_sessions_distribution(days_between_sessions_collection_d, result_direc, file_type)

    plot_starting_time_distribution(starting_time_collection_datetime, result_direc, file_type)
    plot_starting_time_distribution(winter_time_collection, result_direc, file_type, fname='winter_starting_time')
    plot_starting_time_distribution(summer_time_collection, result_direc, file_type, fname='summer_starting_time')
    plot_starting_time_distribution(combined_collection, result_direc, file_type, fname='combined')

    plot_annual_energy_consumption(annual_energy_consumption_kWh, result_direc, file_type)

    if convert_to_winter:
        comb_11_converted = {}
        for key, ele in comb_11kW_collection.items():
            converted = convert_to_winter_time(ele, summer_start_datetime, summer_end_datetime)
            comb_11_converted[key] = converted
        write_11kw_to_csv(comb_11_converted, result_direc)
    else:
        # Writing the 11 kW charging time series to .csv for charging optimization purposes
        write_11kw_to_csv(comb_11kW_collection, result_direc)

    # 100'000 takes approximately 2 hours
    #wc_collection = plot_worst_case_coincidence_parallel(comb_11kW_collection, result_direc, 'eps',
                                                         #num_simulations=100000, num_workers=22)
    print("done")


if __name__ == "__main__":
    main()

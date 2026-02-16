"""
Pre-Processing of private SMD

This file was used on the private SMD-data to break down the export files into single files for each id, containing
only timestamp and the ENERGY value instead of power (since most of the analysis relied on energy values, that are
converted to power values later on). Also, new column names, to correspond to the ones in the open dataset.

Run main, select source files --> find data in sorted_by_id file.

Result: Complete interpolated data for further processing (ideally you still have to break it down into one year of
data - small deviations shouldn't worsen the extraction performance too much - has not been systemically tested though.
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def ask_for_files():
    """
    Ask the user to select multiple .csv files.
    :return: List of selected file paths
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(
        title="Select CSV files",
        filetypes=[("CSV files", "*.csv")]
    )
    return file_paths


def find_missing_timestamps(df, sample_period='15min') -> dict:
    """
    This method generates a dictionary containing the missing periods in df:
    Key: [start_missing_0, end_missing_0]
    Value: number of missing timestamps in this period

    There is a key-value pair for each period, where at least one timestamp is missing

    :param df: Dataframe with possibly missing periods, timestamps are the index
    :param sample_period: Period, such that the frequency of values is known
    :return:
    """
    try:
        # Generate a complete date range from the first to the last timestamp with the given sample period
        complete_range = pd.date_range(start=df.timestamp.min(), end=df.timestamp.max(), freq=sample_period)

        # Find the missing timestamps by comparing the complete range with the DataFrame's index
        missing_timestamps = complete_range.difference(df.timestamp)
    except AttributeError:
        complete_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=sample_period)
        missing_timestamps = complete_range.difference(df.index)

    # Initialize a dictionary to store the missing periods
    missing_periods = {}

    # Iterate over the missing timestamps to find continuous periods
    if not missing_timestamps.empty:
        start_missing = missing_timestamps[0]
        end_missing = start_missing
        count_missing = 1

        for i in range(1, len(missing_timestamps)):
            if missing_timestamps[i] == end_missing + pd.Timedelta(minutes=15):
                end_missing = missing_timestamps[i]
                count_missing += 1
            else:
                missing_periods[(start_missing, end_missing)] = count_missing
                start_missing = missing_timestamps[i]
                end_missing = start_missing
                count_missing = 1

        # Add the last missing period
        missing_periods[(start_missing, end_missing)] = count_missing

    return missing_periods


def interpolate_df(df_smd: pd.DataFrame, num_weeks=8, sample_rate_minutes=15) -> (pd.DataFrame, list, list):
    """
    Interpolates a dataframe of an id using a combination of linear interpolation and historical data from
    time windows around missing timestamps.

    Parameters:
    - df_smd (pd.DataFrame): Dataframe of a specific id containing time-series data.
    - num_weeks (int): Total number of weeks for the sliding window for interpolation (default: 8 weeks).
    - sample_rate_minutes (int): The sample rate for the time-series data (default: 15 min).

    Returns:
    - df (pd.DataFrame): DataFrame containing interpolated data.
    - missing_timestamp_list (list): List of missing timestamps.
    - missing_timestamp_interpolated_values_list (list): Corresponding interpolated values.
    """
    df = df_smd.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")  # Ensure timestamps are properly formatted
    df = df.set_index("timestamp")

    missing_timestamps = find_missing_timestamps(df)  # Ensure this function returns timestamp-based ranges

    # Insert missing timestamps into df
    for (first_missing, last_missing), num_missing in missing_timestamps.items():
        for i in range(num_missing):
            missing_time = first_missing + pd.Timedelta(minutes=i * sample_rate_minutes)

            # Ensure missing_time is a Timestamp before using .union()
            if not isinstance(missing_time, pd.Timestamp):
                missing_time = pd.Timestamp(missing_time)

            df = df.reindex(df.index.union(pd.DatetimeIndex([missing_time]))).sort_index()  # Fix applied

    df.sort_index(inplace=True)

    # Lists to store missing timestamps and interpolated values
    missing_timestamp_list = []
    missing_timestamp_interpolated_values_list = []

    for (first_missing, last_missing), num_missing in missing_timestamps.items():
        for i in range(num_missing):
            missing_time = first_missing + pd.Timedelta(minutes=i * sample_rate_minutes)

            # Ensure missing_time is a Timestamp
            if not isinstance(missing_time, pd.Timestamp):
                missing_time = pd.Timestamp(missing_time)

            missing_timestamp_list.append(missing_time)
            day_of_week = missing_time.weekday()
            time_of_day = missing_time.time()

            # Define the window range
            start_window = max(df.index.min(), missing_time - pd.Timedelta(weeks=num_weeks / 2))
            end_window = min(df.index.max(), missing_time + pd.Timedelta(weeks=num_weeks / 2))

            # Ensure the window is num_weeks long
            if (end_window - start_window).days < (num_weeks * 7):
                if start_window == df.index.min():
                    end_window = start_window + pd.Timedelta(weeks=num_weeks)
                else:
                    start_window = end_window - pd.Timedelta(weeks=num_weeks)

            # Get the values within the window that match the same day of the week and time of day
            window_values = df.loc[
                (df.index >= start_window) & (df.index <= end_window) &
                (df.index.weekday == day_of_week) & (df.index.time == time_of_day),
                "value_kwh"
            ]

            # Calculate the average value and fill the NaN value
            avg_value = window_values.mean() if not window_values.empty else 0  # Avoid NaN values
            missing_timestamp_interpolated_values_list.append(avg_value)
            df.loc[missing_time, "value_kwh"] = avg_value

    return df.reset_index(), missing_timestamp_list, missing_timestamp_interpolated_values_list


def process_files(file_paths, output_dir):
    """
    Process each file and create individual .csv files for each zaehlpunkt_id.
    :param file_paths: List of file paths to process.
    :param output_dir: Directory to save the resulting files.
    """
    zaehlpunkt_data = {}

    # Read each file
    for file_path in file_paths:
        chunks = pd.read_csv(file_path, chunksize=100000, delimiter=',')
        for chunk in tqdm(chunks, desc=f"Processing {os.path.basename(file_path)}"):
            # Ensure required columns exist
            if ('zeitstempel_bis_edm' not in chunk.columns or 'value' not in chunk.columns or 'zaehlpunkt_id'
                    not in chunk.columns):
                print(f"Warning: Missing required columns in {file_path}. Skipping chunk.")
                continue

            # Convert zeitstempel_bis_edm to datetime and drop invalid rows
            chunk['zeitstempel_bis_edm'] = pd.to_datetime(chunk['zeitstempel_bis_edm'], errors='coerce')
            chunk = chunk.dropna(subset=['zeitstempel_bis_edm'])

            # Group by zaehlpunkt_id and store in dictionary
            for zaehlpunkt_id, group in chunk.groupby('zaehlpunkt_id'):
                group = group[['zeitstempel_bis_edm', 'value']].copy()
                group.rename(columns={'zeitstempel_bis_edm': 'timestamp', 'value': 'value_kwh'}, inplace=True)

                if zaehlpunkt_id not in zaehlpunkt_data:
                    zaehlpunkt_data[zaehlpunkt_id] = group
                else:
                    zaehlpunkt_data[zaehlpunkt_id] = pd.concat([zaehlpunkt_data[zaehlpunkt_id], group])

    # Write each zaehlpunkt_id to a separate .csv file
    for zaehlpunkt_id, data in tqdm(zaehlpunkt_data.items(), desc="Writing CSV files"):
        data = data.sort_values(by='timestamp')  # Sort by timestamp
        output_file = os.path.join(output_dir, f"{zaehlpunkt_id}.csv")
        data.to_csv(output_file, index=False)


def filter_and_delete_files(directory):
    """
    Filters and deletes files that do not meet the conditions:
    - Fraction of zero values must be less than 40%
    - Data range must be at least 365 days
    - No negative values
    - Sum of 'value_kwh' must be between 1.4 MWh and 100 MWh
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} not found. Please provide the correct path.")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".csv") and not filename.endswith("_missing_times.csv"):
            file_path = os.path.join(directory, filename)
            try:
                # Load the file
                df = pd.read_csv(file_path)

                if 'index' in df.columns:
                    df = df.rename(columns={'index': 'timestamp'})

                # Ensure 'timestamp' can be parsed as datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])

                # Ensure no negative values
                if (df['value_kwh'] < 0).any():
                    print(f"Deleting {filename}: contains negative values.")
                    os.remove(file_path)
                    continue

                # Calculate filtering conditions
                num_entries = len(df)
                zero_values_count = (df['value_kwh'] == 0).sum()
                zero_fraction = zero_values_count / num_entries if num_entries > 0 else 1
                total_days = (df['timestamp'].max() - df['timestamp'].min()).days
                # To convert kWh to Wh
                total_sum = df['value_kwh'].sum()*1e3

                # Check all filtering criteria and print the reason for deletion
                if zero_fraction >= 0.4:
                    print(f"Deleting {filename}: more than 40% zero values.")
                    os.remove(file_path)
                    continue

                if total_days < 365:
                    print(f"Deleting {filename}: data range is less than 365 days.")
                    os.remove(file_path)
                    continue

                if total_sum < 1.4e6:
                    print(f"Deleting {filename}: total energy consumption {total_sum*1e-6} MWh is less than 1.4 MWh.")
                    os.remove(file_path)
                    continue

                if total_sum >= 100e6:
                    print(f"Deleting {filename}: total energy consumption {total_sum*1e-6} exceeds 100 MWh.")
                    os.remove(file_path)
                    continue

            except Exception as e:
                print(f"Error processing {filename}: {e}")


def plot_histo_energy():
    directory = "//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/unlabelled_smd/sorted_by_id/"

    # List to store the sum of 'value_kwh' for each file
    sums = []

    # Iterate over all CSV files in the directory
    for filename in tqdm(os.listdir(directory), desc="Processing files"):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)

            try:
                # Load the file
                df = pd.read_csv(file_path)

                # Ensure no negative values
                df = df[df['value_kwh'] >= 0]

                # Calculate the sum of 'value_kwh' and add it to the list
                total_sum = df['value_kwh'].sum()
                sums.append(total_sum*1e-3)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Plot the histogram
    plt.hist(sums, bins=30, edgecolor='black')
    plt.xlabel('Sum of Values (MWh)')
    plt.ylabel('Frequency')
    plt.title('Total energy consumption')
    plt.grid()
    #plt.savefig("histogram_sum_of_values.png")
    plt.show()


def plot_missing_timestamps(directory):
    """
    Creates a plot showing when a lot of timestamps are missing across all files.
    It bins the missing timestamps per day to visualize periods with high data loss.

    Parameters:
    - directory (str): The path to the directory containing the CSV files.
    """
    missing_timestamps_all = []

    for filename in tqdm(os.listdir(directory), desc="Collecting missing timestamps"):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                # Load the CSV file
                df = pd.read_csv(file_path)

                # Ensure 'timestamp' is a datetime type and drop rows with missing timestamps
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp']).sort_values(by='timestamp')

                # Create a complete range of timestamps with 15-minute frequency
                full_range = pd.date_range(start=df['timestamp'].min(),
                                           end=df['timestamp'].max(),
                                           freq='15min')

                # Find the missing timestamps
                missing_timestamps = full_range.difference(df['timestamp'])

                if len(missing_timestamps) > 0:
                    missing_timestamps_all.extend(missing_timestamps)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if not missing_timestamps_all:
        print("No missing timestamps found in the dataset.")
        return

    # Convert missing timestamps into a DataFrame for easier grouping
    missing_df = pd.DataFrame({'timestamp': missing_timestamps_all})
    missing_df['date'] = missing_df['timestamp'].dt.date  # Aggregate per day

    # Count missing timestamps per day
    missing_counts = missing_df.groupby('date').size()

    # Plot the missing timestamp counts
    plt.figure(figsize=(15, 6))
    plt.plot(missing_counts.index, missing_counts.values, marker='o', linestyle='-', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Number of Missing Timestamps')
    plt.title('Missing Timestamps Over Time')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_missing_percentage(directory):
    """
    Creates a plot showing the percentage of missing timestamps for all timestamps in the date range.
    If for one timestamp no ID has a value, 100% should be displayed.

    Parameters:
    - directory (str): The path to the directory containing the CSV files.
    """
    total_ids = 0
    all_timestamps = set()
    missing_timestamps_per_timestamp = {}

    for filename in tqdm(os.listdir(directory), desc="Collecting missing timestamps"):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            total_ids += 1  # Count total number of IDs (files)

            # Load the CSV file
            df = pd.read_csv(file_path)

            # Ensure 'timestamp' is a datetime type and drop rows with missing timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).sort_values(by='timestamp')

            # Create a complete range of timestamps with 15-minute frequency
            full_range = pd.date_range(start=df['timestamp'].min(),
                                       end=df['timestamp'].max(),
                                       freq='15min')

            # Track all timestamps in the dataset
            all_timestamps.update(full_range)

            # Find the missing timestamps
            missing_timestamps = full_range.difference(df['timestamp'])

            if len(missing_timestamps) > 0:
                for ts in missing_timestamps:
                    if ts not in missing_timestamps_per_timestamp:
                        missing_timestamps_per_timestamp[ts] = set()
                    missing_timestamps_per_timestamp[ts].add(filename)

    # Convert missing timestamps to percentage of total IDs
    missing_percentages = {ts: len(ids) / total_ids * 100 for ts, ids in missing_timestamps_per_timestamp.items()}

    # Create a DataFrame and fill missing timestamps with 0% missing
    all_timestamps = sorted(all_timestamps)  # Ensure timestamps are in order
    missing_df = pd.DataFrame({'timestamp': all_timestamps})
    missing_df['percentage_missing'] = missing_df['timestamp'].map(missing_percentages).fillna(0)

    # Plot the missing timestamp percentage over time
    plt.figure(figsize=(15, 6))
    plt.plot(missing_df['timestamp'], missing_df['percentage_missing'], marker='o', linestyle='-', alpha=0.7)

    # Formatting the x-axis for better visibility
    plt.xlabel('Date')
    plt.ylabel('Percentage of Missing Timestamps')
    plt.title('Percentage of Missing Timestamps Over Time')
    plt.xticks(rotation=45)

    # Ensure all dates are visible by adjusting x-axis tick frequency
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show one tick per month
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as YYYY-MM
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def handle_duplicates(directory):
    """
    Reads all .csv files in directory, handles duplicates - takes max value if there are different values for the same
    timestamp - overwrites previous .csv.
    :param directory: directory, in which the .csv files are stored
    :return:
    """
    for filename in tqdm(os.listdir(directory), desc='Handling duplicates'):
        if filename.endswith(".csv") and not filename.endswith("missing_times.csv"):
            print(filename)
            file_path = os.path.join(directory, filename)

            df = pd.read_csv(file_path)
            # Some files are named 'index' instead of 'timestamp'
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'timestamp'})

            # Convert to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Remove duplicate timestamps, keeping the maximum value_kwh
            df = df.groupby('timestamp', as_index=False).max()

            # Sort timestamps in ascending order
            df = df.sort_values(by='timestamp', ascending=True)

            # Overwrite the original file
            df.to_csv(file_path, index=False)
    print("----- Duplicates handled -----")


def interpolate_all(directory):
    """
    Function to interpolate all single id csv files.
    :param directory: Directory containing all the files to be interpolated and replaced by the interpolated version
    :return:
    """
    for filename in tqdm(os.listdir(directory), desc="Interpolating files"):
        if filename.endswith(".csv") and not filename.endswith("_missing_times.csv"):  # Ignore missing time files
            file_path = os.path.join(directory, filename)
            id_name = filename.split(".")[0]

            # Load the CSV file
            df = pd.read_csv(file_path, parse_dates=['timestamp'])

            # Interpolate missing values
            df_interpolated, missing_timestamps, interpolated_values = interpolate_df(df)

            # Overwrite original file with interpolated data
            df_interpolated.to_csv(file_path, index=False)

            # Save missing timestamps
            if missing_timestamps:
                missing_df = pd.DataFrame(
                    {'timestamp': missing_timestamps, 'interpolated_value': interpolated_values})
                missing_file_path = os.path.join(directory, f"{id_name}_missing_times.csv")
                missing_df.to_csv(missing_file_path, index=False)

    print("---------- Interpolation for all .csv files complete - Files have been replaced ----------")


if __name__ == "__main__":
    # Ask for files
    file_paths = ask_for_files()
    if not file_paths:
        print("No files selected. Exiting...")
    else:
        # Output directory
        output_dir = "//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/unlabelled_smd/sorted_by_id/"
        os.makedirs(output_dir, exist_ok=True)

        # Process files
        process_files(file_paths, output_dir)

    direc = "//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/unlabelled_smd/sorted_by_id/"

    # Filter and delete files
    filter_and_delete_files(direc)

    # Handle duplicated timestamps
    handle_duplicates(direc)

    # Generates plot for missing timestamps to get an idea of the completeness of the investigated data
    plot_missing_percentage(direc)

    # Interpolation of the missing timestamps
    interpolate_all(direc)

    print("Preprocessing complete")

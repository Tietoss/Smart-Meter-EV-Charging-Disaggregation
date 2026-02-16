import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def ask_for_files():
    """
    Method to ask for multiple file locations if not stored at the same location.
    :return: List of file paths
    """
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 2.0)

    file_paths = filedialog.askopenfilenames(
        title="Select files (Excel or CSV)",
        filetypes=[("Excel files", "*.gz"), ("CSV files", "*.csv")]
    )

    root.withdraw()
    return file_paths

def process_file(file_path, id_data, histogram_data):
    """
    Process a single file and update the id_data and histogram_data dictionaries.
    """
    chunksize = 100000
    if file_path.endswith('.gz'):
        chunks = pd.read_csv(file_path, compression='gzip', chunksize=chunksize, delimiter=',')
    else:
        chunks = pd.read_csv(file_path, chunksize=chunksize, delimiter=',')

    for chunk in tqdm(chunks, desc=f"Processing {os.path.basename(file_path)}"):
        # Ensure the required columns are present
        if 'messzeitstempel' not in chunk.columns or 'wert_W' not in chunk.columns:
            print(f"Warning: Columns 'messzeitstempel' or 'wert_W' not found in chunk from {file_path}. Skipping...")
            continue

        chunk['messzeitstempel'] = pd.to_datetime(chunk['messzeitstempel'])

        grouped = chunk.groupby('zaehlpunkt_id')
        for id, group in grouped:
            if id not in id_data:
                id_data[id] = {
                    'num_entries': 0,
                    'zero_values_count': 0,
                    'earliest_date': pd.Timestamp(group['messzeitstempel'].min()),
                    'latest_date': pd.Timestamp(group['messzeitstempel'].max())
                }
            id_data[id]['num_entries'] += len(group)
            id_data[id]['zero_values_count'] += (group['wert_W'] == 0).sum()

            earliest_date = pd.Timestamp(group['messzeitstempel'].min())
            latest_date = pd.Timestamp(group['messzeitstempel'].max())

            id_data[id]['earliest_date'] = min(id_data[id]['earliest_date'], earliest_date)
            id_data[id]['latest_date'] = max(id_data[id]['latest_date'], latest_date)

            for pwr_value in group['wert_W']:
                if pwr_value == 0:
                    histogram_data['0-1000'] += 1
                elif pwr_value >= 30000:
                    histogram_data['>=30000'] += 1
                elif pwr_value < -30000:
                    histogram_data['<-30000'] += 1
                else:
                    bin_key = f'{int(pwr_value // 1000) * 1000}-{int(pwr_value // 1000) * 1000 + 1000}'
                    histogram_data[bin_key] += 1

# Main script
file_paths = ask_for_files()
id_data = {}
histogram_data = {f'{i}-{i + 1000}': 0 for i in range(-30000, 30000, 1000)}
histogram_data['>=30000'] = 0
histogram_data['<-30000'] = 0

for file_path in file_paths:
    try:
        process_file(file_path, id_data, histogram_data)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Create the result dataframe
result_df = pd.DataFrame([
    {
        'id': id,
        'num_entries': data['num_entries'],
        'avg_zero_values': data['zero_values_count'] / data['num_entries'],
        'total_days': (data['latest_date'] - data['earliest_date']).days
    }
    for id, data in id_data.items()
])

# Filter IDs with < 40% zero values and >= 365 total days
filtered_ids = result_df[(result_df['avg_zero_values'] < 0.4) & (result_df['total_days'] >= 365)]['id']

# Generate CSV files for each filtered ID
for file_path in file_paths:
    if file_path.endswith('.gz'):
        chunks = pd.read_csv(file_path, compression='gzip', chunksize=100000, delimiter=';')
    else:
        chunks = pd.read_csv(file_path, chunksize=100000, delimiter=';')

    for chunk in chunks:
        chunk['messzeitstempel'] = pd.to_datetime(chunk['messzeitstempel'], errors='coerce')
        chunk = chunk.dropna(subset=['messzeitstempel'])

        grouped = chunk.groupby('zaehlpunkt_id')
        for id in filtered_ids:
            if id in grouped.groups:
                group = grouped.get_group(id)
                group = group[['messzeitstempel', 'wert_W']].sort_values(by='messzeitstempel')

                output_file = (f"//d.ethz.ch/groups/itet/eeh/psl/stud/luelmiger/private/unlabelled_smd/"
                               f"sorted_by_id/{id}.csv")
                group.to_csv(output_file, index=False)
                print(f"Created {output_file}")

print("Processing complete.")

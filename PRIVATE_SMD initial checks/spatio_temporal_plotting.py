"""
This file is used to generate spatio-temporal EV charging patterns. The identified charging timeseries are
generated in EV_charging_session_identification.py and are stored in a pickle format. That's why they can be read
directly from pickle, if already generated.
"""
import pickle
import os
import subprocess
import shutil

import geopandas as gpd

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

import numpy as np
import contextily as cx
from tqdm import tqdm
from joblib import Parallel, delayed


def load_base_map(shapefile_path):
    return gpd.read_file(shapefile_path)


def convert_timeseries_with_locations(timeseries_dict, loc_data, crs_map="EPSG:2056"):
    """
    The timeseries data comes from one file, the location data from another - therefore these two are merged and
    the coordinates are adjusted, such that they are aligned with the format of the DSO area shape file.
    :param timeseries_dict:
    :param loc_data:
    :param crs_map:
    :return:
    """
    # Step 1: Convert timeseries dict into long DataFrame
    records = []
    for appliance_id, series in timeseries_dict.items():
        df = series.reset_index()
        df.columns = ['timestamp', 'state']
        df['id'] = appliance_id
        records.append(df)
    timeseries_df = pd.concat(records, ignore_index=True)

    # Step 2: Create GeoDataFrame from location info
    loc_df = loc_data.rename(columns={
        'an_zp': 'id',
        'rechtswert': 'easting',
        'hochwert': 'northing'
    })

    loc_gdf = gpd.GeoDataFrame(
        loc_df,
        geometry=gpd.points_from_xy(loc_df.easting, loc_df.northing),
        crs="EPSG:4326"  # <-- This is the actual CRS of your GPS-like coords in timeseries data
    )

    # Step 3: Reproject to match map (if needed)
    if loc_gdf.crs != crs_map:
        loc_gdf = loc_gdf.to_crs(crs_map)

    # Step 4: Extract X/Y from geometry (in map CRS)
    loc_gdf['x'] = loc_gdf.geometry.x
    loc_gdf['y'] = loc_gdf.geometry.y

    # Step 5: Merge with timeseries
    merged_df = timeseries_df.merge(loc_gdf[['id', 'x', 'y']], on='id', how='left')

    return merged_df


def stitch_frames_to_video(frame_dir, output_path, fps=80):
    input_pattern = os.path.join(frame_dir, "frame_%05d.png")
    temp_output_path = output_path.replace(".mp4", "_temp.mp4")

    # Try libx264 first
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ], check=True)
        return

    except subprocess.CalledProcessError:
        print("libx264 encoding failed or unavailable. Trying fallback...")

    # Fallback to mpeg4
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "mpeg4",
            "-qscale:v", "2",  # 1=best, 31=worst
            temp_output_path
        ], check=True)
        shutil.move(temp_output_path, output_path)

    except subprocess.CalledProcessError:
        print("Failed to create video using both libx264 and mpeg4.")


def render_frame(snapshot, shapefile_gdf, timestamp, output_path, norm, max_marker_size=300):
    fig, ax = plt.subplots()

    shapefile_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8)

    if snapshot.empty:
        plt.close(fig)
        return

    values = snapshot['state'].values
    norm_sizes = (values - values.min()) / (values.max() - values.min() + 1e-5)
    sizes = norm_sizes * max_marker_size + 10

    jitter_strength = 5
    jitter_x = snapshot.geometry.x + np.random.uniform(-jitter_strength, jitter_strength, size=len(snapshot))
    jitter_y = snapshot.geometry.y + np.random.uniform(-jitter_strength, jitter_strength, size=len(snapshot))

    scatter = ax.scatter(
        jitter_x, jitter_y,
        s=sizes,
        c=values,
        cmap='plasma',
        norm=norm,
        alpha=0.7,
        edgecolors='k',
        linewidth=0.5
    )

    cx.add_basemap(ax, crs=shapefile_gdf.crs.to_string(), source=cx.providers.CartoDB.Voyager)

    ax.set_title(f"EV charging activity at {timestamp.strftime('%Y-%m-%d %H:%M')}", fontsize=14)
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Charging Power (kW)')

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_all_frames(df, shapefile_gdf, output_dir, n_jobs=8):
    os.makedirs(output_dir, exist_ok=True)
    timestamps = sorted(df[df['state'] > 0]['timestamp'].unique())

    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['x'], df['y']),
        crs=shapefile_gdf.crs
    ).to_crs(epsg=3857)

    shapefile_gdf = shapefile_gdf.to_crs(epsg=3857)
    norm = mcolors.Normalize(vmin=0, vmax=22.2)

    Parallel(n_jobs=n_jobs)(delayed(render_frame)(
        snapshot=gdf_points[gdf_points['timestamp'] == ts],
        shapefile_gdf=shapefile_gdf,
        timestamp=ts,
        output_path=os.path.join(output_dir, f"frame_{i:05d}.png"),
        norm=norm
    ) for i, ts in enumerate(timestamps))


def plot_snapshot(df, shapefile_gdf, timestamp, save_path=None, max_marker_size=150,
                  municipality_gdf=None, pop_df=None):
    fntsz = 16
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter data for the selected timestamp
    snapshot = df[df['timestamp'] == timestamp]

    # Normalize state values for marker sizes
    values = snapshot['state'].values
    norm_sizes = (values - values.min()) / (values.max() - values.min() + 1e-5)
    sizes = norm_sizes * max_marker_size + 10

    # Add jitter to avoid overlapping markers
    jitter_strength = 5
    jitter_x = snapshot['x'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(snapshot))
    jitter_y = snapshot['y'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(snapshot))

    # --- Plot municipality population density ---
    if municipality_gdf is not None and pop_df is not None:
        # Keep only rows for population density
        pop_df = pop_df[pop_df['VARIABLE'] == "Einwohner/-innen pro km² Gesamtfläche"].copy()

        # Clean and match names
        pop_df['GEO_NAME'] = pop_df['GEO_NAME'].str.strip().str.lower()
        municipality_gdf = municipality_gdf.copy()
        municipality_gdf['NAME'] = municipality_gdf['NAME'].str.strip().str.lower()

        # Reproject municipality_gdf to match shapefile_gdf exactly
        municipality_gdf = municipality_gdf.to_crs(shapefile_gdf.crs)
        # Clip municipalities to DSO area
        muni_clipped_raw = gpd.overlay(municipality_gdf, shapefile_gdf, how='intersection')

        # Merge with population density values
        merged_muni = muni_clipped_raw.merge(
            pop_df[['GEO_NAME', 'VALUE']],
            left_on='NAME', right_on='GEO_NAME',
            how='left'
        )

        # Clean values
        merged_muni['VALUE'] = pd.to_numeric(merged_muni['VALUE'], errors='coerce')
        merged_muni = merged_muni.dropna(subset=['VALUE'])

        # Normalize based on actual values
        pop_min, pop_max = merged_muni['VALUE'].min(), merged_muni['VALUE'].max()
        pop_norm = mcolors.LogNorm(vmin=pop_min, vmax=pop_max)

        # Plot population density layer
        pop_cmap = plt.cm.YlOrRd
        merged_muni.plot(
            column='VALUE',
            cmap=pop_cmap,
            ax=ax,
            alpha=0.5,
            linewidth=0.2,
            norm=pop_norm,
            edgecolor='none'
        )

        # Add colorbar for population density
        sm_pop = ScalarMappable(norm=pop_norm, cmap=pop_cmap)
        sm_pop.set_array([])
        cax2 = fig.add_axes([0.91, 0.25, 0.015, 0.5])  # Population Density (right)
        cbar2 = fig.colorbar(sm_pop, cax=cax2)
        cbar2.set_label('Population Density (people/km²)', fontsize=fntsz)

    # Plot DSO area outline
    shapefile_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8)

    # Plot EV charging activity points
    scatter = ax.scatter(
        jitter_x, jitter_y,
        s=sizes,
        c=values,
        cmap='Blues',
        alpha=0.9,
        edgecolors='k',
        linewidth=0.5
    )

    # Add basemap tiles
    cx.add_basemap(ax, crs=shapefile_gdf.crs.to_string(), source=cx.providers.CartoDB.Voyager)

    # --- Add sum of charging power as text annotation ---
    total_power = snapshot['state'].sum()
    ax.text(
        0.99, 0.01,
        f"Summed EV charging magnitude: {total_power:.1f} kW",
        transform=ax.transAxes,
        fontsize=fntsz,
        color='black',
        ha='right',
        va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.95)
    )

    # Title
    if not save_path:
        ax.set_title(f"EV charging activity at {timestamp}", fontsize=14)
    ax.set_axis_off()

    # Add colorbar for charging activity
    sm_charging = ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=values.max()), cmap='Blues')
    sm_charging.set_array([])
    cax1 = fig.add_axes([0.82, 0.25, 0.015, 0.5])  # Charging Power (left)
    cbar1 = fig.colorbar(sm_charging, cax=cax1)
    cbar1.set_label('Charging Power (kW)', fontsize=fntsz)

    # Zoom to DSO shape
    ax.set_xlim(shapefile_gdf.total_bounds[[0, 2]])
    ax.set_ylim(shapefile_gdf.total_bounds[[1, 3]])

    fig.subplots_adjust(right=0.95)
    # Save or display
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=400)
        fig.show()
    else:
        fig.show()


def plot_municipality_concurrency(df, shapefile_gdf, timestamp, municipality_gdf, pop_df=None, save_path=None):
    """
    Plots the concurrency (0, 1) on the level of the municipality (current charging magnitude in municipality /
    theoretical max value, when everybody in the municipality would charge at the same time
    :param df:
    :param shapefile_gdf:
    :param timestamp:
    :param municipality_gdf:
    :param pop_df:
    :param save_path:
    :return:
    """
    fntsz = 16

    fig, ax = plt.subplots(figsize=(10, 8))

    # Ensure CRS matches for spatial joins
    municipality_gdf = municipality_gdf.to_crs(shapefile_gdf.crs)

    # Convert df to GeoDataFrame using x/y
    df_gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df['x'], df['y']),
        crs=shapefile_gdf.crs
    )

    # Assign each EV to a municipality
    df_gdf = gpd.sjoin(df_gdf, municipality_gdf[['NAME', 'geometry']], how='left', predicate='within')
    df_gdf = df_gdf.rename(columns={'NAME': 'municipality'}).drop(columns='index_right')

    # Filter to timestamp
    df_gdf['timestamp'] = pd.to_datetime(df_gdf['timestamp'])
    snapshot = df_gdf[df_gdf['timestamp'] == timestamp].copy()

    # Calculate max power per EV (across all time)
    max_power_per_ev = df_gdf.groupby('id')['state'].max().reset_index().rename(columns={'state': 'max_state'})
    snapshot = snapshot.merge(max_power_per_ev, on='id', how='left')

    # Compute concurrency per municipality
    grouped = snapshot.groupby('municipality').agg({
        'state': 'sum',
        'max_state': 'sum'
    }).reset_index()
    grouped['concurrency'] = grouped['state'] / grouped['max_state']
    grouped['concurrency'] = grouped['concurrency'].clip(0, 1)

    # Join concurrency back to snapshot
    snapshot = snapshot.merge(grouped[['municipality', 'concurrency']], on='municipality', how='left')

    # Jitter to avoid overlaps
    jitter_strength = 5
    jitter_x = snapshot.geometry.x + np.random.uniform(-jitter_strength, jitter_strength, size=len(snapshot))
    jitter_y = snapshot.geometry.y + np.random.uniform(-jitter_strength, jitter_strength, size=len(snapshot))

    # --- Background: population density ---
    if pop_df is not None:
        pop_df = pop_df[pop_df['VARIABLE'] == "Einwohner/-innen pro km² Gesamtfläche"].copy()
        pop_df['GEO_NAME'] = pop_df['GEO_NAME'].str.strip().str.lower()
        municipality_gdf = municipality_gdf.copy()
        municipality_gdf['NAME'] = municipality_gdf['NAME'].str.strip().str.lower()

        muni_clipped = gpd.overlay(municipality_gdf, shapefile_gdf, how='intersection')
        merged_muni = muni_clipped.merge(
            pop_df[['GEO_NAME', 'VALUE']],
            left_on='NAME', right_on='GEO_NAME',
            how='left'
        )

        merged_muni['VALUE'] = pd.to_numeric(merged_muni['VALUE'], errors='coerce')
        merged_muni = merged_muni.dropna(subset=['VALUE'])
        merged_muni = merged_muni[merged_muni['VALUE'] > 0]

        pop_min, pop_max = merged_muni['VALUE'].min(), merged_muni['VALUE'].max()
        pop_norm = mcolors.LogNorm(vmin=pop_min, vmax=pop_max)
        pop_cmap = plt.cm.YlOrRd

        merged_muni.plot(
            column='VALUE',
            cmap=pop_cmap,
            ax=ax,
            alpha=0.5,
            linewidth=0.2,
            norm=pop_norm,
            edgecolor='none'
        )

        sm_pop = ScalarMappable(norm=pop_norm, cmap=pop_cmap)
        sm_pop.set_array([])
        cax2 = fig.add_axes([0.91, 0.25, 0.015, 0.5])
        cbar2 = fig.colorbar(sm_pop, cax=cax2)
        cbar2.set_label('Population Density (people/km²)', fontsize=fntsz)

    # --- Plot EV points colored by municipality concurrency ---
    conc_norm = mcolors.PowerNorm(gamma=0.9, vmin=0, vmax=1)

    scatter = ax.scatter(
        jitter_x, jitter_y,
        s=30,
        c=snapshot['concurrency'],
        cmap='Purples',
        norm=conc_norm,
        alpha=0.8,
        edgecolors='k',
        linewidth=0.5
    )

    shapefile_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8)
    cx.add_basemap(ax, crs=shapefile_gdf.crs.to_string(), source=cx.providers.CartoDB.Voyager)

    sm_conc = ScalarMappable(norm=conc_norm, cmap='BuPu')
    sm_conc.set_array([])
    cax1 = fig.add_axes([0.82, 0.25, 0.015, 0.5])
    cbar1 = fig.colorbar(sm_conc, cax=cax1)
    cbar1.set_label('EV Charging Concurrency per Municipality', fontsize=fntsz)

    # Total power annotation
    total_power = snapshot['state'].sum()
    ax.text(
        0.99, 0.01,
        f"Summed EV charging magnitude: {total_power:.1f} kW",
        transform=ax.transAxes,
        fontsize=fntsz,
        color='black',
        ha='right',
        va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.95)
    )

    if not save_path:
        ax.set_title(f"Municipality concurrency at {timestamp}", fontsize=14)
    ax.set_axis_off()
    ax.set_xlim(shapefile_gdf.total_bounds[[0, 2]])
    ax.set_ylim(shapefile_gdf.total_bounds[[1, 3]])

    fig.subplots_adjust(right=0.95)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=400)
        plt.close()
    else:
        plt.show()


def generate_movie(df, shapefile_gdf, output_path, fps=80, max_marker_size=300):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    timestamps = sorted(df[df['state'] > 0]['timestamp'].unique())

    # Convert df to GeoDataFrame with geometry
    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['x'], df['y']),
        crs=shapefile_gdf.crs
    )

    # Reproject to EPSG:3857 for tile map
    gdf_points = gdf_points.to_crs(epsg=3857)
    shapefile_gdf = shapefile_gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots()
    norm = mcolors.Normalize(vmin=0, vmax=22.2)  # Fixed normalization for color mapping
    scatter = None  # placeholder for colorbar binding

    progress = tqdm(total=len(timestamps), desc="Generating movie")

    def update(i):
        nonlocal scatter  # allow access outside function
        ax.clear()
        current_time = timestamps[i]
        shapefile_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8)

        snapshot = gdf_points[gdf_points['timestamp'] == current_time]
        if snapshot.empty:
            return

        values = snapshot['state'].values
        norm_sizes = (values - values.min()) / (values.max() - values.min() + 1e-5)
        sizes = norm_sizes * max_marker_size + 10

        jitter_strength = 5
        jitter_x = snapshot.geometry.x + np.random.uniform(-jitter_strength, jitter_strength, size=len(snapshot))
        jitter_y = snapshot.geometry.y + np.random.uniform(-jitter_strength, jitter_strength, size=len(snapshot))

        scatter = ax.scatter(
            jitter_x, jitter_y,
            s=sizes,
            c=values,
            cmap='plasma',
            norm=norm,  # Use fixed normalization here
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )

        cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Voyager)

        ax.set_title(f"EV charging activity at {current_time.strftime('%Y-%m-d %H:%M')}", fontsize=14)
        ax.set_axis_off()

        progress.update(1)

    # First dummy frame for colorbar setup
    dummy_values = np.linspace(0, 22.2, 100)
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array(dummy_values)
    fig.colorbar(sm, ax=ax, label='Charging Power (kW)')

    ani = animation.FuncAnimation(fig, update, frames=len(timestamps), repeat=False)

    writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
    ani.save(
        output_path,
        writer=writer,
        dpi=200,
    )
    progress.close()
    plt.close()


def main():
    # File path to the pickled timeseries EV charging data
    fpath_ts = ("fpath/results/all_identified_charging_ts.pkl")
    # Shape file path for CKW area
    fpath_shp = "fpath/strom_oe_flaeche.shp"
    # File path containing the preprocessed location data
    fpath_loc = "fpath/loc_data.csv"
    res_direc = "fpath/map_plot/"
    # Output paths
    frame_dir = "fpath/map_plot/frames"
    output_video_path = res_direc + "ev_charging.mp4"
    population_density_path = ("fpath/population_density_switzerland_2023_bfs.csv")

    municipality_shp_file = "fpath/swiss_municipalities/swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.shp"

    # Read the pickled data
    with open(fpath_ts, "rb") as f:
        ts_data = pickle.load(f)

    # df containing id, municipality, name of the transformer station(s), r_value/l_value of coordinate
    loc_data = pd.read_csv(fpath_loc)
    # Load map data
    gdf = load_base_map(fpath_shp)
    municipality_gdf = load_base_map(municipality_shp_file)
    # Read population density data
    pop_df = pd.read_csv(population_density_path, delimiter=';')
    # merge location and time series data
    merged_df = convert_timeseries_with_locations(ts_data, loc_data)

    """
    plot_municipality_concurrency(merged_df,
                                  gdf,
                                  pd.Timestamp("2024-11-21 23:30"),
                                  municipality_gdf,
                                  pop_df,
                                  save_path=res_direc+'snapshot_municipality_concurrency')

    plot_snapshot(
        merged_df,
        gdf,
        pd.Timestamp("2024-11-21 23:30"),
        municipality_gdf=municipality_gdf,
        pop_df=pop_df,
        save_path=res_direc+'snapshot.png'
    )
    """
    # Filter for movie
    start_ts = pd.Timestamp("2024-11-03 00:00:00")
    end_ts = pd.Timestamp("2024-11-28 23:45:00")
    filtered_df = merged_df[(merged_df['timestamp'] >= start_ts) & (merged_df['timestamp'] <= end_ts)]

    generate_all_frames(filtered_df, gdf, output_dir=frame_dir, n_jobs=22)
    stitch_frames_to_video(frame_dir, output_video_path, fps=80)

    # Make the movie
    generate_movie(filtered_df, gdf, output_path=res_direc + "ev_charging.mp4", fps=80)


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import re
from geopy.distance import geodesic
import os

def clean_station_name(name):
    """
    Standardizes station names: converts to lowercase, strips whitespace,
    removes special characters, removes all internal spaces, and corrects known typos.
    """
    # Dictionary for known typos or alternative spellings.
    # This maps coordinate file names to discharge file names
    name_correction_map = {
        "parmanpur": "paramanpur",
        "bamnidih": "bamnidhi",
    }
    
    # Standardize to a common format first
    name = str(name).strip().lower()
    name = re.sub(r"[’'\"`]", "", name)
    # FIX: Remove all whitespace characters (including spaces between words)
    name = re.sub(r"\s+", "", name)
    
    # Apply correction if the cleaned name is in our map
    corrected_name = name_correction_map.get(name, name)
    
    # Ensure the corrected name matches what's expected in the discharge data
    # The discharge data has these exact lowercase names
    expected_names = {
        'andhiyarkhore', 'bamnidhi', 'baronda', 'basantpur', 'boudh', 
        'ghatora', 'jondhra', 'kantamal', 'kelo', 'kesinga', 'kotni', 
        'kurubhata', 'padampur', 'paramanpur', 'patharidih', 'rajim', 
        'rampur', 'salebhata', 'seorinarayan', 'simga', 'sundergarh', 'tikarapara'
    }
    
    if corrected_name not in expected_names:
        print(f"Warning: Station name '{corrected_name}' not in expected list. Original: '{name}'")
    
    return corrected_name

def parse_lat_lon(coord_str, is_latitude=True):
    """Parses a latitude or longitude string into a decimal degree float."""
    if isinstance(coord_str, (int, float)):
        val = float(coord_str)
        if (is_latitude and not -90 <= val <= 90) or (not is_latitude and not -180 <= val <= 180):
            raise ValueError(f"Coordinate {val} is out of bounds.")
        return val

    s = str(coord_str).strip()
    s_spaced = re.sub(r"[o°'\"`‘’]", " ", s)
    numbers = [float(n) for n in re.findall(r'(-?\d+\.?\d*)', s_spaced)]
    direction = re.search(r'([NSEWnsew])', s)
    direction_char = direction.group(1).upper() if direction else None

    if not numbers:
        raise ValueError(f"No numerical parts found in coordinate string: '{coord_str}'")

    deg, min_, sec = numbers[0], 0.0, 0.0
    if len(numbers) > 1: min_ = numbers[1]
    if len(numbers) > 2: sec = numbers[2]
        
    decimal_deg_abs = abs(deg) + min_/60 + sec/3600
    final_decimal_deg = decimal_deg_abs
    
    if direction_char:
        if (is_latitude and direction_char == 'S') or (not is_latitude and direction_char == 'W'):
            final_decimal_deg *= -1
    elif deg < 0:
        final_decimal_deg *= -1

    if (is_latitude and not -90 <= final_decimal_deg <= 90) or \
       (not is_latitude and not -180 <= final_decimal_deg <= 180):
        raise ValueError(f"Parsed coordinate {final_decimal_deg} is out of bounds.")
        
    return final_decimal_deg

def load_and_preprocess_data(discharge_path, lat_long_path, contrib_path=None):
    """
    Loads, cleans, and standardizes all input data sources, ensuring a consistent
    set of stations is used throughout the pipeline.
    """
    print("--- Loading and Preprocessing All Data Sources ---")
    
    # 1. Load Discharge Data
    try:
        df_discharge = pd.read_csv(discharge_path, index_col=0, parse_dates=True, dayfirst=True)
        df_discharge.index.name = 'date'
        
        all_cols = df_discharge.columns.tolist()
        potential_discharge_cols = [col for col in all_cols if not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))]
        temporal_feature_cols = [col for col in all_cols if col not in potential_discharge_cols]

        df_discharge_stations = df_discharge[potential_discharge_cols].copy()
        df_discharge_stations.columns = df_discharge_stations.columns.map(clean_station_name)
        
        if temporal_feature_cols:
            df_temporal_features = df_discharge[temporal_feature_cols].copy()
            df_discharge = pd.concat([df_discharge_stations, df_temporal_features], axis=1)
        else:
            df_discharge = df_discharge_stations

        for col in potential_discharge_cols: 
            cleaned_col_name = clean_station_name(col)
            if cleaned_col_name in df_discharge.columns:
                df_discharge[cleaned_col_name] = pd.to_numeric(df_discharge[cleaned_col_name], errors='coerce')

    except Exception as e:
        print(f"FATAL: Could not load discharge data from '{discharge_path}'. Error: {e}")
        return None, None, None, None, None

    # 2. Load Coordinate Data
    try:
        df_coords = pd.read_csv(lat_long_path)
        column_rename_map = {'Latitude (N)': 'Latitude', 'Longitude (E)': 'Longitude', 'Name of site': 'Station'}
        df_coords.rename(columns={k: v for k, v in column_rename_map.items() if k in df_coords.columns}, inplace=True)
        if 'Station' in df_coords.columns:
            df_coords.set_index('Station', inplace=True)
        else:
            raise ValueError("'Name of site' column not found in coordinate file.")
        df_coords.index = df_coords.index.map(clean_station_name)
    except Exception as e:
        print(f"FATAL: Could not load coordinate data from '{lat_long_path}'. Error: {e}")
        return None, None, None, None, None

    # *** FUNDAMENTAL FIX: Create a canonical list of stations ***
    discharge_stations_after_cleaning = set([col for col in df_discharge.columns if col in df_coords.index or not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))])
    discharge_stations_after_cleaning = set([col for col in discharge_stations_after_cleaning if not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))])
    coords_stations = set(df_coords.index)
    canonical_stations = sorted(list(discharge_stations_after_cleaning.intersection(coords_stations)))

    dropped_from_discharge = discharge_stations_after_cleaning - coords_stations
    if dropped_from_discharge:
        print(f"Warning: The following stations are in the discharge file but missing coordinates. They will be DROPPED: {list(dropped_from_discharge)}")

    dropped_from_coords = coords_stations - discharge_stations_after_cleaning
    if dropped_from_coords:
        print(f"Warning: The following stations have coordinates but are not in the discharge file. They will be IGNORED: {list(dropped_from_coords)}")

    print(f"\nUsing {len(canonical_stations)} stations found in both discharge and coordinate files.")
    print(f"DEBUG: Canonical station list: {canonical_stations}") # Your original debug line

    cols_to_keep_in_discharge = canonical_stations + temporal_feature_cols
    df_discharge = df_discharge[cols_to_keep_in_discharge]
    df_coords = df_coords.loc[canonical_stations]

    df_coords['Latitude'] = df_coords['Latitude'].apply(lambda x: parse_lat_lon(x, is_latitude=True))
    df_coords['Longitude'] = df_coords['Longitude'].apply(lambda x: parse_lat_lon(x, is_latitude=False))

    # 3. Load Contributor Data (Optional)
    df_contrib = None
    if contrib_path:
        try:
            df_contrib_matrix = pd.read_csv(contrib_path, index_col=0)
            
            # --- NEW DEBUG ---
            print(f"DEBUG: Contrib matrix loaded. Index: {df_contrib_matrix.index.tolist()[:5]}...")
            print(f"DEBUG: Contrib matrix columns: {df_contrib_matrix.columns.tolist()[:5]}...")
            
            v_code_to_name = {v_code: clean_station_name(name) for v_code, name in df_contrib_matrix['Name of site'].items()}
            
            # --- NEW DEBUG ---
            print(f"DEBUG: v_code_to_name map created. Example: V1 -> {v_code_to_name.get('V1')}")

            matrix_v_codes = [col for col in df_contrib_matrix.columns if col.startswith('V')]
            
            # --- NEW DEBUG ---
            print(f"DEBUG: Found {len(matrix_v_codes)} matrix_v_codes (columns): {matrix_v_codes[:5]}...")

            contrib_pairs = []
            
            # --- NEW DEBUG ---
            print("DEBUG: Starting iteration over matrix rows...")
            
            for station_v_code, row in df_contrib_matrix.iterrows():
                station_name = v_code_to_name.get(station_v_code)
                if not station_name:
                    # --- NEW DEBUG ---
                    print(f"DEBUG: Skipping row. Could not find name for station_v_code: {station_v_code}")
                    continue
                
                for contributor_v_code in matrix_v_codes:
                    # --- MODIFIED & MORE ROBUST CHECK ---
                    if pd.to_numeric(row[contributor_v_code]) == 1:
                        contributor_name = v_code_to_name.get(contributor_v_code)
                        if contributor_name:
                            # --- NEW DEBUG ---
                            print(f"DEBUG: Found pair! Station '{station_name}' (from {station_v_code}) -> Contributor '{contributor_name}' (from {contributor_v_code})")
                            contrib_pairs.append({'station': station_name, 'contributor': contributor_name})
                        else:
                            # --- NEW DEBUG ---
                            print(f"DEBUG: Skipping pair. Found '1' but no name for contributor_v_code: {contributor_v_code}")

            if contrib_pairs:
                df_contrib = pd.DataFrame(contrib_pairs)
                print(f"DEBUG: Found {len(df_contrib)} contributor pairs BEFORE filtering.") # Your original line
                df_contrib = df_contrib[df_contrib['station'].isin(canonical_stations) & df_contrib['contributor'].isin(canonical_stations)]
                print(f"DEBUG: Have {len(df_contrib)} contributor pairs remaining AFTER filtering.") # Your original line
                
                if df_contrib.empty:
                    print("DEBUG: Contrib pairs were found, but ALL were filtered out because names did not match canonical_stations list.")
                
                print(f"Loaded and parsed contributor matrix with {len(df_contrib)} relationships (filtered for canonical stations).")
            else:
                print("DEBUG: Contributor file was loaded, but no 'contrib_pairs' (rows with '1') were found in the matrix.") # Your original line
                df_contrib = None
        except Exception as e:
            print(f"FATAL DEBUG: Could not load or parse contributor data from '{contrib_path}'. Error: {e}")
            df_contrib = None
    else:
        print("DEBUG: 'contrib_path' was not provided (is None or empty string).")
            
    # 4. Create Station Name/V-code Mappings
    vcode_to_station, station_to_vcode = {}, {}
    if 'v_code' in df_coords.columns:
        coords_unique = df_coords[~df_coords.index.duplicated(keep='first')]
        vcode_to_station = coords_unique['v_code'].to_dict()
        station_to_vcode = {v: k for k, v in vcode_to_station.items()}
    
    print("--- Initial data loading and preprocessing complete. ---\n")
    return df_discharge, df_contrib, df_coords, vcode_to_station, station_to_vcode

def add_temporal_features(df):
    """Adds cyclical day-of-year features."""
    print("Adding cyclical temporal features (sin/cos)...")
    df_temp = df.copy()
    df_temp.index = pd.to_datetime(df_temp.index)
    day_of_year = df_temp.index.dayofyear
    df_temp['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 366.0)
    df_temp['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 366.0)
    return df_temp

def build_distance_matrix(df_coords, discharge_cols):
    """Builds a distance matrix between all stations."""
    print("Building distance matrix...")
    stations = discharge_cols
    distance_matrix = pd.DataFrame(np.inf, index=stations, columns=stations)
    
    for station_i in stations:
        for station_j in stations:
            if station_i == station_j:
                distance_matrix.loc[station_i, station_j] = 0
                continue
            if station_i in df_coords.index and station_j in df_coords.index:
                lat1, lon1 = df_coords.loc[station_i, ['Latitude', 'Longitude']]
                lat2, lon2 = df_coords.loc[station_j, ['Latitude', 'Longitude']]
                dist = geodesic((lat1, lon1), (lat2, lon2)).km
                distance_matrix.loc[station_i, station_j] = dist
    return distance_matrix

def build_connectivity_matrix(df_contrib, discharge_cols, station_name_to_vcode):
    """Builds a directed connectivity matrix."""
    print("Building connectivity matrix...")
    stations = discharge_cols
    connectivity_matrix = pd.DataFrame(0.0, index=stations, columns=stations)
    
    if df_contrib is None or df_contrib.empty:
        return connectivity_matrix
        
    for _, row in df_contrib.iterrows():
        station = row.get('station')
        contributor = row.get('contributor')
        if station in connectivity_matrix.index and contributor in connectivity_matrix.columns:
            connectivity_matrix.loc[station, contributor] = 1.0
            
    return connectivity_matrix


# In utils.py

# In utils.py

import math # Add this import at the top of utils.py

def create_contiguous_segment_gaps(df_data, discharge_target_columns, gap_lengths, target_overall_gap_percentage=10.0, random_seed=42):
    """
    Introduces continuous gaps (NaNs) into discharge columns so that the
    OVERALL percentage of missing data across ALL target columns is approximately
    the target percentage, for each specified gap length. Gaps are distributed
    among columns.

    Args:
        df_data (pd.DataFrame): The input DataFrame.
        discharge_target_columns (list): Discharge column names PRESENT in df_data.
        gap_lengths (list): List of gap lengths in days (e.g., [3, 7, 30]).
        target_overall_gap_percentage (float): The desired overall percentage
                                              of data points across all target
                                              columns to turn into NaNs (0-100).
        random_seed (int): Seed for reproducibility.

    Returns:
        dict: A dictionary mapping gap lengths to results ('gapped_data', etc.).
    """
    np.random.seed(random_seed)
    results = {}

    # Ensure target percentage is valid
    if not 0 < target_overall_gap_percentage <= 100:
        print("Error: target_overall_gap_percentage must be between 0 (exclusive) and 100 (inclusive).")
        return results

    # Identify actual discharge columns *that are present in the input df_data*
    # And exclude any potential feature columns passed inadvertently
    valid_discharge_columns = [
        col for col in discharge_target_columns
        if col in df_data.columns and not (
            col.startswith('day_of_year_') or
            col.startswith('month_') or
            col.startswith('week_of_year_')
        )
    ]

    if not valid_discharge_columns:
        print("Warning: No valid discharge target columns found in the provided DataFrame to introduce gaps. Returning empty results.")
        return results

    data_length = len(df_data)
    num_target_columns = len(valid_discharge_columns)
    total_data_points = data_length * num_target_columns

    if data_length == 0 or num_target_columns == 0:
        print("Warning: Data length or number of target columns is zero. Cannot create gaps.")
        return results

    print(f"Total data points across {num_target_columns} valid columns: {total_data_points}")

    for gap_length in gap_lengths:
        if gap_length <= 0:
            print(f"Warning: Skipping gap_length {gap_length} as it's not positive.")
            continue

        print(f"\nCreating gaps for length {gap_length} aiming for ~{target_overall_gap_percentage:.1f}% OVERALL missing data.")
        df_gapped_for_length = df_data.copy()
        true_values_for_length = {}
        mask_for_length = {}

        if gap_length >= data_length:
             print(f"Warning: Gap length ({gap_length}) >= data length ({data_length}). Cannot create gaps effectively with this strategy. Skipping gap length.")
             results[gap_length] = { # Add empty results for consistency
                 'gapped_data': df_gapped_for_length, # Return unchanged data
                 'true_values': {},
                 'mask': {}
             }
             continue # Skip to the next gap length

        # --- Calculate TOTAL segments needed ---
        total_points_to_gap_overall = int(total_data_points * (target_overall_gap_percentage / 100.0))
        # Ensure gap_length is not zero before division
        total_segments_needed = total_points_to_gap_overall // gap_length if gap_length > 0 else 0
        if total_segments_needed == 0 and total_points_to_gap_overall > 0 and gap_length > 0:
             total_segments_needed = 1 # Ensure at least one segment if percentage > 0

        print(f"  Target total points to gap: {total_points_to_gap_overall}")
        print(f"  Target total segments needed ({gap_length}-day): {total_segments_needed}")
        # --- END Calculation ---

        if total_segments_needed == 0:
            print("  Calculated 0 total segments needed. No gaps will be created for this length.")
            results[gap_length] = {
                 'gapped_data': df_gapped_for_length,
                 'true_values': {col: np.array([]) for col in valid_discharge_columns},
                 'mask': {col: np.zeros(data_length, dtype=bool) for col in valid_discharge_columns}
            }
            continue # Skip to the next gap length

        # --- Distribute segments somewhat evenly ---
        base_segments_per_column = total_segments_needed // num_target_columns
        extra_segments = total_segments_needed % num_target_columns
        segments_per_column_list = [base_segments_per_column + 1] * extra_segments + \
                                   [base_segments_per_column] * (num_target_columns - extra_segments)
        np.random.shuffle(segments_per_column_list) # Randomize assignment

        column_to_segments = dict(zip(valid_discharge_columns, segments_per_column_list))
        # --- End Distribution ---

        total_actual_points_gapped = 0
        all_gaps_created_this_length = [] # Track all gaps [(start, end, col), ...]

        for target_column in valid_discharge_columns:
            num_intervals_to_create = column_to_segments[target_column]

            # Limit intervals if they physically can't fit without overlap in this column
            max_possible_intervals = data_length // gap_length if gap_length > 0 else 0
            num_intervals_to_create = min(num_intervals_to_create, max_possible_intervals)

            if num_intervals_to_create == 0:
                true_values_for_length[target_column] = np.array([])
                mask_for_length[target_column] = np.zeros(data_length, dtype=bool)
                continue

            # Find non-overlapping gaps for this specific column
            possible_start_indices = np.arange(data_length - gap_length + 1)
            np.random.shuffle(possible_start_indices)
            gaps_to_apply_this_column = []

            for start_candidate in possible_start_indices:
                end_candidate = start_candidate + gap_length
                is_overlapping = False
                # Check overlap ONLY within gaps already selected FOR THIS COLUMN
                for existing_start, existing_end in gaps_to_apply_this_column:
                    if not (end_candidate <= existing_start or start_candidate >= existing_end):
                        is_overlapping = True
                        break
                if not is_overlapping:
                    gaps_to_apply_this_column.append((start_candidate, end_candidate))
                    all_gaps_created_this_length.append((start_candidate, end_candidate, target_column)) # Add to overall list
                    if len(gaps_to_apply_this_column) == num_intervals_to_create:
                        break

            if len(gaps_to_apply_this_column) < num_intervals_to_create:
                print(f"Warning: Could only create {len(gaps_to_apply_this_column)} non-overlapping segments for column {target_column} (target was {num_intervals_to_create}).")

            # Apply gaps for this column
            col_true_values = np.array([])
            col_mask = np.zeros(data_length, dtype=bool)
            points_gapped_this_column = 0
            for start_idx, end_idx in gaps_to_apply_this_column:
                # Ensure indices are within bounds
                if start_idx < 0 or end_idx > data_length: continue

                true_vals_segment = df_data.loc[df_data.index[start_idx:end_idx], target_column].copy().values
                col_true_values = np.concatenate([col_true_values, true_vals_segment])
                col_mask[start_idx:end_idx] = True
                df_gapped_for_length.loc[df_gapped_for_length.index[start_idx:end_idx], target_column] = np.nan
                points_gapped_this_column += (end_idx - start_idx)

            true_values_for_length[target_column] = col_true_values
            mask_for_length[target_column] = col_mask
            total_actual_points_gapped += points_gapped_this_column

        actual_overall_percentage = (total_actual_points_gapped / total_data_points) * 100 if total_data_points > 0 else 0
        print(f"  Overall for {gap_length}-day gaps: Created {len(all_gaps_created_this_length)} segments, gapping {total_actual_points_gapped} points across {num_target_columns} columns.")
        print(f"  Actual overall missing %: {actual_overall_percentage:.2f}%")

        results[gap_length] = {
            'gapped_data': df_gapped_for_length,
            'true_values': true_values_for_length,
            'mask': mask_for_length
        }
    return results

def create_single_point_gaps(df_data, discharge_target_columns, num_gaps_per_column, random_seed=42):
    """
    Introduces single random missing data points (NaNs) into specified discharge columns.

    Args:
        df_data (pd.DataFrame): The input DataFrame (assumed to be complete for creating gaps).
        discharge_target_columns (list): A list of column names representing the true discharge data.
        num_gaps_per_column (dict): A dictionary where keys are gap counts and values are dictionaries
                                    containing 'gapped_data', 'true_values', and 'mask' for that gap count.
                                    'true_values' and 'mask' will be dictionaries mapping column names to arrays.
        random_seed (int): Seed for reproducibility.

    Returns:
        dict: A dictionary where keys are gap lengths and values are dictionaries
              containing 'gapped_data', 'true_values', and 'mask' for that gap count.
              'true_values' and 'mask' will be dictionaries mapping column names to arrays.
    """
    np.random.seed(random_seed)
    results = {}

    discharge_target_columns = [col for col in df_data.columns if not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))]

    if not discharge_target_columns:
        print("Warning: No discharge target columns provided to introduce gaps. Returning empty results.")
        return results

    for num_gaps_key, num_gaps_value in num_gaps_per_column.items():
        df_gapped_for_length = df_data.copy()
        true_values_for_length = {}
        mask_for_length = {}

        for target_column in discharge_target_columns:
            data_length = len(df_gapped_for_length)
            if data_length == 0: continue

            # Calculate number of gaps as a percentage of data length
            n_gaps = int(data_length * (num_gaps_value / 365.0)) # Assuming num_gaps_value is in days for comparison
            if n_gaps == 0 and num_gaps_value > 0: n_gaps = 1 # Ensure at least one gap if requested

            if n_gaps >= data_length:
                print(f"Warning: Number of single point gaps ({n_gaps}) is >= data length ({data_length}) for column {target_column}. Setting entire column to NaN.")
                true_values_for_length[target_column] = df_data[target_column].copy().values
                df_gapped_for_length[target_column] = np.nan
                mask_for_length[target_column] = np.ones(data_length, dtype=bool)
            else:
                # Select random indices for gaps
                gap_indices = np.random.choice(data_length, n_gaps, replace=False)

                # Store true values before masking
                true_values_for_length[target_column] = df_data.loc[df_data.index[gap_indices], target_column].copy().values

                # Create mask for the specific gaps
                col_mask = np.zeros(data_length, dtype=bool)
                col_mask[gap_indices] = True
                mask_for_length[target_column] = col_mask

                # Introduce NaN values
                df_gapped_for_length.loc[df_gapped_for_length.index[gap_indices], target_column] = np.nan
        
        results[num_gaps_key] = {
            'gapped_data': df_gapped_for_length,
            'true_values': true_values_for_length,
            'mask': mask_for_length
        }
    return results

def evaluate_metrics(y_true, y_pred):
    """
    Evaluates imputation performance using RMSE, MAE, and R2.
    
    Args:
        y_true (np.array): Original true values.
        y_pred (np.array): Imputed/predicted values.
        
    Returns:
        dict: Dictionary of calculated metrics.
    """
    # Ensure only non-NaN values are used for metric calculation
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[valid_indices]
    y_pred_clean = y_pred[valid_indices]

    if len(y_true_clean) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}

    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    ss_total = np.sum((y_true_clean - np.mean(y_true_clean))**2)
    ss_residual = np.sum((y_true_clean - y_pred_clean)**2)
    
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan

    # Calculate Nash-Sutcliffe Efficiency (NSE)
    # NSE = 1 - (SS_res / SS_tot)
    nse = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'NSE': nse}

def plot_results(y_true, y_pred, gap_length, plot_dir="plots"):
    """
    Plots original, gapped, and imputed data for visual comparison.
    
    Args:
        y_true (pd.Series or np.array): The true values.
        y_pred (pd.Series or np.array): The imputed values.
        gap_length (int): The length of the gap (for plot title/filename).
        plot_dir (str): Directory to save the plots.
    """
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True Values', alpha=0.7)
    plt.plot(y_pred, label='Imputed Values', alpha=0.7, linestyle='--')
    plt.title(f'Imputation for {gap_length}-day Gap')
    plt.xlabel('Time')
    plt.ylabel('Discharge')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'imputation_gap_{gap_length}.png'))
    plt.close()
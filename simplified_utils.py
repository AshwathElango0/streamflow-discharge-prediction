# simplified_utils.py - Essential utility functions for burst imputation
import pandas as pd
import numpy as np
import re
from geopy.distance import geodesic

def clean_station_name(name):
    """Standardizes station names for consistent matching."""
    name_correction_map = {
        "parmanpur": "paramanpur",
        "bamnidih": "bamnidhi",
    }
    
    name = str(name).strip().lower()
    name = re.sub(r"[''\"`]", "", name)
    name = re.sub(r"\s+", "", name)
    
    return name_correction_map.get(name, name)

def parse_lat_lon(coord_str, is_latitude=True):
    """Parses coordinate strings into decimal degrees."""
    if isinstance(coord_str, (int, float)):
        val = float(coord_str)
        if (is_latitude and not -90 <= val <= 90) or (not is_latitude and not -180 <= val <= 180):
            raise ValueError(f"Coordinate {val} is out of bounds.")
        return val

    s = str(coord_str).strip()
    s_spaced = re.sub(r"[o°'\"`'']", " ", s)
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
    """Load and preprocess all data sources with consistent station matching."""
    print("--- Loading and Preprocessing Data ---")
    
    # Load discharge data
    try:
        df_discharge = pd.read_csv(discharge_path, index_col=0, parse_dates=True, dayfirst=True)
        df_discharge.index.name = 'date'
        
        # Identify discharge vs temporal columns
        all_cols = df_discharge.columns.tolist()
        discharge_cols = [col for col in all_cols if not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))]
        temporal_cols = [col for col in all_cols if col not in discharge_cols]

        # Clean discharge column names
        df_discharge_stations = df_discharge[discharge_cols].copy()
        df_discharge_stations.columns = df_discharge_stations.columns.map(clean_station_name)
        
        # Recombine with temporal features
        if temporal_cols:
            df_temporal = df_discharge[temporal_cols].copy()
            df_discharge = pd.concat([df_discharge_stations, df_temporal], axis=1)
        else:
            df_discharge = df_discharge_stations

        # Convert to numeric
        for col in discharge_cols:
            cleaned_name = clean_station_name(col)
            if cleaned_name in df_discharge.columns:
                df_discharge[cleaned_name] = pd.to_numeric(df_discharge[cleaned_name], errors='coerce')

    except Exception as e:
        print(f"ERROR: Could not load discharge data: {e}")
        return None, None, None, None, None

    # Load coordinate data
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
        print(f"ERROR: Could not load coordinate data: {e}")
        return None, None, None, None, None

    # Find canonical stations (present in both discharge and coordinate data)
    discharge_stations = set([col for col in df_discharge.columns 
                            if not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))])
    coords_stations = set(df_coords.index)
    canonical_stations = sorted(list(discharge_stations.intersection(coords_stations)))

    print(f"Using {len(canonical_stations)} stations found in both files.")

    # Filter data to canonical stations
    cols_to_keep = canonical_stations + temporal_cols
    df_discharge = df_discharge[cols_to_keep]
    df_coords = df_coords.loc[canonical_stations]

    # Parse coordinates
    df_coords['Latitude'] = df_coords['Latitude'].apply(lambda x: parse_lat_lon(x, is_latitude=True))
    df_coords['Longitude'] = df_coords['Longitude'].apply(lambda x: parse_lat_lon(x, is_latitude=False))

    # Load contributor data (optional)
    df_contrib = None
    if contrib_path:
        try:
            df_contrib_matrix = pd.read_csv(contrib_path, index_col=0)
            v_code_to_name = {v_code: clean_station_name(name) for v_code, name in df_contrib_matrix['Name of site'].items()}
            matrix_v_codes = [col for col in df_contrib_matrix.columns if col.startswith('V')]
            
            contrib_pairs = []
            for station_v_code, row in df_contrib_matrix.iterrows():
                station_name = v_code_to_name.get(station_v_code)
                if not station_name: continue
                for contributor_v_code in matrix_v_codes:
                    if row[contributor_v_code] == 1:
                        contributor_name = v_code_to_name.get(contributor_v_code)
                        if contributor_name:
                            contrib_pairs.append({'station': station_name, 'contributor': contributor_name})
            
            if contrib_pairs:
                df_contrib = pd.DataFrame(contrib_pairs)
                df_contrib = df_contrib[df_contrib['station'].isin(canonical_stations) & 
                                      df_contrib['contributor'].isin(canonical_stations)]
                print(f"Loaded contributor matrix with {len(df_contrib)} relationships.")
            else:
                df_contrib = None
        except Exception as e:
            print(f"Warning: Could not load contributor data: {e}")
            df_contrib = None
            
    # Create station mappings
    vcode_to_station, station_to_vcode = {}, {}
    if 'v_code' in df_coords.columns:
        coords_unique = df_coords[~df_coords.index.duplicated(keep='first')]
        vcode_to_station = coords_unique['v_code'].to_dict()
        station_to_vcode = {v: k for k, v in vcode_to_station.items()}
    
    print("Data loading complete.")
    return df_discharge, df_contrib, df_coords, vcode_to_station, station_to_vcode

def add_temporal_features(df):
    """Add cyclical temporal features (sin/cos of day of year)."""
    print("Adding temporal features...")
    df_temp = df.copy()
    df_temp.index = pd.to_datetime(df_temp.index)
    day_of_year = df_temp.index.dayofyear
    df_temp['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 366.0)
    df_temp['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 366.0)
    return df_temp

def build_distance_matrix(df_coords, discharge_cols):
    """Build distance matrix between all stations."""
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
    """Build directed connectivity matrix from contributor data."""
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

def create_contiguous_segment_gaps(df_data, discharge_cols, gap_lengths, random_seed=42, num_intervals_per_column=5):
    """Create contiguous gaps in discharge data for evaluation."""
    np.random.seed(random_seed)
    results = {}

    for gap_length in gap_lengths:
        if gap_length <= 0:
            continue

        df_gapped = df_data.copy()
        true_values = {}
        mask = {}

        for target_column in discharge_cols:
            data_length = len(df_gapped)
            
            if gap_length >= data_length:
                true_values[target_column] = df_data[target_column].copy().values
                df_gapped[target_column] = np.nan
                mask[target_column] = np.ones(data_length, dtype=bool)
            else:
                num_intervals = min(num_intervals_per_column, data_length // gap_length)
                possible_starts = np.arange(data_length - gap_length + 1)
                np.random.shuffle(possible_starts)

                gaps_to_apply = []
                for start_candidate in possible_starts:
                    end_candidate = start_candidate + gap_length
                    
                    # Check for overlap
                    is_overlapping = False
                    for existing_start, existing_end in gaps_to_apply:
                        if not (end_candidate <= existing_start or start_candidate >= existing_end):
                            is_overlapping = True
                            break
                    
                    if not is_overlapping:
                        gaps_to_apply.append((start_candidate, end_candidate))
                        if len(gaps_to_apply) == num_intervals:
                            break
                
                # Apply gaps
                col_true_values = np.array([])
                col_mask = np.zeros(data_length, dtype=bool)

                for start_idx, end_idx in gaps_to_apply:
                    col_true_values = np.concatenate([col_true_values, df_data.loc[df_data.index[start_idx:end_idx], target_column].copy().values])
                    col_mask[start_idx:end_idx] = True
                    df_gapped.loc[df_gapped.index[start_idx:end_idx], target_column] = np.nan
                
                true_values[target_column] = col_true_values
                mask[target_column] = col_mask
        
        results[gap_length] = {
            'gapped_data': df_gapped,
            'true_values': true_values, 
            'mask': mask
        }
    return results

def create_single_point_gaps(df_data, discharge_cols, num_gaps_per_column, random_seed=42):
    """Create single point gaps in discharge data for evaluation."""
    np.random.seed(random_seed)
    results = {}

    for num_gaps_key, num_gaps_value in num_gaps_per_column.items():
        df_gapped = df_data.copy()
        true_values = {}
        mask = {}

        for target_column in discharge_cols:
            data_length = len(df_gapped)
            if data_length == 0: continue

            # Calculate number of gaps as percentage of data length
            n_gaps = int(data_length * (num_gaps_value / 365.0))
            if n_gaps == 0 and num_gaps_value > 0: n_gaps = 1

            if n_gaps >= data_length:
                true_values[target_column] = df_data[target_column].copy().values
                df_gapped[target_column] = np.nan
                mask[target_column] = np.ones(data_length, dtype=bool)
            else:
                # Select random indices for gaps
                gap_indices = np.random.choice(data_length, n_gaps, replace=False)
                
                # Store true values and create mask
                true_values[target_column] = df_data.loc[df_data.index[gap_indices], target_column].copy().values
                col_mask = np.zeros(data_length, dtype=bool)
                col_mask[gap_indices] = True
                mask[target_column] = col_mask

                # Introduce NaN values
                df_gapped.loc[df_gapped.index[gap_indices], target_column] = np.nan
        
        results[num_gaps_key] = {
            'gapped_data': df_gapped,
            'true_values': true_values,
            'mask': mask
        }
    return results

def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics (RMSE, MAE, R2, NSE)."""
    # Filter out NaN values
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[valid_indices]
    y_pred_clean = y_pred[valid_indices]

    if len(y_true_clean) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan}

    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    ss_total = np.sum((y_true_clean - np.mean(y_true_clean))**2)
    ss_residual = np.sum((y_true_clean - y_pred_clean)**2)
    
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan
    nse = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'NSE': nse}

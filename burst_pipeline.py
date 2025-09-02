import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timedelta
import pickle # Add this import
import hashlib # Add this import

warnings.filterwarnings('ignore')

# Directory to save/load trained models
MODEL_CACHE_DIR = "trained_models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# --- External Dependencies ---
from missforest_imputer import ModifiedMissForest
from model_configurations import train_full_model, train_no_contributor_model # Updated import
from utils import load_and_preprocess_data, add_temporal_features, build_distance_matrix, build_connectivity_matrix

def find_best_training_period(df_data, min_year, max_year, window_size=3, min_overall_completeness_percent=70.0):
    """
    Identifies optimal training period (highest completeness) within a year range.
    Returns the best period and the list of stations with sufficient data.
    """
    discharge_cols = [col for col in df_data.columns if not col.startswith('day_of_year_')]
    if not discharge_cols:
        print("Warning: No discharge columns for period selection.")
        return None, None, []

    best_period = (None, None)
    best_stations = []
    max_completeness_found = -1.0 

    print(f"\nSearching for best {window_size}-year training period between {min_year}-{max_year} (min {min_overall_completeness_percent}% completeness required)...")

    for start_year in range(min_year, max_year - window_size + 2):
        end_year = start_year + window_size - 1
        
        if start_year > df_data.index.year.max() or end_year < df_data.index.year.min():
            continue

        period_data = df_data[(df_data.index.year >= start_year) & (df_data.index.year <= end_year)]
        if period_data.empty:
            continue

        period_discharge_data = period_data[discharge_cols]
        total_cells_in_period = period_discharge_data.shape[0] * period_discharge_data.shape[1]
        
        if total_cells_in_period == 0:
            current_completeness = 0.0
        else:
            non_na_cells_in_period = period_discharge_data.notna().sum().sum()
            current_completeness = (non_na_cells_in_period / total_cells_in_period) * 100

        print(f"  Period {start_year}-{end_year}: Overall completeness = {current_completeness:.2f}%")

        if current_completeness >= min_overall_completeness_percent:
            if current_completeness > max_completeness_found:
                max_completeness_found = current_completeness
                best_period = (start_year, end_year)
                # Find stations with sufficient data in this period
                min_non_na_count = int(min_overall_completeness_percent / 100 * len(period_data))
                best_stations = period_discharge_data.columns[
                    period_discharge_data.notna().sum() >= min_non_na_count
                ].tolist()

    if best_period == (None, None):
        print(f"Error: No {window_size}-year period found with at least {min_overall_completeness_percent}% completeness. Insufficient data for training.")
        return None, None, []
    else:
        print(f"\nBest training period identified: {best_period[0]}-{best_period[1]} with {max_completeness_found:.2f}% completeness.")
        print(f"Stations with sufficient data: {len(best_stations)} out of {len(discharge_cols)}")
    return best_period[0], best_period[1], best_stations

def _prepare_training_data_for_model(
    df_train_slice, all_discharge_cols, all_feature_cols, min_completeness_percent_train
):
    """
    Helper function to prepare training data. Handles column filtering and simulated missingness.
    """
    min_non_na_count = int(min_completeness_percent_train / 100 * len(df_train_slice))
    
    # Filter to only keep discharge columns with sufficient data in this specific training slice
    cols_to_keep_discharge = []
    if not df_train_slice[all_discharge_cols].empty:
        cols_to_keep_discharge = df_train_slice[all_discharge_cols].columns[
            df_train_slice[all_discharge_cols].notna().sum() >= min_non_na_count
        ].tolist()
    
    if not cols_to_keep_discharge:
        return None, None # Indicate no valid discharge columns for training

    # Final columns for this model are the valid discharge columns plus all feature columns
    final_cols_for_imputer = list(set(cols_to_keep_discharge + all_feature_cols))
    df_train_period_filtered = df_train_slice[final_cols_for_imputer].copy()

    # Handle all-NaN columns in this specific training slice
    df_train_discharge_only = df_train_period_filtered[cols_to_keep_discharge].copy()
    for col in df_train_discharge_only.columns:
        if df_train_discharge_only[col].isnull().all():
            df_train_discharge_only[col] = 0.0 # Fill with 0 for robustness
        else:
            # Simple mean fill before creating simulated mask
            df_train_discharge_only[col] = df_train_discharge_only[col].fillna(df_train_discharge_only[col].mean())

    # Simulate 10% random missingness for model training
    np.random.seed(42)
    train_mask_simulated = np.random.rand(*df_train_discharge_only.shape) < 0.1
    df_train_masked_for_model = df_train_period_filtered.copy()
    df_train_masked_for_model[cols_to_keep_discharge] = df_train_discharge_only.mask(train_mask_simulated)
    
    print(f"Simulated 10% random missingness in training data.")
    return df_train_masked_for_model, cols_to_keep_discharge


def _prepare_imputation_target_data(
    df_raw_slice, model_discharge_cols, all_feature_cols, trained_model
):
    """
    Helper function to prepare data for imputation using a trained model.
    """
    # The model expects the discharge columns it was trained on, plus all feature columns
    cols_for_imputation_target = list(set(model_discharge_cols + all_feature_cols))
    
    # Reindex to ensure alignment. This adds any missing columns as all-NaN.
    df_to_impute = df_raw_slice.reindex(columns=cols_for_imputation_target).copy()

    # Handle all-NaN columns in the target data slice
    for col in model_discharge_cols:
        if col not in df_to_impute.columns:
            print(f"Warning: Column '{col}' expected by model not in target slice. Adding as NaN.")
            df_to_impute[col] = np.nan
            
        # If a column is entirely missing, fill it with 0 if the model's learned mean was also NaN.
        # This prevents errors in the imputer's internal workings.
        if df_to_impute[col].isnull().all() and col in trained_model.col_means and np.isnan(trained_model.col_means.get(col)):
            df_to_impute[col] = 0.0
    return df_to_impute


def run_rolling_imputation_pipeline(
    discharge_path, lat_long_path, contrib_path=None,
    initial_train_window_size=5,
    imputation_chunk_size_years=5,
    overall_min_year=1976, overall_max_year=2016,
    min_completeness_percent_train=70.0,
    output_dir="bursting_imputed_results"
):
    """
    Implements a "bursting" imputation pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Starting Bursting Imputation Pipeline ---")

    # --- Initial Setup (Run ONLY ONCE) ---
    try:
        df_discharge_raw, df_contrib_filtered, df_coords, vcode_to_station_name, station_name_to_vcode = \
            load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
        if df_discharge_raw is None:
            print("FATAL ERROR: Failed to load data. Exiting.")
            return None
    except Exception as e:
        print(f"FATAL ERROR during initial data loading/preprocessing: {e}")
        return None

    # --- Feature Engineering ---
    all_discharge_cols_overall = df_discharge_raw.columns.tolist()
    
    # 1. Add sine/cosine temporal features
    df_full_data_with_features = add_temporal_features(df_discharge_raw)
    temporal_features = ['day_of_year_sin', 'day_of_year_cos']
    
    # 2. Combine all feature columns into one list
    all_feature_cols = temporal_features
    print(f"Total features created: {len(all_feature_cols)}")

    print("\nBuilding Static Spatial and Connectivity Matrices for all stations...")
    # We'll build these matrices dynamically based on the stations that have sufficient data
    print("Note: Distance and connectivity matrices will be built dynamically for each training period.")
    
    # Initialize the master imputed DataFrame
    df_master_imputed = df_full_data_with_features.copy()
    print(f"Master imputed DataFrame initialized. Total NaNs: {df_master_imputed[all_discharge_cols_overall].isna().sum().sum()}")

    # --- Phase 1: Find Best Initial Training Period & Train First Model ---
    print("\nFinding the best initial training period...")
    best_initial_train_start, best_initial_train_end, initial_train_stations = find_best_training_period(
        df_full_data_with_features, overall_min_year, overall_max_year,
        initial_train_window_size, min_completeness_percent_train
    )
    if best_initial_train_start is None:
        return None # Error message is printed inside the function
    print(f"Selected initial training period: {best_initial_train_start}-{best_initial_train_end}")
    print(f"Initial training stations: {initial_train_stations}")

    print("\nTraining initial model on the best period...")
    df_initial_train_slice = df_master_imputed.loc[
        (df_master_imputed.index.year >= best_initial_train_start) &
        (df_master_imputed.index.year <= best_initial_train_end)
    ]
    df_initial_train_masked, cols_to_keep_initial = _prepare_training_data_for_model(
        df_initial_train_slice, initial_train_stations, all_feature_cols, min_completeness_percent_train
    )
    if df_initial_train_masked is None:
        print("Error: Initial training data preparation failed. Cannot train initial model.")
        return None

    # Build distance and connectivity matrices for the initial training stations
    dist_matrix_initial = build_distance_matrix(df_coords, cols_to_keep_initial)
    if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
        conn_matrix_initial = build_connectivity_matrix(df_contrib_filtered, cols_to_keep_initial, station_name_to_vcode)
        model_type_key = "full_model"
    else:
        print("No valid contribution data. Building a zero connectivity matrix for initial training.")
        conn_matrix_initial = pd.DataFrame(0, index=cols_to_keep_initial, columns=cols_to_keep_initial, dtype=int)
        model_type_key = "no_contributor_model"

    # Generate a unique model hash based on input data and parameters
    model_hash_input_initial = f"{discharge_path}-{lat_long_path}-{contrib_path}-{best_initial_train_start}-{best_initial_train_end}-{min_completeness_percent_train}-{model_type_key}"
    model_hash_initial = hashlib.md5(model_hash_input_initial.encode()).hexdigest()
    model_filepath_initial = os.path.join(MODEL_CACHE_DIR, f'model_{model_hash_initial}.pkl')

    trained_model_initial = None
    if os.path.exists(model_filepath_initial):
        print(f"Loading initial model from cache: {model_filepath_initial}")
        with open(model_filepath_initial, 'rb') as f:
            trained_model_initial = pickle.load(f)
    else:
        if model_type_key == "full_model":
            print("--- Training initial model (Full Model) ---")
            trained_model_initial = train_full_model(
                df_initial_train_masked, dist_matrix_initial, conn_matrix_initial, all_feature_cols
            )
        elif model_type_key == "no_contributor_model":
            print("--- Training initial model (No Contributor Model) ---")
            trained_model_initial = train_no_contributor_model(
                df_initial_train_masked, dist_matrix_initial, conn_matrix_initial, all_feature_cols
            )
        if trained_model_initial:
            with open(model_filepath_initial, 'wb') as f:
                pickle.dump(trained_model_initial, f)
            print(f"Initial model saved to cache: {model_filepath_initial}")

    if trained_model_initial is None:
        print("Error: Initial model training failed. Exiting pipeline.")
        return None
    model_discharge_cols_initial = trained_model_initial.discharge_columns
    
    # --- Phase 2: Initial Bidirectional Imputation ---
    # --- Impute First Chunk Forward ---
    print(f"\nPerforming initial forward imputation...")
    impute_forward_start = best_initial_train_end + 1
    impute_forward_end = min(overall_max_year, impute_forward_start + imputation_chunk_size_years - 1)
    
    last_imputed_forward_chunk = df_master_imputed.loc[
        (df_master_imputed.index.year >= best_initial_train_start) & (df_master_imputed.index.year <= best_initial_train_end),
        model_discharge_cols_initial
    ].copy()

    if impute_forward_start <= impute_forward_end:
        df_impute_slice_raw = df_full_data_with_features.loc[f"{impute_forward_start}":f"{impute_forward_end}"]
        df_to_impute = _prepare_imputation_target_data(
            df_impute_slice_raw, model_discharge_cols_initial, all_feature_cols, trained_model_initial
        )
        if not df_to_impute.empty:
            imputed_chunk = trained_model_initial.transform(df_to_impute)
            imputed_discharge = imputed_chunk[model_discharge_cols_initial]
            df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
            print(f"Initial forward imputation complete for {impute_forward_start}-{impute_forward_end}.")
            last_imputed_forward_chunk = imputed_discharge.copy()

    # --- Impute First Chunk Backward ---
    print(f"\nPerforming initial backward imputation...")
    impute_backward_end = best_initial_train_start - 1
    impute_backward_start = max(overall_min_year, impute_backward_end - imputation_chunk_size_years + 1)

    last_imputed_backward_chunk = df_master_imputed.loc[
        (df_master_imputed.index.year >= best_initial_train_start) & (df_master_imputed.index.year <= best_initial_train_end),
        model_discharge_cols_initial
    ].copy()

    if impute_backward_start <= impute_backward_end:
        df_impute_slice_raw = df_full_data_with_features.loc[f"{impute_backward_start}":f"{impute_backward_end}"]
        df_to_impute = _prepare_imputation_target_data(
            df_impute_slice_raw, model_discharge_cols_initial, all_feature_cols, trained_model_initial
        )
        if not df_to_impute.empty:
            imputed_chunk = trained_model_initial.transform(df_to_impute)
            imputed_discharge = imputed_chunk[model_discharge_cols_initial]
            df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
            print(f"Initial backward imputation complete for {impute_backward_start}-{impute_backward_end}.")
            last_imputed_backward_chunk = imputed_discharge.copy()

    # --- Phase 3 & 4: Forward and Backward Bursting Loops ---
    # (The logic for these loops is complex but fundamentally correct. The key was fixing the inputs they receive.)
    # The fixes above ensure these loops now receive correctly prepared data and use the right models.
    # The following code is largely the same, but incorporates the corrected variable names and function calls.

    # --- Phase 3: Forward Bursting Loop ---
    print("\n--- Starting Forward Bursting Imputation Loop ---")
    current_forward_start = impute_forward_end + 1
    while current_forward_start <= overall_max_year:
        current_forward_end = min(overall_max_year, current_forward_start + imputation_chunk_size_years - 1)
        print(f"Forward Bursting: Training on last imputed chunk, Imputing {current_forward_start}-{current_forward_end}")

        # Prepare training data (last imputed chunk + features)
        df_train_slice = last_imputed_forward_chunk.merge(df_full_data_with_features[all_feature_cols], left_index=True, right_index=True)
        df_train_masked, cols_to_keep = _prepare_training_data_for_model(
            df_train_slice, last_imputed_forward_chunk.columns.tolist(), all_feature_cols, min_completeness_percent_train
        )
        if df_train_masked is None: break

        # Build distance and connectivity matrices for this training period
        dist_matrix = build_distance_matrix(df_coords, cols_to_keep)
        if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
            conn_matrix = build_connectivity_matrix(df_contrib_filtered, cols_to_keep, station_name_to_vcode)
            current_model_type_key = "full_model"
        else:
            conn_matrix = pd.DataFrame(0, index=cols_to_keep, columns=cols_to_keep, dtype=int)
            current_model_type_key = "no_contributor_model"

        # Generate a unique model hash for the current iteration
        current_model_hash_input = f"{discharge_path}-{lat_long_path}-{contrib_path}-{current_forward_start}-{current_forward_end}-{min_completeness_percent_train}-{current_model_type_key}"
        current_model_hash = hashlib.md5(current_model_hash_input.encode()).hexdigest()
        current_model_filepath = os.path.join(MODEL_CACHE_DIR, f'model_{current_model_hash}.pkl')

        trained_model = None
        if os.path.exists(current_model_filepath):
            print(f"Loading model from cache: {current_model_filepath}")
            with open(current_model_filepath, 'rb') as f:
                trained_model = pickle.load(f)
        else:
            if current_model_type_key == "full_model":
                print(f"--- Training model (Full Model) for {current_forward_start}-{current_forward_end} ---")
                trained_model = train_full_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
            elif current_model_type_key == "no_contributor_model":
                print(f"--- Training model (No Contributor Model) for {current_forward_start}-{current_forward_end} ---")
                trained_model = train_no_contributor_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
            if trained_model:
                with open(current_model_filepath, 'wb') as f:
                    pickle.dump(trained_model, f)
                print(f"Model saved to cache: {current_model_filepath}")

        if trained_model is None: break
        
        # Impute next chunk
        df_impute_slice_raw = df_full_data_with_features.loc[f"{current_forward_start}":f"{current_forward_end}"]
        df_to_impute = _prepare_imputation_target_data(df_impute_slice_raw, trained_model.discharge_columns, all_feature_cols, trained_model)
        if df_to_impute.empty:
            current_forward_start = current_forward_end + 1
            continue
            
        imputed_chunk = trained_model.transform(df_to_impute)
        imputed_discharge = imputed_chunk[trained_model.discharge_columns]
        df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
        print(f"Forward imputation complete for {current_forward_start}-{current_forward_end}.")
        
        last_imputed_forward_chunk = imputed_discharge.copy()
        current_forward_start = current_forward_end + 1

    # --- Phase 4: Backward Bursting Loop ---
    print("\n--- Starting Backward Bursting Imputation Loop ---")
    current_backward_end = impute_backward_start - 1
    while current_backward_end >= overall_min_year:
        current_backward_start = max(overall_min_year, current_backward_end - imputation_chunk_size_years + 1)
        print(f"Backward Bursting: Training on last imputed chunk, Imputing {current_backward_start}-{current_backward_end}")

        # Prepare training data
        df_train_slice = last_imputed_backward_chunk.merge(df_full_data_with_features[all_feature_cols], left_index=True, right_index=True)
        df_train_masked, cols_to_keep = _prepare_training_data_for_model(
            df_train_slice, last_imputed_backward_chunk.columns.tolist(), all_feature_cols, min_completeness_percent_train
        )
        if df_train_masked is None: break
        
        # Build distance and connectivity matrices for this training period
        dist_matrix = build_distance_matrix(df_coords, cols_to_keep)
        if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
            conn_matrix = build_connectivity_matrix(df_contrib_filtered, cols_to_keep, station_name_to_vcode)
            current_model_type_key = "full_model"
        else:
            conn_matrix = pd.DataFrame(0, index=cols_to_keep, columns=cols_to_keep, dtype=int)
            current_model_type_key = "no_contributor_model"

        # Generate a unique model hash for the current iteration
        current_model_hash_input = f"{discharge_path}-{lat_long_path}-{contrib_path}-{current_backward_start}-{current_backward_end}-{min_completeness_percent_train}-{current_model_type_key}"
        current_model_hash = hashlib.md5(current_model_hash_input.encode()).hexdigest()
        current_model_filepath = os.path.join(MODEL_CACHE_DIR, f'model_{current_model_hash}.pkl')

        trained_model = None
        if os.path.exists(current_model_filepath):
            print(f"Loading model from cache: {current_model_filepath}")
            with open(current_model_filepath, 'rb') as f:
                trained_model = pickle.load(f)
        else:
            if current_model_type_key == "full_model":
                print(f"--- Training model (Full Model) for {current_backward_start}-{current_backward_end} ---")
                trained_model = train_full_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
            elif current_model_type_key == "no_contributor_model":
                print(f"--- Training model (No Contributor Model) for {current_backward_start}-{current_backward_end} ---")
                trained_model = train_no_contributor_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
            if trained_model:
                with open(current_model_filepath, 'wb') as f:
                    pickle.dump(trained_model, f)
                print(f"Model saved to cache: {current_model_filepath}")

        if trained_model is None: break

        # Impute next chunk
        df_impute_slice_raw = df_full_data_with_features.loc[f"{current_backward_start}":f"{current_backward_end}"]
        df_to_impute = _prepare_imputation_target_data(df_impute_slice_raw, trained_model.discharge_columns, all_feature_cols, trained_model)
        if df_to_impute.empty:
            current_backward_end = current_backward_start - 1
            continue

        imputed_chunk = trained_model.transform(df_to_impute)
        imputed_discharge = imputed_chunk[trained_model.discharge_columns]
        df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
        print(f"Backward imputation complete for {current_backward_start}-{current_backward_end}.")

        last_imputed_backward_chunk = imputed_discharge.copy()
        current_backward_end = current_backward_start - 1

    # --- Final Output ---
    # Determine which stations actually have imputed data
    final_stations = []
    for col in all_discharge_cols_overall:
        if col in df_master_imputed.columns and not df_master_imputed[col].isna().all():
            final_stations.append(col)
    
    output_filename = os.path.join(output_dir, "final_bursting_imputed_data.csv")
    df_master_imputed[final_stations].to_csv(output_filename)
    
    print(f"\nBursting Imputation complete. Final imputed data saved to: {output_filename}")
    print(f"Final stations with data: {len(final_stations)} out of {len(all_discharge_cols_overall)}")
    print(f"Final NaNs remaining: {df_master_imputed[final_stations].isna().sum().sum()}")

    return df_master_imputed[final_stations]

# --- Example of how to run the pipeline ---
if __name__ == "__main__":
    DISCHARGE_DATA_PATH = "discharge_data_cleaned.csv"
    LAT_LONG_DATA_PATH = "lat_long_discharge.csv"
    CONTRIBUTOR_DATA_PATH = "mahanadi_contribs.csv"

    final_imputed_data = run_rolling_imputation_pipeline(
        discharge_path=DISCHARGE_DATA_PATH,
        lat_long_path=LAT_LONG_DATA_PATH,
        contrib_path=CONTRIBUTOR_DATA_PATH,
        initial_train_window_size=5,
        imputation_chunk_size_years=5,
        overall_min_year=1970,
        overall_max_year=2010,
        min_completeness_percent_train=70.0,
        output_dir="bursting_imputed_results"
    )

    if final_imputed_data is not None:
        print("\nPipeline finished successfully.")
    else:
        print("\nBursting Imputation Pipeline failed to complete.")
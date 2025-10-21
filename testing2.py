# # import pandas as pd
# # import numpy as np
# # import os
# # import warnings
# # from datetime import datetime, timedelta
# # import pickle 
# # import hashlib 
# # from sklearn.metrics import mean_squared_error, mean_absolute_error

# # # Import actual implementations from your files
# # from missforest_imputer import ModifiedMissForest
# # from model_configurations import train_full_model, train_no_contributor_model 
# # from utils import (
# #     load_and_preprocess_data,
# #     add_temporal_features,
# #     build_distance_matrix,
# #     build_connectivity_matrix
# # )

# # warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# # # Directory to save/load trained models
# # MODEL_CACHE_DIR = "trained_models_cache"
# # os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


# # # --- KLING-GUPTA EFFICIENCY FUNCTION ---

# # def modified_kling_gupta_efficiency(simulated, observed):
# #     """
# #     Calculates the Modified Kling-Gupta Efficiency (KGE') score.

# #     KGE' assesses model performance based on correlation (r), bias ratio (beta),
# #     and variability ratio (gamma), giving a score between -inf and 1 (perfect).

# #     Formula: KGE' = 1 - sqrt((r-1)^2 + (beta-1)^2 + (gamma-1)^2)

# #     Args:
# #         simulated (np.ndarray): Array of simulated (predicted) values.
# #         observed (np.ndarray): Array of observed (actual) values.

# #     Returns:
# #         float: The Modified Kling-Gupta Efficiency (KGE') score.
# #     """
# #     # 1. Ensure inputs are numpy arrays
# #     simulated = np.asarray(simulated)
# #     observed = np.asarray(observed)

# #     # 2. Handle missing data (NaNs) by removing them from both arrays
# #     valid_indices = ~np.isnan(simulated) & ~np.isnan(observed)
# #     simulated = simulated[valid_indices]
# #     observed = observed[valid_indices]

# #     # Check for empty arrays after cleaning
# #     if len(simulated) == 0:
# #         print("Warning: No valid data points found after removing NaNs.")
# #         return np.nan

# #     # 3. Calculate correlation coefficient (r)
# #     # Using np.corrcoef returns a 2x2 matrix; we take the off-diagonal element
# #     # Check if standard deviations are zero to prevent RuntimeWarning/NaN
# #     if np.std(simulated) == 0 or np.std(observed) == 0:
# #         # If one series is constant, correlation is undefined or 0 (treat as 0)
# #         r = 0.0
# #     else:
# #         r = np.corrcoef(simulated, observed)[0, 1]

# #     # 4. Calculate Bias ratio (beta)
# #     # Ratio of means: beta = (mean_simulated / mean_observed)
# #     # Check if mean_observed is zero to prevent division by zero
# #     mean_obs = np.mean(observed)
# #     mean_sim = np.mean(simulated)

# #     if mean_obs == 0:
# #         # If observed mean is zero, bias is ill-defined.
# #         # Handle defensively, possibly returning a poor score if simulation is non-zero
# #         if mean_sim == 0:
# #             beta = 1.0 # Perfect bias if both are zero
# #         else:
# #             beta = np.nan # Undefined, but we'll proceed using 1.0 in the KGE' error term
# #     else:
# #         beta = mean_sim / mean_obs

# #     # 5. Calculate Variability ratio (gamma)
# #     # Ratio of Coefficients of Variation: gamma = (std_sim / mean_sim) / (std_obs / mean_obs)
# #     std_obs = np.std(observed)
# #     std_sim = np.std(simulated)

# #     # Calculate CVs
# #     cv_obs = (std_obs / mean_obs) if mean_obs != 0 else np.nan
# #     cv_sim = (std_sim / mean_sim) if mean_sim != 0 else np.nan

# #     if np.isnan(cv_obs) or cv_obs == 0:
# #         # If observed data is constant or mean is zero, gamma is problematic.
# #         # We treat this defensively. If simulated is also constant, gamma=1.
# #         if np.isnan(cv_sim) or cv_sim == 0:
# #             gamma = 1.0
# #         else:
# #             gamma = np.nan # Undefined, proceed using 1.0 in the KGE' error term
# #     else:
# #         gamma = cv_sim / cv_obs

# #     # Clean up components for KGE' calculation (if any component is NaN, treat as worst case)
# #     # Note: KGE' formulation uses the squares of the errors, so (component - 1)^2.
# #     # If a component is NaN (due to zero mean/std), we treat its error as 1 (worst case)
# #     r_error = (r - 1) ** 2 if not np.isnan(r) else 1.0
# #     beta_error = (beta - 1) ** 2 if not np.isnan(beta) else 1.0
# #     gamma_error = (gamma - 1) ** 2 if not np.isnan(gamma) else 1.0

# #     # 6. Apply KGE' formula
# #     kge_prime = 1 - np.sqrt(r_error + beta_error + gamma_error)

# #     return kge_prime


# # # --- HELPER FUNCTIONS FOR PIPELINE ---

# # def find_best_training_period(df_data, min_year, max_year, window_size=3, min_overall_completeness_percent=70.0):
# #     """
# #     Identifies optimal training period (highest completeness) within a year range.
# #     Returns the best period and the list of stations with sufficient data.
# #     """
# #     discharge_cols = [col for col in df_data.columns if not col.startswith('day_of_year_')]
# #     if not discharge_cols:
# #         print("Warning: No discharge columns for period selection.")
# #         return None, None, []

# #     best_period = (None, None)
# #     best_stations = []
# #     max_completeness_found = -1.0 

# #     print(f"\nSearching for best {window_size}-year training period between {min_year}-{max_year} (min {min_overall_completeness_percent}% completeness required)...")

# #     for start_year in range(min_year, max_year - window_size + 2):
# #         end_year = start_year + window_size - 1
        
# #         if start_year > df_data.index.year.max() or end_year < df_data.index.year.min():
# #             continue

# #         period_data = df_data[(df_data.index.year >= start_year) & (df_data.index.year <= end_year)]
# #         if period_data.empty:
# #             continue

# #         period_discharge_data = period_data[discharge_cols]
# #         total_cells_in_period = period_discharge_data.shape[0] * period_discharge_data.shape[1]
        
# #         if total_cells_in_period == 0:
# #             current_completeness = 0.0
# #         else:
# #             non_na_cells_in_period = period_discharge_data.notna().sum().sum()
# #             current_completeness = (non_na_cells_in_period / total_cells_in_period) * 100

# #         print(f"  Period {start_year}-{end_year}: Overall completeness = {current_completeness:.2f}%")

# #         if current_completeness >= min_overall_completeness_percent:
# #             if current_completeness > max_completeness_found:
# #                 max_completeness_found = current_completeness
# #                 best_period = (start_year, end_year)
# #                 # Find stations with sufficient data in this period
# #                 min_non_na_count = int(min_overall_completeness_percent / 100 * len(period_data))
# #                 best_stations = period_discharge_data.columns[
# #                     period_discharge_data.notna().sum() >= min_non_na_count
# #                 ].tolist()

# #     if best_period == (None, None):
# #         print(f"Error: No {window_size}-year period found with at least {min_overall_completeness_percent}% completeness. Insufficient data for training.")
# #         return None, None, []
# #     else:
# #         print(f"\nBest training period identified: {best_period[0]}-{best_period[1]} with {max_completeness_found:.2f}% completeness.")
# #         print(f"Stations with sufficient data: {len(best_stations)} out of {len(discharge_cols)}")
# #     return best_period[0], best_period[1], best_stations

# # def _prepare_training_data_for_model(
# #     df_train_slice, all_discharge_cols, all_feature_cols, min_completeness_percent_train
# # ):
# #     """
# #     Helper function to prepare training data. Handles column filtering, 
# #     historical mean initialization, and simulated missingness.
# #     """
# #     min_non_na_count = int(min_completeness_percent_train / 100 * len(df_train_slice))
    
# #     cols_to_keep_discharge = []
# #     if not df_train_slice[all_discharge_cols].empty:
# #         cols_to_keep_discharge = df_train_slice[all_discharge_cols].columns[
# #             df_train_slice[all_discharge_cols].notna().sum() >= min_non_na_count
# #         ].tolist()
    
# #     if not cols_to_keep_discharge:
# #         return None, None 

# #     final_cols_for_imputer = list(set(cols_to_keep_discharge + all_feature_cols))
# #     df_train_period_filtered = df_train_slice[final_cols_for_imputer].copy()

# #     # --- START: MODIFIED INITIALIZATION ---
# #     df_train_discharge_only = df_train_period_filtered[cols_to_keep_discharge].copy()
    
# #     print("Applying historical (day-of-year) mean initialization...")
    
# #     # We need the full slice (with index) to calculate means
# #     df_train_slice_for_means = df_train_period_filtered.copy()
    
# #     if 'doy' not in df_train_slice_for_means.columns:
# #          df_train_slice_for_means['doy'] = df_train_slice_for_means.index.dayofyear
    
# #     # Calculate historical mean for each day of the year (doy) for each station *within this slice*
# #     historical_means = df_train_slice_for_means.groupby('doy')[cols_to_keep_discharge].mean()

# #     # --- FIX: Get the 'doy' Series (which has the correct DatetimeIndex) ---
# #     doy_series = df_train_slice_for_means['doy']

# #     # Now, fill NaNs in the discharge-only dataframe
# #     for col in cols_to_keep_discharge:
# #         if df_train_discharge_only[col].isnull().all():
# #             df_train_discharge_only[col] = 0.0 # Keep the all-NaN fallback
# #         else:
# #             # --- FIX: Map using the 'doy_series' (a Series) instead of 'index.dayofyear' (an Index) ---
# #             doy_map = doy_series.map(historical_means[col])
            
# #             # doy_map is now a Series with the DatetimeIndex, which fillna can use
# #             df_train_discharge_only[col] = df_train_discharge_only[col].fillna(doy_map)
            
# #             # If any NaNs remain (e.g., a day of year had no data in the slice),
# #             # fill with the overall column mean as a final fallback.
# #             if df_train_discharge_only[col].isnull().any():
# #                 col_mean = df_train_discharge_only[col].mean()
# #                 if pd.isna(col_mean): # Handle case where column is *still* all-NaN after doy fill
# #                     col_mean = 0.0
# #                 df_train_discharge_only[col] = df_train_discharge_only[col].fillna(col_mean)
    
# #     print("Historical mean initialization complete.")
# #     # --- END: MODIFIED INITIALIZATION ---


# #     # Simulate 10% random missingness for model training
# #     np.random.seed(42)
# #     train_mask_simulated = np.random.rand(*df_train_discharge_only.shape) < 0.1
# #     df_train_masked_for_model = df_train_period_filtered.copy()
# #     df_train_masked_for_model[cols_to_keep_discharge] = df_train_discharge_only.mask(train_mask_simulated)
    
# #     print(f"Simulated 10% random missingness in training data.")
# #     return df_train_masked_for_model, cols_to_keep_discharge


# # def _prepare_imputation_target_data(
# #     df_raw_slice, model_discharge_cols, all_feature_cols, trained_model
# # ):
# #     """
# #     Helper function to prepare data for imputation using a trained model.
# #     """
# #     # The model expects the discharge columns it was trained on, plus all feature columns
# #     cols_for_imputation_target = list(set(model_discharge_cols + all_feature_cols))
    
# #     # Reindex to ensure alignment. This adds any missing columns as all-NaN.
# #     df_to_impute = df_raw_slice.reindex(columns=cols_for_imputation_target).copy()

# #     # Handle all-NaN columns in the target data slice
# #     for col in model_discharge_cols:
# #         if col not in df_to_impute.columns:
# #             print(f"Warning: Column '{col}' expected by model not in target slice. Adding as NaN.")
# #             df_to_impute[col] = np.nan
            
# #         # If a column is entirely missing, fill it with 0 if the model's learned mean was also NaN.
# #         if df_to_impute[col].isnull().all() and col in trained_model.col_means and np.isnan(trained_model.col_means.get(col)):
# #             df_to_impute[col] = 0.0
# #     return df_to_impute

# # # --- BURSTING IMPUTATION PIPELINE ---
# # def run_bursting_imputation_pipeline(
# #     df_initial_masked=None, 
# #     df_full_data_with_features_param=None, 
# #     all_discharge_cols_overall_param=None, 
# #     df_contrib_filtered_param=None, 
# #     df_coords_param=None,
# #     station_name_to_vcode_param=None,
    
# #     discharge_path=None, lat_long_path=None, contrib_path=None,
    
# #     initial_train_window_size=5,
# #     imputation_chunk_size_years=5,
# #     overall_min_year=1976, overall_max_year=2016,
# #     min_completeness_percent_train=70.0,
# #     output_dir="bursting_imputed_results"
# # ):
# #     """
# #     Implements a "bursting" imputation pipeline with bidirectional imputation
# #     and model caching. Can accept pre-loaded data for evaluation.
# #     """
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     print("--- Starting Bursting Imputation Pipeline ---")

# #     # --- Initial Setup ---
# #     df_full_data_with_features = None
# #     all_discharge_cols_overall = None
# #     df_contrib_filtered = None
# #     df_coords = None
# #     station_name_to_vcode = None
# #     df_master_imputed = None
    
# #     if df_initial_masked is not None:
# #         print("Using pre-loaded and pre-masked initial data for evaluation.")
# #         df_master_imputed = df_initial_masked.copy()
        
# #         if (df_full_data_with_features_param is None or 
# #             all_discharge_cols_overall_param is None or
# #             df_coords_param is None):
# #             raise ValueError("When df_initial_masked is provided, all helper dataframes must also be provided.")
        
# #         df_full_data_with_features = df_full_data_with_features_param
# #         all_discharge_cols_overall = all_discharge_cols_overall_param
# #         df_contrib_filtered = df_contrib_filtered_param 
# #         df_coords = df_coords_param
# #         station_name_to_vcode = station_name_to_vcode_param 
    
# #     else:
# #         print("Loading data from paths for standalone run.")
# #         try:
# #             df_discharge_raw, df_contrib_filtered, df_coords, vcode_to_station_name, station_name_to_vcode = \
# #                 load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
# #             if df_discharge_raw is None:
# #                 print("FATAL ERROR: Failed to load data. Exiting.")
# #                 return None
# #         except Exception as e:
# #             print(f"FATAL ERROR during initial data loading/preprocessing: {e}")
# #             return None
        
# #         all_discharge_cols_overall = df_discharge_raw.columns.tolist()
# #         df_full_data_with_features = add_temporal_features(df_discharge_raw)
# #         df_master_imputed = df_full_data_with_features.copy()

# #     # --- Feature Engineering ---
# #     temporal_features = ['day_of_year_sin', 'day_of_year_cos']
# #     all_feature_cols = temporal_features
# #     print(f"Total features created: {len(all_feature_cols)}")
# #     print(f"Master imputed DataFrame initialized. Total NaNs: {df_master_imputed[all_discharge_cols_overall].isna().sum().sum()}")

# #     # --- Phase 1: Find Best Initial Training Period & Train First Model ---
# #     print("\nFinding the best initial training period...")
# #     best_initial_train_start, best_initial_train_end, initial_train_stations = find_best_training_period(
# #         df_full_data_with_features, overall_min_year, overall_max_year,
# #         initial_train_window_size, min_completeness_percent_train
# #     )
# #     if best_initial_train_start is None:
# #         return None 
# #     print(f"Selected initial training period: {best_initial_train_start}-{best_initial_train_end}")
# #     print(f"Initial training stations: {initial_train_stations}")

# #     print("\nTraining initial model on the best period...")
# #     df_initial_train_slice = df_master_imputed.loc[
# #         (df_master_imputed.index.year >= best_initial_train_start) &
# #         (df_master_imputed.index.year <= best_initial_train_end)
# #     ]
# #     df_initial_train_masked, cols_to_keep_initial = _prepare_training_data_for_model(
# #         df_initial_train_slice, initial_train_stations, all_feature_cols, min_completeness_percent_train
# #     )
# #     if df_initial_train_masked is None:
# #         print("Error: Initial training data preparation failed. Cannot train initial model.")
# #         return None

# #     dist_matrix_initial = build_distance_matrix(df_coords, cols_to_keep_initial)
# #     if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
# #         conn_matrix_initial = build_connectivity_matrix(df_contrib_filtered, cols_to_keep_initial, station_name_to_vcode)
# #         model_type_key = "full_model"
# #     else:
# #         print("No valid contribution data. Building a zero connectivity matrix for initial training.")
# #         conn_matrix_initial = pd.DataFrame(0, index=cols_to_keep_initial, columns=cols_to_keep_initial, dtype=int)
# #         model_type_key = "no_contributor_model"

# #     model_hash_input_initial = f"{discharge_path}-{lat_long_path}-{contrib_path}-{best_initial_train_start}-{best_initial_train_end}-{min_completeness_percent_train}-{model_type_key}"
# #     model_hash_initial = hashlib.md5(model_hash_input_initial.encode()).hexdigest()
# #     model_filepath_initial = os.path.join(MODEL_CACHE_DIR, f'model_{model_hash_initial}.pkl')

# #     trained_model_initial = None
# #     if os.path.exists(model_filepath_initial):
# #         print(f"Loading initial model from cache: {model_filepath_initial}")
# #         with open(model_filepath_initial, 'rb') as f:
# #             trained_model_initial = pickle.load(f)
# #     else:
# #         if model_type_key == "full_model":
# #             print("--- Training initial model (Full Model) ---")
# #             trained_model_initial = train_full_model(
# #                 df_initial_train_masked, dist_matrix_initial, conn_matrix_initial, all_feature_cols
# #             )
# #         elif model_type_key == "no_contributor_model":
# #             print("--- Training initial model (No Contributor Model) ---")
# #             trained_model_initial = train_no_contributor_model(
# #                 df_initial_train_masked, dist_matrix_initial, conn_matrix_initial, all_feature_cols
# #             )
# #         if trained_model_initial:
# #             with open(model_filepath_initial, 'wb') as f:
# #                 pickle.dump(trained_model_initial, f)
# #             print(f"Initial model saved to cache: {model_filepath_initial}")

# #     if trained_model_initial is None:
# #         print("Error: Initial model training failed. Exiting pipeline.")
# #         return None
# #     model_discharge_cols_initial = trained_model_initial.discharge_columns
    
# #     # --- Phase 2: Initial Bidirectional Imputation ---
# #     print(f"\nPerforming initial forward imputation...")
# #     impute_forward_start = best_initial_train_end + 1
# #     impute_forward_end = min(overall_max_year, impute_forward_start + imputation_chunk_size_years - 1)
    
# #     print(f"Imputing the initial training chunk ({best_initial_train_start}-{best_initial_train_end}) itself...")
# #     df_initial_train_to_impute = _prepare_imputation_target_data(
# #         df_initial_train_slice, model_discharge_cols_initial, all_feature_cols, trained_model_initial
# #     )
# #     imputed_initial_train_chunk = trained_model_initial.transform(df_initial_train_to_impute)
# #     imputed_initial_discharge = imputed_initial_train_chunk[model_discharge_cols_initial]
# #     df_master_imputed.loc[imputed_initial_discharge.index, imputed_initial_discharge.columns] = imputed_initial_discharge.values
# #     last_imputed_forward_chunk = imputed_initial_discharge.copy()
# #     last_imputed_backward_chunk = imputed_initial_discharge.copy() 


# #     if impute_forward_start <= impute_forward_end:
# #         df_impute_slice_raw = df_master_imputed.loc[f"{impute_forward_start}":f"{impute_forward_end}"]
# #         df_to_impute = _prepare_imputation_target_data(
# #             df_impute_slice_raw, model_discharge_cols_initial, all_feature_cols, trained_model_initial
# #         )
# #         if not df_to_impute.empty:
# #             imputed_chunk = trained_model_initial.transform(df_to_impute)
# #             imputed_discharge = imputed_chunk[model_discharge_cols_initial]
# #             df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
# #             print(f"Initial forward imputation complete for {impute_forward_start}-{impute_forward_end}.")
# #             last_imputed_forward_chunk = imputed_discharge.copy()

# #     print(f"\nPerforming initial backward imputation...")
# #     impute_backward_end = best_initial_train_start - 1
# #     impute_backward_start = max(overall_min_year, impute_backward_end - imputation_chunk_size_years + 1)

# #     if impute_backward_start <= impute_backward_end:
# #         df_impute_slice_raw = df_master_imputed.loc[f"{impute_backward_start}":f"{impute_backward_end}"]
# #         df_to_impute = _prepare_imputation_target_data(
# #             df_impute_slice_raw, model_discharge_cols_initial, all_feature_cols, trained_model_initial
# #         )
# #         if not df_to_impute.empty:
# #             imputed_chunk = trained_model_initial.transform(df_to_impute)
# #             imputed_discharge = imputed_chunk[model_discharge_cols_initial]
# #             df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
# #             print(f"Initial backward imputation complete for {impute_backward_start}-{impute_backward_end}.")
# #             last_imputed_backward_chunk = imputed_discharge.copy()

# #     # --- Phase 3: Forward Bursting Loop ---
# #     print("\n--- Starting Forward Bursting Imputation Loop ---")
# #     current_forward_start = impute_forward_end + 1
# #     while current_forward_start <= overall_max_year:
# #         current_forward_end = min(overall_max_year, current_forward_start + imputation_chunk_size_years - 1)
# #         print(f"Forward Bursting: Training on last imputed chunk, Imputing {current_forward_start}-{current_forward_end}")

# #         df_train_slice = last_imputed_forward_chunk.merge(df_full_data_with_features[all_feature_cols], left_index=True, right_index=True)
# #         df_train_masked, cols_to_keep = _prepare_training_data_for_model(
# #             df_train_slice, last_imputed_forward_chunk.columns.tolist(), all_feature_cols, min_completeness_percent_train
# #         )
# #         if df_train_masked is None: break

# #         dist_matrix = build_distance_matrix(df_coords, cols_to_keep)
# #         if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
# #             conn_matrix = build_connectivity_matrix(df_contrib_filtered, cols_to_keep, station_name_to_vcode)
# #             current_model_type_key = "full_model"
# #         else:
# #             conn_matrix = pd.DataFrame(0, index=cols_to_keep, columns=cols_to_keep, dtype=int)
# #             current_model_type_key = "no_contributor_model"

# #         current_model_hash_input = f"{discharge_path}-{lat_long_path}-{contrib_path}-{current_forward_start}-{current_forward_end}-{min_completeness_percent_train}-{current_model_type_key}"
# #         current_model_hash = hashlib.md5(current_model_hash_input.encode()).hexdigest()
# #         current_model_filepath = os.path.join(MODEL_CACHE_DIR, f'model_{current_model_hash}.pkl')

# #         trained_model = None
# #         if os.path.exists(current_model_filepath):
# #             print(f"Loading model from cache: {current_model_filepath}")
# #             with open(current_model_filepath, 'rb') as f:
# #                 trained_model = pickle.load(f)
# #         else:
# #             if current_model_type_key == "full_model":
# #                 print(f"--- Training model (Full Model) for {current_forward_start}-{current_forward_end} ---")
# #                 trained_model = train_full_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
# #             elif current_model_type_key == "no_contributor_model":
# #                 print(f"--- Training model (No Contributor Model) for {current_forward_start}-{current_forward_end} ---")
# #                 trained_model = train_no_contributor_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
# #             if trained_model:
# #                 with open(current_model_filepath, 'wb') as f:
# #                     pickle.dump(trained_model, f)
# #                 print(f"Model saved to cache: {current_model_filepath}")

# #         if trained_model is None: break
        
# #         df_impute_slice_raw = df_master_imputed.loc[f"{current_forward_start}":f"{current_forward_end}"]
# #         df_to_impute = _prepare_imputation_target_data(df_impute_slice_raw, trained_model.discharge_columns, all_feature_cols, trained_model)
# #         if df_to_impute.empty:
# #             current_forward_start = current_forward_end + 1
# #             continue
            
# #         imputed_chunk = trained_model.transform(df_to_impute)
# #         imputed_discharge = imputed_chunk[trained_model.discharge_columns]
# #         df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
# #         print(f"Forward imputation complete for {current_forward_start}-{current_forward_end}.")
        
# #         last_imputed_forward_chunk = imputed_discharge.copy()
# #         current_forward_start = current_forward_end + 1

# #     # --- Phase 4: Backward Bursting Loop ---
# #     print("\n--- Starting Backward Bursting Imputation Loop ---")
# #     current_backward_end = impute_backward_start - 1
# #     while current_backward_end >= overall_min_year:
# #         current_backward_start = max(overall_min_year, current_backward_end - imputation_chunk_size_years + 1)
# #         print(f"Backward Bursting: Training on last imputed chunk, Imputing {current_backward_start}-{current_backward_end}")

# #         df_train_slice = last_imputed_backward_chunk.merge(df_full_data_with_features[all_feature_cols], left_index=True, right_index=True)
# #         df_train_masked, cols_to_keep = _prepare_training_data_for_model(
# #             df_train_slice, last_imputed_backward_chunk.columns.tolist(), all_feature_cols, min_completeness_percent_train
# #         )
# #         if df_train_masked is None: break
        
# #         dist_matrix = build_distance_matrix(df_coords, cols_to_keep)
# #         if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
# #             conn_matrix = build_connectivity_matrix(df_contrib_filtered, cols_to_keep, station_name_to_vcode)
# #             current_model_type_key = "full_model"
# #         else:
# #             conn_matrix = pd.DataFrame(0, index=cols_to_keep, columns=cols_to_keep, dtype=int)
# #             current_model_type_key = "no_contributor_model"

# #         current_model_hash_input = f"{discharge_path}-{lat_long_path}-{contrib_path}-{current_backward_start}-{current_backward_end}-{min_completeness_percent_train}-{current_model_type_key}"
# #         current_model_hash = hashlib.md5(current_model_hash_input.encode()).hexdigest()
# #         current_model_filepath = os.path.join(MODEL_CACHE_DIR, f'model_{current_model_hash}.pkl')

# #         trained_model = None
# #         if os.path.exists(current_model_filepath):
# #             print(f"Loading model from cache: {current_model_filepath}")
# #             with open(current_model_filepath, 'rb') as f:
# #                 trained_model = pickle.load(f)
# #         else:
# #             if current_model_type_key == "full_model":
# #                 print(f"--- Training model (Full Model) for {current_backward_start}-{current_backward_end} ---")
# #                 trained_model = train_full_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
# #             elif current_model_type_key == "no_contributor_model":
# #                 print(f"--- Training model (No Contributor Model) for {current_backward_start}-{current_backward_end} ---")
# #                 trained_model = train_no_contributor_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
# #             if trained_model:
# #                 with open(current_model_filepath, 'wb') as f:
# #                     pickle.dump(trained_model, f)
# #                 print(f"Model saved to cache: {current_model_filepath}")

# #         if trained_model is None: break

# #         df_impute_slice_raw = df_master_imputed.loc[f"{current_backward_start}":f"{current_backward_end}"]
# #         df_to_impute = _prepare_imputation_target_data(df_impute_slice_raw, trained_model.discharge_columns, all_feature_cols, trained_model)
# #         if df_to_impute.empty:
# #             current_backward_end = current_backward_start - 1
# #             continue

# #         imputed_chunk = trained_model.transform(df_to_impute)
# #         imputed_discharge = imputed_chunk[trained_model.discharge_columns]
# #         df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
# #         print(f"Backward imputation complete for {current_backward_start}-{current_backward_end}.")

# #         last_imputed_backward_chunk = imputed_discharge.copy()
# #         current_backward_end = current_backward_start - 1

# #     # --- Final Output ---
# #     output_filename = os.path.join(output_dir, "final_bursting_imputed_data.csv")
# #     df_master_imputed[all_discharge_cols_overall].to_csv(output_filename)
    
# #     print(f"\nBursting Imputation complete. Final imputed data saved to: {output_filename}")
# #     print(f"Final NaNs remaining in master imputed data: {df_master_imputed[all_discharge_cols_overall].isna().sum().sum()}")

# #     return df_master_imputed[all_discharge_cols_overall]


# # # --- EVALUATION FUNCTION ---
# # def evaluate_bursting_pipeline(discharge_path, lat_long_path, contrib_path=None,
# #                                overall_min_year=1976, overall_max_year=2016,
# #                                min_completeness_percent_train=70.0,
# #                                initial_train_window_size=3, 
# #                                imputation_chunk_size_years=3,
# #                                masking_percentage=0.10, 
# #                                evaluation_dir="evaluation_results"):
# #     """
# #     Evaluates the accuracy of the Bursting imputation pipeline.
# #     """
# #     os.makedirs(evaluation_dir, exist_ok=True)
# #     temporal_features = ['day_of_year_sin', 'day_of_year_cos']

# #     print("--- Starting Evaluation of Bursting Imputation Method ---")

# #     # --- Phase 1: Load and Prepare Original Data ---
# #     try:
# #         df_discharge_raw, df_contrib_filtered, df_coords, vcode_to_station_name, station_name_to_vcode = \
# #             load_and_preprocess_data(discharge_path, contrib_path, lat_long_path)
        
# #         if df_discharge_raw.empty:
# #             print("Error: Discharge data is empty after initial load. Exiting evaluation.")
# #             return None
            
# #     except Exception as e:
# #         print(f"FATAL ERROR during initial data loading/preprocessing for evaluation: {e}")
# #         return None

# #     df_full_data_with_features = add_temporal_features(df_discharge_raw)

# #     df_full_data_with_features = df_full_data_with_features.loc[
# #         (df_full_data_with_features.index.year >= overall_min_year) &
# #         (df_full_data_with_features.index.year <= overall_max_year)
# #     ].copy()
# #     print(f"Dataset trimmed to range: {overall_min_year}-{overall_max_year}. New shape: {df_full_data_with_features.shape}")

# #     if df_full_data_with_features.empty:
# #         print("Error: Dataset is empty after trimming to specified year range. Exiting evaluation.")
# #         return None

# #     all_discharge_cols_overall = [col for col in df_full_data_with_features.columns if col not in temporal_features]
    
# #     if not all_discharge_cols_overall:
# #         print("Error: No discharge columns identified. Exiting evaluation.")
# #         return None
    
# #     # --- Phase 2: Create Masked Data for Evaluation ---
# #     print(f"\nCreating masked data for evaluation ({masking_percentage*100:.1f}% of discharge data will be masked)...")
    
# #     df_original_truth = df_full_data_with_features[all_discharge_cols_overall].copy()
# #     df_to_impute_for_pipelines = df_full_data_with_features.copy()
# #     existing_nan_mask = df_original_truth.isna()

# #     np.random.seed(42) 
# #     comparison_mask = np.random.rand(*df_original_truth.shape) < masking_percentage
    
# #     test_locations_mask = comparison_mask & (~existing_nan_mask)
    
# #     if test_locations_mask.sum().sum() == 0:
# #         print("Warning: No original (non-NaN) values were masked for comparison. Cannot evaluate accuracy.")
# #         return None

# #     original_values_at_masked_locs = df_original_truth.values[test_locations_mask]
    
# #     for col_idx, col_name in enumerate(all_discharge_cols_overall):
# #         df_to_impute_for_pipelines.loc[test_locations_mask.index, col_name] = \
# #             df_to_impute_for_pipelines.loc[test_locations_mask.index, col_name].mask(test_locations_mask[col_name])

# #     print(f"Total original NaNs in discharge data: {existing_nan_mask.sum().sum()}")
# #     print(f"Total additional cells masked for comparison: {test_locations_mask.sum().sum()}")
# #     print(f"Total NaNs in data sent to pipelines: {df_to_impute_for_pipelines[all_discharge_cols_overall].isna().sum().sum()}")

# #     # --- Phase 3: Run Bursting Imputation Pipeline ---
# #     print("\n--- Running Bursting Imputation ---")
# #     imputed_data_bursting = run_bursting_imputation_pipeline(
# #         df_initial_masked=df_to_impute_for_pipelines,
# #         df_full_data_with_features_param=df_full_data_with_features,
# #         all_discharge_cols_overall_param=all_discharge_cols_overall,
# #         df_contrib_filtered_param=df_contrib_filtered,
# #         df_coords_param=df_coords,
# #         station_name_to_vcode_param=station_name_to_vcode,
# #         contrib_path=contrib_path, 
        
# #         overall_min_year=overall_min_year,
# #         overall_max_year=overall_max_year,
# #         initial_train_window_size=initial_train_window_size,
# #         imputation_chunk_size_years=imputation_chunk_size_years,
# #         min_completeness_percent_train=min_completeness_percent_train,
# #         output_dir=os.path.join(evaluation_dir, "bursting_results")
# #     )

# #     if imputed_data_bursting is None:
# #         print("Error: Bursting imputation pipeline failed. Cannot perform evaluation.")
# #         return None

# #     # --- Phase 4: Compare Accuracy (MODIFIED) ---
# #     print("\n--- Comparing Imputation Accuracy ---")

# #     imputed_data_bursting_aligned = imputed_data_bursting.reindex(index=df_original_truth.index, columns=df_original_truth.columns)
# #     imputed_values_bursting = imputed_data_bursting_aligned.values[test_locations_mask]
    
# #     valid_comparison_mask_bursting = ~np.isnan(imputed_values_bursting)
    
# #     y_true_bursting = original_values_at_masked_locs[valid_comparison_mask_bursting]
# #     y_pred_bursting = imputed_values_bursting[valid_comparison_mask_bursting]

# #     if y_true_bursting.size == 0 or y_pred_bursting.size == 0:
# #         print("Error: No valid data points for comparison after imputation and NaN filtering. Check masking and pipeline output.")
# #         return None
    
# #     # Calculate all four metrics
# #     rmse_bursting = np.sqrt(mean_squared_error(y_true_bursting, y_pred_bursting))
# #     mae_bursting = mean_absolute_error(y_true_bursting, y_pred_bursting)
    
# #     # Calculate NSE
# #     numerator = np.sum((y_true_bursting - y_pred_bursting) ** 2)
# #     denominator = np.sum((y_true_bursting - y_true_bursting.mean()) ** 2)
# #     nse_bursting = 1 - (numerator / denominator) if denominator != 0 else float('-inf')
    
# #     # Calculate KGE
# #     kge_bursting = modified_kling_gupta_efficiency(y_pred_bursting, y_true_bursting) # Note: (simulated, observed)

# #     results = {
# #         "bursting": {"RMSE": rmse_bursting, "MAE": mae_bursting, "NSE": nse_bursting, "KGE": kge_bursting}
# #     }

# #     print("\n--- Imputation Evaluation Results ---")
# #     print(f"Valid Data Points for Bursting Evaluation: {y_true_bursting.size}")
# #     print(f"Bursting Pipeline: RMSE = {rmse_bursting:.4f}, MAE = {mae_bursting:.4f}, NSE = {nse_bursting:.4f}, KGE = {kge_bursting:.4f}")

# #     results_df = pd.DataFrame.from_dict(results, orient='index')
# #     results_file = os.path.join(evaluation_dir, "imputation_evaluation_metrics.csv")
# #     results_df.to_csv(results_file)
# #     print(f"Evaluation metrics saved to {results_file}")

# #     return results

# # # --- Example of how to run the evaluation ---
# # if __name__ == "__main__":
# #     DISCHARGE_DATA_PATH = "D:\Downloads 17Oct24 copy\Chrome Downloads 2025 Oct onwards\streamflow-discharge-prediction-main\streamflow-discharge-prediction-main\discharge_data_cleaned.csv"
# #     LAT_LONG_DATA_PATH = "streamflow-discharge-prediction-main/lat_long_discharge.csv"
# #     CONTRIBUTOR_DATA_PATH = "streamflow-discharge-prediction-main/mahanadi_contribs.csv" # Set to None if no contributor data

# #     EVAL_OVERALL_MIN_YEAR = 1980 
# #     EVAL_OVERALL_MAX_YEAR = 1990 
# #     EVAL_MIN_COMPLETENESS_TRAIN = 70.0
# #     EVAL_INITIAL_TRAIN_WINDOW_SIZE = 5 
# #     EVAL_IMPUTATION_CHUNK_SIZE = 5 
# #     EVAL_MASKING_PERCENTAGE = 0.05 
# #     EVAL_OUTPUT_DIR = "evaluation_results" 

# #     evaluation_results = evaluate_bursting_pipeline(
# #         discharge_path=DISCHARGE_DATA_PATH,
# #         lat_long_path=LAT_LONG_DATA_PATH,
# #         contrib_path=CONTRIBUTOR_DATA_PATH,
# #         overall_min_year=EVAL_OVERALL_MIN_YEAR,
# #         overall_max_year=EVAL_OVERALL_MAX_YEAR,
# #         min_completeness_percent_train=EVAL_MIN_COMPLETENESS_TRAIN,
# #         initial_train_window_size=EVAL_INITIAL_TRAIN_WINDOW_SIZE,
# #         imputation_chunk_size_years=EVAL_IMPUTATION_CHUNK_SIZE,
# #         masking_percentage=EVAL_MASKING_PERCENTAGE,
# #         evaluation_dir=EVAL_OUTPUT_DIR
# #     )

# #     if evaluation_results:
# #         print("\n--- Final Evaluation Summary ---")
# #         for method, metrics in evaluation_results.items():
# #             print(f"{method.capitalize()} Metrics: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, NSE={metrics['NSE']:.4f}, KGE={metrics['KGE']:.4f}")
# #     else:
# #         print("\nEvaluation process did not complete successfully.")





# import pandas as pd
# import numpy as np
# import os
# import warnings
# from datetime import datetime, timedelta
# import pickle 
# import hashlib 
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# # Import actual implementations from your files
# from missforest_imputer import ModifiedMissForest
# from model_configurations import train_full_model, train_no_contributor_model 
# from utils import (
#     load_and_preprocess_data,
#     add_temporal_features,
#     build_distance_matrix,
#     build_connectivity_matrix,
#     create_contiguous_segment_gaps
# )

# warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# # Directory to save/load trained models
# MODEL_CACHE_DIR = "trained_models_cache"
# os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


# # --- KLING-GUPTA EFFICIENCY FUNCTION ---

# def modified_kling_gupta_efficiency(simulated, observed):
#     """
#     Calculates the Modified Kling-Gupta Efficiency (KGE') score.
#     """
#     simulated = np.asarray(simulated)
#     observed = np.asarray(observed)
#     valid_indices = ~np.isnan(simulated) & ~np.isnan(observed)
#     simulated = simulated[valid_indices]
#     observed = observed[valid_indices]
#     if len(simulated) == 0:
#         print("Warning: No valid data points found for KGE calculation.")
#         return np.nan
#     if np.std(simulated) == 0 or np.std(observed) == 0:
#         r = 0.0
#     else:
#         r = np.corrcoef(simulated, observed)[0, 1]
#     mean_obs = np.mean(observed)
#     mean_sim = np.mean(simulated)
#     if mean_obs == 0:
#         beta = 1.0 if mean_sim == 0 else np.nan 
#     else:
#         beta = mean_sim / mean_obs
#     std_obs = np.std(observed)
#     std_sim = np.std(simulated)
#     cv_obs = (std_obs / mean_obs) if mean_obs != 0 else np.nan
#     cv_sim = (std_sim / mean_sim) if mean_sim != 0 else np.nan
#     if np.isnan(cv_obs) or cv_obs == 0:
#         gamma = 1.0 if np.isnan(cv_sim) or cv_sim == 0 else np.nan 
#     else:
#         gamma = cv_sim / cv_obs
#     r_error = (r - 1) ** 2 if not np.isnan(r) else 1.0
#     beta_error = (beta - 1) ** 2 if not np.isnan(beta) else 1.0
#     gamma_error = (gamma - 1) ** 2 if not np.isnan(gamma) else 1.0
#     kge_prime = 1 - np.sqrt(r_error + beta_error + gamma_error)
#     return kge_prime


# # --- NEW HELPER FUNCTION FOR EVALUATION (MODIFIED) ---

# def _evaluate_at_gap_locations(df_original, df_gapped, df_imputed, discharge_cols, output_csv_path=None):
#     """
#     Calculates metrics only at the locations of the artificially created gaps.
#     Optionally saves the true vs. predicted values (with date and station context) to a CSV file.
#     """
#     y_true_eval, y_pred_eval = [], []
    
#     # --- MODIFICATION: Create a list to store detailed gap data ---
#     all_gap_data_rows = []

#     for station in discharge_cols:
#         if station not in df_imputed.columns or station not in df_original.columns:
#             continue
            
#         # Find where the artificial gaps were created
#         gap_mask = df_gapped[station].isnull() & df_original[station].notna()
        
#         if gap_mask.sum() > 0:
#             # Get values as a Series to keep the date index
#             predicted_vals_series = df_imputed.loc[gap_mask, station]
#             true_vals_series = df_original.loc[gap_mask, station]
            
#             # --- MODIFICATION: Loop through the gap points and store context ---
#             for date_index, true_val in true_vals_series.items():
#                 pred_val = predicted_vals_series.get(date_index)
                
#                 # Store data for the CSV
#                 all_gap_data_rows.append({
#                     'date': date_index,
#                     'station': station,
#                     'true_value': true_val,
#                     'predicted_value': pred_val
#                 })
                
#                 # Store data for metric calculation
#                 y_pred_eval.append(pred_val)
#                 y_true_eval.append(true_val)

#     if not y_true_eval:
#         print("Warning: No valid gap data found for evaluation.")
#         return {'RMSE': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

#     # --- Standard metric calculation (unchanged) ---
#     y_true = np.array(y_true_eval)
#     y_pred = np.array(y_pred_eval)
#     valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
#     y_true = y_true[valid_indices]
#     y_pred = y_pred[valid_indices]
    
#     if y_true.size == 0:
#         print("Warning: No valid data points left after filtering NaNs from imputation.")
#         return {'RMSE': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

#     # --- MODIFICATION: Save the detailed DataFrame ---
#     if output_csv_path:
#         try:
#             df_gap_data = pd.DataFrame(all_gap_data_rows)
#             # Reorder columns for clarity
#             df_gap_data = df_gap_data[['date', 'station', 'true_value', 'predicted_value']]
#             df_gap_data.to_csv(output_csv_path, index=False)
#             print(f"Saved gap true vs. predicted data (with context) to {output_csv_path}")
#         except Exception as e:
#             print(f"Warning: Could not save gap data to {output_csv_path}. Error: {e}")
#     # --- END MODIFIED SECTION ---

#     # Calculate metrics (unchanged)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     numerator = np.sum((y_true - y_pred) ** 2)
#     denominator = np.sum((y_true - y_true.mean()) ** 2)
#     nse = 1 - (numerator / denominator) if denominator != 0 else float('-inf')
#     kge = modified_kling_gupta_efficiency(y_pred, y_true) # (simulated, observed)
    
#     return {'RMSE': rmse, 'MAE': mae, 'NSE': nse, 'KGE': kge}


# # --- HELPER FUNCTIONS FOR PIPELINE ---

# def find_best_training_period(df_data, min_year, max_year, window_size=3, min_overall_completeness_percent=70.0):
#     """
#     Identifies optimal training period (highest completeness) within a year range.
#     Returns the best period and the list of stations with sufficient data.
#     """
#     discharge_cols = [col for col in df_data.columns if not col.startswith('day_of_year_')]
#     if not discharge_cols:
#         print("Warning: No discharge columns for period selection.")
#         return None, None, []
#     best_period = (None, None)
#     best_stations = []
#     max_completeness_found = -1.0 
#     print(f"\nSearching for best {window_size}-year training period between {min_year}-{max_year} (min {min_overall_completeness_percent}% completeness required)...")
#     for start_year in range(min_year, max_year - window_size + 2):
#         end_year = start_year + window_size - 1
#         if start_year > df_data.index.year.max() or end_year < df_data.index.year.min():
#             continue
#         period_data = df_data[(df_data.index.year >= start_year) & (df_data.index.year <= end_year)]
#         if period_data.empty:
#             continue
#         period_discharge_data = period_data[discharge_cols]
#         total_cells_in_period = period_discharge_data.shape[0] * period_discharge_data.shape[1]
#         if total_cells_in_period == 0:
#             current_completeness = 0.0
#         else:
#             non_na_cells_in_period = period_discharge_data.notna().sum().sum()
#             current_completeness = (non_na_cells_in_period / total_cells_in_period) * 100
#         if current_completeness >= min_overall_completeness_percent:
#             if current_completeness > max_completeness_found:
#                 max_completeness_found = current_completeness
#                 best_period = (start_year, end_year)
#                 min_non_na_count = int(min_overall_completeness_percent / 100 * len(period_data))
#                 best_stations = period_discharge_data.columns[
#                     period_discharge_data.notna().sum() >= min_non_na_count
#                 ].tolist()
#     if best_period == (None, None):
#         print(f"Error: No {window_size}-year period found with at least {min_overall_completeness_percent}% completeness. Insufficient data for training.")
#         return None, None, []
#     else:
#         print(f"\nBest training period identified: {best_period[0]}-{best_period[1]} with {max_completeness_found:.2f}% completeness.")
#         print(f"Stations with sufficient data: {len(best_stations)} out of {len(discharge_cols)}")
#     return best_period[0], best_period[1], best_stations

# def _prepare_training_data_for_model(
#     df_train_slice, all_discharge_cols, all_feature_cols, min_completeness_percent_train
# ):
#     """
#     Helper function to prepare training data. Handles column filtering, 
#     historical mean initialization, and simulated missingness.
#     """
#     min_non_na_count = int(min_completeness_percent_train / 100 * len(df_train_slice))
#     cols_to_keep_discharge = []
#     if not df_train_slice[all_discharge_cols].empty:
#         cols_to_keep_discharge = df_train_slice[all_discharge_cols].columns[
#             df_train_slice[all_discharge_cols].notna().sum() >= min_non_na_count
#         ].tolist()
#     if not cols_to_keep_discharge:
#         return None, None 
#     final_cols_for_imputer = list(set(cols_to_keep_discharge + all_feature_cols))
#     df_train_period_filtered = df_train_slice[final_cols_for_imputer].copy()
#     df_train_discharge_only = df_train_period_filtered[cols_to_keep_discharge].copy()
#     print("Applying historical (day-of-year) mean initialization...")
#     df_train_slice_for_means = df_train_period_filtered.copy()
#     if 'doy' not in df_train_slice_for_means.columns:
#          df_train_slice_for_means['doy'] = df_train_slice_for_means.index.dayofyear
#     historical_means = df_train_slice_for_means.groupby('doy')[cols_to_keep_discharge].mean()
#     doy_series = df_train_slice_for_means['doy']
#     for col in cols_to_keep_discharge:
#         if df_train_discharge_only[col].isnull().all():
#             df_train_discharge_only[col] = 0.0 
#         else:
#             doy_map = doy_series.map(historical_means[col])
#             df_train_discharge_only[col] = df_train_discharge_only[col].fillna(doy_map)
#             if df_train_discharge_only[col].isnull().any():
#                 col_mean = df_train_discharge_only[col].mean()
#                 if pd.isna(col_mean): 
#                     col_mean = 0.0
#                 df_train_discharge_only[col] = df_train_discharge_only[col].fillna(col_mean)
#     print("Historical mean initialization complete.")
#     np.random.seed(42)
#     train_mask_simulated = np.random.rand(*df_train_discharge_only.shape) < 0.1
#     df_train_masked_for_model = df_train_period_filtered.copy()
#     df_train_masked_for_model[cols_to_keep_discharge] = df_train_discharge_only.mask(train_mask_simulated)
#     print(f"Simulated 10% random missingness in training data.")
#     return df_train_masked_for_model, cols_to_keep_discharge


# def _prepare_imputation_target_data(
#     df_raw_slice, model_discharge_cols, all_feature_cols, trained_model
# ):
#     """
#     Helper function to prepare data for imputation using a trained model.
#     """
#     cols_for_imputation_target = list(set(model_discharge_cols + all_feature_cols))
#     df_to_impute = df_raw_slice.reindex(columns=cols_for_imputation_target).copy()
#     for col in model_discharge_cols:
#         if col not in df_to_impute.columns:
#             print(f"Warning: Column '{col}' expected by model not in target slice. Adding as NaN.")
#             df_to_impute[col] = np.nan
#         if df_to_impute[col].isnull().all() and col in trained_model.col_means and np.isnan(trained_model.col_means.get(col)):
#             df_to_impute[col] = 0.0
#     return df_to_impute


# # --- BURSTING IMPUTATION PIPELINE ---
# def run_bursting_imputation_pipeline(
#     df_initial_masked=None, 
#     df_full_data_with_features_param=None, 
#     all_discharge_cols_overall_param=None, 
#     df_contrib_filtered_param=None, 
#     df_coords_param=None,
#     station_name_to_vcode_param=None,
#     discharge_path=None, lat_long_path=None, contrib_path=None,
#     initial_train_window_size=5,
#     imputation_chunk_size_years=5,
#     overall_min_year=1976, overall_max_year=2016,
#     min_completeness_percent_train=70.0,
#     output_dir="bursting_imputed_results"
# ):
#     """
#     Implements a "bursting" imputation pipeline with bidirectional imputation
#     and model caching. Can accept pre-loaded data for evaluation.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     print("--- Starting Bursting Imputation Pipeline ---")
#     df_full_data_with_features = None
#     all_discharge_cols_overall = None
#     df_contrib_filtered = None
#     df_coords = None
#     station_name_to_vcode = None
#     df_master_imputed = None
#     if df_initial_masked is not None:
#         print("Using pre-loaded and pre-masked initial data for evaluation.")
#         df_master_imputed = df_initial_masked.copy()
#         if (df_full_data_with_features_param is None or 
#             all_discharge_cols_overall_param is None or
#             df_coords_param is None):
#             raise ValueError("When df_initial_masked is provided, all helper dataframes must also be provided.")
#         df_full_data_with_features = df_full_data_with_features_param
#         all_discharge_cols_overall = all_discharge_cols_overall_param
#         df_contrib_filtered = df_contrib_filtered_param 
#         df_coords = df_coords_param
#         station_name_to_vcode = station_name_to_vcode_param 
#     else:
#         print("Loading data from paths for standalone run.")
#         try:
#             df_discharge_raw, df_contrib_filtered, df_coords, vcode_to_station_name, station_name_to_vcode = \
#                 load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
#             if df_discharge_raw is None:
#                 print("FATAL ERROR: Failed to load data. Exiting.")
#                 return None
#         except Exception as e:
#             print(f"FATAL ERROR during initial data loading/preprocessing: {e}")
#             return None
#         all_discharge_cols_overall = df_discharge_raw.columns.tolist()
#         df_full_data_with_features = add_temporal_features(df_discharge_raw)
#         df_master_imputed = df_full_data_with_features.copy()

#     temporal_features = ['day_of_year_sin', 'day_of_year_cos']
#     all_feature_cols = temporal_features
#     print(f"Total features created: {len(all_feature_cols)}")
#     print(f"Master imputed DataFrame initialized. Total NaNs: {df_master_imputed[all_discharge_cols_overall].isna().sum().sum()}")

#     print("\nFinding the best initial training period...")
#     best_initial_train_start, best_initial_train_end, initial_train_stations = find_best_training_period(
#         df_full_data_with_features, overall_min_year, overall_max_year,
#         initial_train_window_size, min_completeness_percent_train
#     )
#     if best_initial_train_start is None:
#         return None 
#     print(f"Selected initial training period: {best_initial_train_start}-{best_initial_train_end}")

#     print("\nTraining initial model on the best period...")
#     df_initial_train_slice = df_master_imputed.loc[
#         (df_master_imputed.index.year >= best_initial_train_start) &
#         (df_master_imputed.index.year <= best_initial_train_end)
#     ]
#     df_initial_train_masked, cols_to_keep_initial = _prepare_training_data_for_model(
#         df_initial_train_slice, initial_train_stations, all_feature_cols, min_completeness_percent_train
#     )
#     if df_initial_train_masked is None:
#         print("Error: Initial training data preparation failed. Cannot train initial model.")
#         return None

#     dist_matrix_initial = build_distance_matrix(df_coords, cols_to_keep_initial)
#     if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
#         conn_matrix_initial = build_connectivity_matrix(df_contrib_filtered, cols_to_keep_initial, station_name_to_vcode)
#         model_type_key = "full_model"
#     else:
#         print("No valid contribution data. Building a zero connectivity matrix for initial training.")
#         conn_matrix_initial = pd.DataFrame(0, index=cols_to_keep_initial, columns=cols_to_keep_initial, dtype=int)
#         model_type_key = "no_contributor_model"

#     model_hash_input_initial = f"{discharge_path}-{lat_long_path}-{contrib_path}-{best_initial_train_start}-{best_initial_train_end}-{min_completeness_percent_train}-{model_type_key}"
#     model_hash_initial = hashlib.md5(model_hash_input_initial.encode()).hexdigest()
#     model_filepath_initial = os.path.join(MODEL_CACHE_DIR, f'model_{model_hash_initial}.pkl')

#     trained_model_initial = None
#     if os.path.exists(model_filepath_initial):
#         print(f"Loading initial model from cache: {model_filepath_initial}")
#         with open(model_filepath_initial, 'rb') as f:
#             trained_model_initial = pickle.load(f)
#     else:
#         if model_type_key == "full_model":
#             print("--- Training initial model (Full Model) ---")
#             trained_model_initial = train_full_model(
#                 df_initial_train_masked, dist_matrix_initial, conn_matrix_initial, all_feature_cols
#             )
#         elif model_type_key == "no_contributor_model":
#             print("--- Training initial model (No Contributor Model) ---")
#             trained_model_initial = train_no_contributor_model(
#                 df_initial_train_masked, dist_matrix_initial, conn_matrix_initial, all_feature_cols
#             )
#         if trained_model_initial:
#             with open(model_filepath_initial, 'wb') as f:
#                 pickle.dump(trained_model_initial, f)
#             print(f"Initial model saved to cache: {model_filepath_initial}")

#     if trained_model_initial is None:
#         print("Error: Initial model training failed. Exiting pipeline.")
#         return None
#     model_discharge_cols_initial = trained_model_initial.discharge_columns
    
#     print(f"\nPerforming initial forward imputation...")
#     impute_forward_start = best_initial_train_end + 1
#     impute_forward_end = min(overall_max_year, impute_forward_start + imputation_chunk_size_years - 1)
    
#     print(f"Imputing the initial training chunk ({best_initial_train_start}-{best_initial_train_end}) itself...")
#     df_initial_train_to_impute = _prepare_imputation_target_data(
#         df_initial_train_slice, model_discharge_cols_initial, all_feature_cols, trained_model_initial
#     )
#     imputed_initial_train_chunk = trained_model_initial.transform(df_initial_train_to_impute)
#     imputed_initial_discharge = imputed_initial_train_chunk[model_discharge_cols_initial]
#     df_master_imputed.loc[imputed_initial_discharge.index, imputed_initial_discharge.columns] = imputed_initial_discharge.values
#     last_imputed_forward_chunk = imputed_initial_discharge.copy()
#     last_imputed_backward_chunk = imputed_initial_discharge.copy() 

#     if impute_forward_start <= impute_forward_end:
#         df_impute_slice_raw = df_master_imputed.loc[f"{impute_forward_start}":f"{impute_forward_end}"]
#         df_to_impute = _prepare_imputation_target_data(
#             df_impute_slice_raw, model_discharge_cols_initial, all_feature_cols, trained_model_initial
#         )
#         if not df_to_impute.empty:
#             imputed_chunk = trained_model_initial.transform(df_to_impute)
#             imputed_discharge = imputed_chunk[model_discharge_cols_initial]
#             df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
#             print(f"Initial forward imputation complete for {impute_forward_start}-{impute_forward_end}.")
#             last_imputed_forward_chunk = imputed_discharge.copy()

#     print(f"\nPerforming initial backward imputation...")
#     impute_backward_end = best_initial_train_start - 1
#     impute_backward_start = max(overall_min_year, impute_backward_end - imputation_chunk_size_years + 1)

#     if impute_backward_start <= impute_backward_end:
#         df_impute_slice_raw = df_master_imputed.loc[f"{impute_backward_start}":f"{impute_backward_end}"]
#         df_to_impute = _prepare_imputation_target_data(
#             df_impute_slice_raw, model_discharge_cols_initial, all_feature_cols, trained_model_initial
#         )
#         if not df_to_impute.empty:
#             imputed_chunk = trained_model_initial.transform(df_to_impute)
#             imputed_discharge = imputed_chunk[model_discharge_cols_initial]
#             df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
#             print(f"Initial backward imputation complete for {impute_backward_start}-{impute_backward_end}.")
#             last_imputed_backward_chunk = imputed_discharge.copy()

#     print("\n--- Starting Forward Bursting Imputation Loop ---")
#     current_forward_start = impute_forward_end + 1
#     while current_forward_start <= overall_max_year:
#         current_forward_end = min(overall_max_year, current_forward_start + imputation_chunk_size_years - 1)
#         print(f"Forward Bursting: Training on last imputed chunk, Imputing {current_forward_start}-{current_forward_end}")
#         df_train_slice = last_imputed_forward_chunk.merge(df_full_data_with_features[all_feature_cols], left_index=True, right_index=True)
#         df_train_masked, cols_to_keep = _prepare_training_data_for_model(
#             df_train_slice, last_imputed_forward_chunk.columns.tolist(), all_feature_cols, min_completeness_percent_train
#         )
#         if df_train_masked is None: break
#         dist_matrix = build_distance_matrix(df_coords, cols_to_keep)
#         if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
#             conn_matrix = build_connectivity_matrix(df_contrib_filtered, cols_to_keep, station_name_to_vcode)
#             current_model_type_key = "full_model"
#         else:
#             conn_matrix = pd.DataFrame(0, index=cols_to_keep, columns=cols_to_keep, dtype=int)
#             current_model_type_key = "no_contributor_model"
#         current_model_hash_input = f"{discharge_path}-{lat_long_path}-{contrib_path}-{current_forward_start}-{current_forward_end}-{min_completeness_percent_train}-{current_model_type_key}"
#         current_model_hash = hashlib.md5(current_model_hash_input.encode()).hexdigest()
#         current_model_filepath = os.path.join(MODEL_CACHE_DIR, f'model_{current_model_hash}.pkl')
#         trained_model = None
#         if os.path.exists(current_model_filepath):
#             print(f"Loading model from cache: {current_model_filepath}")
#             with open(current_model_filepath, 'rb') as f:
#                 trained_model = pickle.load(f)
#         else:
#             if current_model_type_key == "full_model":
#                 print(f"--- Training model (Full Model) for {current_forward_start}-{current_forward_end} ---")
#                 trained_model = train_full_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
#             elif current_model_type_key == "no_contributor_model":
#                 print(f"--- Training model (No Contributor Model) for {current_forward_start}-{current_forward_end} ---")
#                 trained_model = train_no_contributor_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
#             if trained_model:
#                 with open(current_model_filepath, 'wb') as f:
#                     pickle.dump(trained_model, f)
#                 print(f"Model saved to cache: {current_model_filepath}")
#         if trained_model is None: break
#         df_impute_slice_raw = df_master_imputed.loc[f"{current_forward_start}":f"{current_forward_end}"]
#         df_to_impute = _prepare_imputation_target_data(df_impute_slice_raw, trained_model.discharge_columns, all_feature_cols, trained_model)
#         if df_to_impute.empty:
#             current_forward_start = current_forward_end + 1
#             continue
#         imputed_chunk = trained_model.transform(df_to_impute)
#         imputed_discharge = imputed_chunk[trained_model.discharge_columns]
#         df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
#         print(f"Forward imputation complete for {current_forward_start}-{current_forward_end}.")
#         last_imputed_forward_chunk = imputed_discharge.copy()
#         current_forward_start = current_forward_end + 1

#     print("\n--- Starting Backward Bursting Imputation Loop ---")
#     current_backward_end = impute_backward_start - 1
#     while current_backward_end >= overall_min_year:
#         current_backward_start = max(overall_min_year, current_backward_end - imputation_chunk_size_years + 1)
#         print(f"Backward Bursting: Training on last imputed chunk, Imputing {current_backward_start}-{current_backward_end}")
#         df_train_slice = last_imputed_backward_chunk.merge(df_full_data_with_features[all_feature_cols], left_index=True, right_index=True)
#         df_train_masked, cols_to_keep = _prepare_training_data_for_model(
#             df_train_slice, last_imputed_backward_chunk.columns.tolist(), all_feature_cols, min_completeness_percent_train
#         )
#         if df_train_masked is None: break
#         dist_matrix = build_distance_matrix(df_coords, cols_to_keep)
#         if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
#             conn_matrix = build_connectivity_matrix(df_contrib_filtered, cols_to_keep, station_name_to_vcode)
#             current_model_type_key = "full_model"
#         else:
#             conn_matrix = pd.DataFrame(0, index=cols_to_keep, columns=cols_to_keep, dtype=int)
#             current_model_type_key = "no_contributor_model"
#         current_model_hash_input = f"{discharge_path}-{lat_long_path}-{contrib_path}-{current_backward_start}-{current_backward_end}-{min_completeness_percent_train}-{current_model_type_key}"
#         current_model_hash = hashlib.md5(current_model_hash_input.encode()).hexdigest()
#         current_model_filepath = os.path.join(MODEL_CACHE_DIR, f'model_{current_model_hash}.pkl')
#         trained_model = None
#         if os.path.exists(current_model_filepath):
#             print(f"Loading model from cache: {current_model_filepath}")
#             with open(current_model_filepath, 'rb') as f:
#                 trained_model = pickle.load(f)
#         else:
#             if current_model_type_key == "full_model":
#                 print(f"--- Training model (Full Model) for {current_backward_start}-{current_backward_end} ---")
#                 trained_model = train_full_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
#             elif current_model_type_key == "no_contributor_model":
#                 print(f"--- Training model (No Contributor Model) for {current_backward_start}-{current_backward_end} ---")
#                 trained_model = train_no_contributor_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
#             if trained_model:
#                 with open(current_model_filepath, 'wb') as f:
#                     pickle.dump(trained_model, f)
#                 print(f"Model saved to cache: {current_model_filepath}")
#         if trained_model is None: break
#         df_impute_slice_raw = df_master_imputed.loc[f"{current_backward_start}":f"{current_backward_end}"]
#         df_to_impute = _prepare_imputation_target_data(df_impute_slice_raw, trained_model.discharge_columns, all_feature_cols, trained_model)
#         if df_to_impute.empty:
#             current_backward_end = current_backward_start - 1
#             continue
#         imputed_chunk = trained_model.transform(df_to_impute)
#         imputed_discharge = imputed_chunk[trained_model.discharge_columns]
#         df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
#         print(f"Backward imputation complete for {current_backward_start}-{current_backward_end}.")
#         last_imputed_backward_chunk = imputed_discharge.copy()
#         current_backward_end = current_backward_start - 1

#     output_filename = os.path.join(output_dir, "final_bursting_imputed_data.csv")
#     df_master_imputed[all_discharge_cols_overall].to_csv(output_filename)
#     print(f"\nBursting Imputation complete. Final imputed data saved to: {output_filename}")
#     print(f"Final NaNs remaining in master imputed data: {df_master_imputed[all_discharge_cols_overall].isna().sum().sum()}")
#     return df_master_imputed[all_discharge_cols_overall]


# # --- EVALUATION FUNCTION (MODIFIED) ---
# def evaluate_bursting_pipeline(discharge_path, lat_long_path, contrib_path=None,
#                                overall_min_year=1976, overall_max_year=2016,
#                                min_completeness_percent_train=70.0,
#                                initial_train_window_size=3,
#                                imputation_chunk_size_years=3,
#                                evaluation_dir="evaluation_results",
#                                use_model_cache=True,
#                                target_gap_percentage=10.0): # <-- PARAMETER FOR PERCENTAGE
#     """
#     Evaluates the accuracy of the Bursting imputation pipeline across
#     multiple continuous gap lengths (3, 7, 30, 100 days), aiming for a
#     consistent target percentage of missing data introduced.
#     """
#     os.makedirs(evaluation_dir, exist_ok=True)
#     temporal_features = ['day_of_year_sin', 'day_of_year_cos']
#     gap_lengths_to_test = [3, 7, 30, 100] # Standard gap lengths to evaluate
#     all_results = {} # Dictionary to store metrics for the final table

#     print("--- Starting Evaluation of Bursting Imputation Method ---")
#     print(f"Targeting ~{target_gap_percentage:.1f}% missing data for gap evaluation.")

#     # --- Phase 1: Load and Prepare Original Data (Done ONCE) ---
#     try:
#         # Ensure correct order: discharge, lat_long, contrib
#         df_discharge_raw, df_contrib_filtered, df_coords, vcode_to_station_name, station_name_to_vcode = \
#             load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)

#         if df_discharge_raw is None or df_discharge_raw.empty:
#             print("Error: Discharge data is empty or failed to load. Exiting evaluation.")
#             return None
#     except Exception as e:
#         print(f"FATAL ERROR during initial data loading/preprocessing for evaluation: {e}")
#         return None

#     df_full_data_with_features = add_temporal_features(df_discharge_raw)

#     # Trim data to the specified evaluation period
#     df_full_data_with_features = df_full_data_with_features.loc[
#         (df_full_data_with_features.index.year >= overall_min_year) &
#         (df_full_data_with_features.index.year <= overall_max_year)
#     ].copy()
#     print(f"Dataset trimmed to range: {overall_min_year}-{overall_max_year}. Shape: {df_full_data_with_features.shape}")

#     if df_full_data_with_features.empty:
#         print("Error: Dataset is empty after trimming to specified year range. Exiting evaluation.")
#         return None

#     # Identify the final list of discharge columns after loading and filtering
#     all_discharge_cols_overall = [col for col in df_full_data_with_features.columns if col not in temporal_features]

#     if not all_discharge_cols_overall:
#         print("Error: No discharge columns identified after preprocessing. Exiting evaluation.")
#         return None

#     # --- START EVALUATION LOOP FOR EACH GAP LENGTH ---
#     for gap_length in gap_lengths_to_test:
#         print(f"\n{'-'*20} EVALUATING FOR {gap_length}-DAY GAPS {'-'*20}")

#         # --- Phase 2: Create Masked Data (Gaps) for Evaluation ---
#         print(f"Creating {gap_length}-day continuous gaps (target ~{target_gap_percentage:.1f}%)...")
#         # Use a fresh copy of the original data for each gap length test
#         df_original_truth_copy = df_full_data_with_features.copy()

#         # Call the modified gap creation function using the target percentage
#         gap_info = create_contiguous_segment_gaps(
#             df_original_truth_copy,
#             all_discharge_cols_overall, # Pass the actual discharge columns
#             gap_lengths=[gap_length],   # Process one gap length at a time
#             target_gap_percentage=target_gap_percentage # Pass the desired percentage
#             # random_seed uses the default value from the function
#         )

#         # Check if gap creation was successful for this length
#         if gap_length not in gap_info or 'gapped_data' not in gap_info[gap_length]:
#             print(f"Warning: Failed to create gaps for length {gap_length}. Skipping.")
#             all_results[f"{gap_length}-day_gap"] = {'RMSE': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
#             continue

#         df_to_impute_for_pipelines = gap_info[gap_length]['gapped_data']
#         print(f"Total NaNs in data sent to pipeline (gap length {gap_length}): {df_to_impute_for_pipelines[all_discharge_cols_overall].isna().sum().sum()}")

#         # --- Phase 3: Run Bursting Imputation Pipeline ---
#         print("\n--- Running Bursting Imputation ---")
#         # Create a specific output directory for this run's artifacts (like the imputed file)
#         run_output_dir = os.path.join(evaluation_dir, f"bursting_results_gap_{gap_length}")

#         imputed_data_bursting = run_bursting_imputation_pipeline(
#             # Pass pre-loaded and gapped data
#             df_initial_masked=df_to_impute_for_pipelines,
#             df_full_data_with_features_param=df_full_data_with_features, # Pass original for reference
#             all_discharge_cols_overall_param=all_discharge_cols_overall,
#             df_contrib_filtered_param=df_contrib_filtered,
#             df_coords_param=df_coords,
#             station_name_to_vcode_param=station_name_to_vcode,
#             # Pass paths for hashing/logic checks
#             discharge_path=discharge_path,
#             lat_long_path=lat_long_path,
#             contrib_path=contrib_path,
#             # Pass pipeline control parameters
#             overall_min_year=overall_min_year,
#             overall_max_year=overall_max_year,
#             initial_train_window_size=initial_train_window_size,
#             imputation_chunk_size_years=imputation_chunk_size_years,
#             min_completeness_percent_train=min_completeness_percent_train,
#             output_dir=run_output_dir,
#             use_model_cache=use_model_cache # Pass the caching flag
#         )

#         # Handle potential pipeline failure for this gap length
#         if imputed_data_bursting is None:
#             print(f"Error: Bursting imputation pipeline failed for gap length {gap_length}. Skipping.")
#             all_results[f"{gap_length}-day_gap"] = {'RMSE': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
#             continue

#         # --- Phase 4: Compare Accuracy ---
#         print(f"\n--- Comparing Imputation Accuracy for {gap_length}-day gaps ---")

#         # Reindex imputed data to match original shape for easier comparison
#         imputed_data_aligned = imputed_data_bursting.reindex(
#             index=df_full_data_with_features.index,
#             columns=all_discharge_cols_overall
#         )

#         # Define the output path for the detailed true vs predicted values CSV
#         gap_data_filename = os.path.join(evaluation_dir, f"gap_true_vs_predicted_{gap_length}days.csv")

#         # Calculate metrics using the dedicated helper function
#         metrics = _evaluate_at_gap_locations(
#             df_full_data_with_features,  # Original data (ground truth)
#             df_to_impute_for_pipelines, # Data with artificial gaps
#             imputed_data_aligned,       # Imputed data
#             all_discharge_cols_overall, # List of columns to evaluate
#             output_csv_path=gap_data_filename # Path to save detailed gap data
#         )

#         # Store the calculated metrics
#         all_results[f"{gap_length}-day_gap"] = metrics
#         print(f"Metrics for {gap_length}-day gap: {metrics}")

#     # --- FINAL SUMMARY ---
#     print("\n" + "="*50)
#     print("--- FINAL EVALUATION SUMMARY TABLE ---")
#     print("="*50)

#     # Create and display the summary DataFrame
#     results_df = pd.DataFrame.from_dict(all_results, orient='index')
#     results_df.columns.name = "Metric"
#     results_df.index.name = "Gap Scenario" # Changed index name slightly

#     # Print nicely formatted table to console
#     print(results_df.to_string(float_format="%.4f"))

#     # Save the summary table to a CSV file
#     results_file = os.path.join(evaluation_dir, "imputation_gap_comparison_metrics.csv")
#     results_df.to_csv(results_file)
#     print(f"\nFinal summary table saved to {results_file}")

#     return results_df

# # --- Example of how to run the evaluation ---
# if __name__ == "__main__":
#     # FIX: Use r"..." for Windows paths to prevent syntax errors
#     DISCHARGE_DATA_PATH = r"C:\Users\ethan\Downloads\cauv_discharge.csv"
#     LAT_LONG_DATA_PATH = r"c:\Users\ethan\Downloads\lat_long_cauv.csv"
#     CONTRIBUTOR_DATA_PATH = None # Set to None if no contributor data

#     EVAL_OVERALL_MIN_YEAR = 1980 
#     EVAL_OVERALL_MAX_YEAR = 1990 
#     EVAL_MIN_COMPLETENESS_TRAIN = 70.0
#     EVAL_INITIAL_TRAIN_WINDOW_SIZE = 3 
#     EVAL_IMPUTATION_CHUNK_SIZE = 3 
#     EVAL_OUTPUT_DIR = "evaluation_results" 
#     TARGET_GAP_PERCENTAGE = 8.0

#     evaluation_results_df = evaluate_bursting_pipeline(
#         discharge_path=DISCHARGE_DATA_PATH,
#         lat_long_path=LAT_LONG_DATA_PATH,
#         contrib_path=CONTRIBUTOR_DATA_PATH,
#         overall_min_year=EVAL_OVERALL_MIN_YEAR,
#         overall_max_year=EVAL_OVERALL_MAX_YEAR,
#         min_completeness_percent_train=EVAL_MIN_COMPLETENESS_TRAIN,
#         initial_train_window_size=EVAL_INITIAL_TRAIN_WINDOW_SIZE,
#         imputation_chunk_size_years=EVAL_IMPUTATION_CHUNK_SIZE,
#         evaluation_dir=EVAL_OUTPUT_DIR
#     )

#     if evaluation_results_df is not None:
#         print("\n--- Final Evaluation Summary ---")
#         print(evaluation_results_df.to_string(float_format="%.4f"))
#     else:
#         print("\nEvaluation process did not complete successfully.")



import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timedelta
import pickle
import hashlib
import math # Make sure math is imported
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import actual implementations from your files
from missforest_imputer import ModifiedMissForest
from model_configurations import train_full_model, train_no_contributor_model
# Ensure you have these functions in utils.py
from utils import (
    load_and_preprocess_data,
    add_temporal_features,
    build_distance_matrix,
    build_connectivity_matrix,
    create_contiguous_segment_gaps
)

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# Directory to save/load trained models
MODEL_CACHE_DIR = "trained_models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


# --- KLING-GUPTA EFFICIENCY FUNCTION ---

def modified_kling_gupta_efficiency(simulated, observed):
    """
    Calculates the Modified Kling-Gupta Efficiency (KGE') score.
    """
    simulated = np.asarray(simulated)
    observed = np.asarray(observed)
    valid_indices = ~np.isnan(simulated) & ~np.isnan(observed)
    simulated = simulated[valid_indices]
    observed = observed[valid_indices]
    if len(simulated) == 0:
        print("Warning: No valid data points found for KGE calculation.")
        return np.nan
    if np.std(simulated) == 0 or np.std(observed) == 0:
        r = 0.0
    else:
        # Handle potential RuntimeWarning for invalid correlation
        with np.errstate(invalid='ignore'):
             r = np.corrcoef(simulated, observed)[0, 1]
        if np.isnan(r): r = 0.0 # Treat NaN correlation as 0

    mean_obs = np.mean(observed)
    mean_sim = np.mean(simulated)
    if mean_obs == 0:
        beta = 1.0 if mean_sim == 0 else np.nan
    else:
        beta = mean_sim / mean_obs
    std_obs = np.std(observed)
    std_sim = np.std(simulated)
    cv_obs = (std_obs / mean_obs) if mean_obs != 0 else np.nan
    cv_sim = (std_sim / mean_sim) if mean_sim != 0 else np.nan
    if np.isnan(cv_obs) or cv_obs == 0:
        gamma = 1.0 if np.isnan(cv_sim) or cv_sim == 0 else np.nan
    else:
        gamma = cv_sim / cv_obs
    r_error = (r - 1) ** 2 if not np.isnan(r) else 1.0
    beta_error = (beta - 1) ** 2 if not np.isnan(beta) else 1.0
    gamma_error = (gamma - 1) ** 2 if not np.isnan(gamma) else 1.0
    # Handle potential RuntimeWarning from sqrt of negative number if errors are large
    error_sum = r_error + beta_error + gamma_error
    if error_sum < 0: error_sum = max(r_error, beta_error, gamma_error) # Use largest single error if sum is somehow negative
    kge_prime = 1 - np.sqrt(error_sum)
    return kge_prime


# --- HELPER FUNCTION FOR EVALUATION ---

def _evaluate_at_gap_locations(df_original, df_gapped, df_imputed, discharge_cols, output_csv_path=None):
    """
    Calculates metrics only at the locations of the artificially created gaps.
    Optionally saves the true vs. predicted values (with date and station context) to a CSV file.
    """
    y_true_eval, y_pred_eval = [], []
    all_gap_data_rows = []

    for station in discharge_cols:
        if station not in df_imputed.columns or station not in df_original.columns:
            continue

        gap_mask = df_gapped[station].isnull() & df_original[station].notna()

        if gap_mask.sum() > 0:
            predicted_vals_series = df_imputed.loc[gap_mask, station]
            true_vals_series = df_original.loc[gap_mask, station]

            for date_index, true_val in true_vals_series.items():
                pred_val = predicted_vals_series.get(date_index)
                all_gap_data_rows.append({
                    'date': date_index,
                    'station': station,
                    'true_value': true_val,
                    'predicted_value': pred_val
                })
                y_pred_eval.append(pred_val)
                y_true_eval.append(true_val)

    if not y_true_eval:
        print("Warning: No valid gap data found for evaluation.")
        return {'RMSE': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    y_true = np.array(y_true_eval)
    y_pred = np.array(y_pred_eval)
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]

    if y_true.size == 0:
        print("Warning: No valid data points left after filtering NaNs from imputation.")
        return {'RMSE': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    if output_csv_path:
        try:
            df_gap_data = pd.DataFrame(all_gap_data_rows)
            # Filter the detailed df based on valid indices used for metrics
            valid_gap_data_df = df_gap_data.iloc[np.where(valid_indices)[0]].copy()
            valid_gap_data_df = valid_gap_data_df[['date', 'station', 'true_value', 'predicted_value']]
            valid_gap_data_df.to_csv(output_csv_path, index=False)
            print(f"Saved gap true vs. predicted data (with context) to {output_csv_path}")
        except Exception as e:
            print(f"Warning: Could not save gap data to {output_csv_path}. Error: {e}")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2) # Use mean of true values
    nse = 1 - (numerator / denominator) if denominator != 0 else -np.inf # Use -inf if denominator is 0
    kge = modified_kling_gupta_efficiency(y_pred, y_true)

    return {'RMSE': rmse, 'MAE': mae, 'NSE': nse, 'KGE': kge}


# --- HELPER FUNCTIONS FOR PIPELINE ---

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
        # Only consider discharge columns present in this slice
        period_discharge_cols = [col for col in discharge_cols if col in period_data.columns]
        if not period_discharge_cols: continue
        period_discharge_data = period_data[period_discharge_cols]
        total_cells_in_period = period_discharge_data.shape[0] * period_discharge_data.shape[1]
        if total_cells_in_period == 0:
            current_completeness = 0.0
        else:
            non_na_cells_in_period = period_discharge_data.notna().sum().sum()
            current_completeness = (non_na_cells_in_period / total_cells_in_period) * 100
        if current_completeness >= min_overall_completeness_percent:
            if current_completeness > max_completeness_found:
                max_completeness_found = current_completeness
                best_period = (start_year, end_year)
                min_non_na_count = int(min_overall_completeness_percent / 100 * len(period_data))
                # Check completeness based on period_discharge_data
                best_stations = period_discharge_data.columns[
                    period_discharge_data.notna().sum() >= min_non_na_count
                ].tolist()
    if best_period == (None, None):
        print(f"Error: No {window_size}-year period found with at least {min_overall_completeness_percent}% completeness. Insufficient data for training.")
        return None, None, []
    else:
        print(f"\nBest training period identified: {best_period[0]}-{best_period[1]} with {max_completeness_found:.2f}% completeness.")
        # Report based on the original full list vs. found best stations
        print(f"Stations with sufficient data in best period: {len(best_stations)} out of {len(discharge_cols)}")
    return best_period[0], best_period[1], best_stations

def _prepare_training_data_for_model(
    df_train_slice, all_discharge_cols, all_feature_cols, min_completeness_percent_train
):
    """
    Helper function to prepare training data. Handles column filtering,
    historical mean initialization, and simulated missingness.
    """
    min_non_na_count = int(min_completeness_percent_train / 100 * len(df_train_slice))
    # Filter based on discharge columns *available in this slice*
    slice_discharge_cols = [col for col in all_discharge_cols if col in df_train_slice.columns]
    if not slice_discharge_cols:
         print("Warning: No discharge columns found in this training slice.")
         return None, None

    cols_to_keep_discharge = []
    # Check completeness within the slice's discharge columns
    df_slice_discharge_data = df_train_slice[slice_discharge_cols]
    if not df_slice_discharge_data.empty:
        cols_to_keep_discharge = df_slice_discharge_data.columns[
            df_slice_discharge_data.notna().sum() >= min_non_na_count
        ].tolist()

    if not cols_to_keep_discharge:
        print("Warning: No discharge columns met completeness criteria in this training slice.")
        return None, None

    # Filter feature cols to only those present in the slice
    slice_feature_cols = [col for col in all_feature_cols if col in df_train_slice.columns]
    final_cols_for_imputer = list(set(cols_to_keep_discharge + slice_feature_cols))
    df_train_period_filtered = df_train_slice[final_cols_for_imputer].copy()
    df_train_discharge_only = df_train_period_filtered[cols_to_keep_discharge].copy()

    print("Applying historical (day-of-year) mean initialization...")
    df_train_slice_for_means = df_train_period_filtered.copy()
    if 'doy' not in df_train_slice_for_means.columns:
         df_train_slice_for_means['doy'] = df_train_slice_for_means.index.dayofyear
    # Calculate means only for columns we are keeping
    historical_means = df_train_slice_for_means.groupby('doy')[cols_to_keep_discharge].mean()
    doy_series = df_train_slice_for_means['doy']

    for col in cols_to_keep_discharge:
        if df_train_discharge_only[col].isnull().all():
            df_train_discharge_only[col] = 0.0
        else:
            doy_map = doy_series.map(historical_means[col])
            df_train_discharge_only[col] = df_train_discharge_only[col].fillna(doy_map)
            if df_train_discharge_only[col].isnull().any():
                col_mean = df_train_discharge_only[col].mean()
                if pd.isna(col_mean):
                    col_mean = 0.0
                df_train_discharge_only[col] = df_train_discharge_only[col].fillna(col_mean)
    print("Historical mean initialization complete.")
    np.random.seed(42)
    # Create mask based on the shape of df_train_discharge_only
    train_mask_simulated = np.random.rand(*df_train_discharge_only.shape) < 0.1
    df_train_masked_for_model = df_train_period_filtered.copy()
    # Apply mask using the correct columns
    df_train_masked_for_model[cols_to_keep_discharge] = df_train_discharge_only.mask(train_mask_simulated)
    print(f"Simulated 10% random missingness in training data.")
    return df_train_masked_for_model, cols_to_keep_discharge


def _prepare_imputation_target_data(
    df_raw_slice, model_discharge_cols, all_feature_cols, trained_model
):
    """
    Helper function to prepare data for imputation using a trained model.
    """
    # Filter feature cols to only those present in the raw slice
    slice_feature_cols = [col for col in all_feature_cols if col in df_raw_slice.columns]
    cols_for_imputation_target = list(set(model_discharge_cols + slice_feature_cols))
    df_to_impute = df_raw_slice.reindex(columns=cols_for_imputation_target).copy()
    for col in model_discharge_cols:
        if col not in df_to_impute.columns:
            print(f"Warning: Column '{col}' expected by model not in target slice. Adding as NaN.")
            df_to_impute[col] = np.nan
        # Check trained_model.col_means safely using .get()
        model_col_mean = trained_model.col_means.get(col, np.nan) # Default to NaN if col not in means
        if df_to_impute[col].isnull().all() and np.isnan(model_col_mean):
            df_to_impute[col] = 0.0
    return df_to_impute


# --- BURSTING IMPUTATION PIPELINE (FIXED Definition) ---
def run_bursting_imputation_pipeline(
    df_initial_masked=None,
    df_full_data_with_features_param=None,
    all_discharge_cols_overall_param=None,
    df_contrib_filtered_param=None,
    df_coords_param=None,
    station_name_to_vcode_param=None,
    discharge_path=None, lat_long_path=None, contrib_path=None,
    initial_train_window_size=5,
    imputation_chunk_size_years=5,
    overall_min_year=1976, overall_max_year=2016,
    min_completeness_percent_train=70.0,
    output_dir="bursting_imputed_results",
    use_model_cache=True  # <-- FIXED: Added parameter to definition
):
    """
    Implements a "bursting" imputation pipeline with bidirectional imputation
    and model caching. Can accept pre-loaded data for evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("--- Starting Bursting Imputation Pipeline ---")
    df_full_data_with_features = None
    all_discharge_cols_overall = None
    df_contrib_filtered = None
    df_coords = None
    station_name_to_vcode = None
    df_master_imputed = None
    if df_initial_masked is not None:
        print("Using pre-loaded and pre-masked initial data for evaluation.")
        df_master_imputed = df_initial_masked.copy()
        if (df_full_data_with_features_param is None or
            all_discharge_cols_overall_param is None or
            df_coords_param is None):
            raise ValueError("When df_initial_masked is provided, all helper dataframes must also be provided.")
        df_full_data_with_features = df_full_data_with_features_param
        all_discharge_cols_overall = all_discharge_cols_overall_param
        df_contrib_filtered = df_contrib_filtered_param
        df_coords = df_coords_param
        station_name_to_vcode = station_name_to_vcode_param
    else:
        print("Loading data from paths for standalone run.")
        try:
            df_discharge_raw, df_contrib_filtered, df_coords, vcode_to_station_name, station_name_to_vcode = \
                load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
            if df_discharge_raw is None:
                print("FATAL ERROR: Failed to load data. Exiting.")
                return None
        except Exception as e:
            print(f"FATAL ERROR during initial data loading/preprocessing: {e}")
            return None
        all_discharge_cols_overall = [col for col in df_discharge_raw.columns if not col.startswith('day_of_year_')] # Get discharge cols before adding features
        df_full_data_with_features = add_temporal_features(df_discharge_raw)
        df_master_imputed = df_full_data_with_features.copy()

    temporal_features = ['day_of_year_sin', 'day_of_year_cos']
    # Ensure only features present in the loaded data are used
    all_feature_cols = [f for f in temporal_features if f in df_full_data_with_features.columns]
    print(f"Using features: {all_feature_cols}")
    # Use the definitive list of discharge columns from parameters if available, else from loaded data
    if all_discharge_cols_overall_param:
        all_discharge_cols_overall = all_discharge_cols_overall_param
    print(f"Master imputed DataFrame initialized. Total NaNs in discharge cols: {df_master_imputed[all_discharge_cols_overall].isna().sum().sum()}")

    print("\nFinding the best initial training period...")
    best_initial_train_start, best_initial_train_end, initial_train_stations_in_period = find_best_training_period(
        df_full_data_with_features, overall_min_year, overall_max_year,
        initial_train_window_size, min_completeness_percent_train
    )
    if best_initial_train_start is None:
        return None
    print(f"Selected initial training period: {best_initial_train_start}-{best_initial_train_end}")

    print("\nTraining initial model on the best period...")
    df_initial_train_slice = df_master_imputed.loc[
        (df_master_imputed.index.year >= best_initial_train_start) &
        (df_master_imputed.index.year <= best_initial_train_end)
    ]
    # Pass the stations found specifically in the best period
    df_initial_train_masked, cols_to_keep_initial = _prepare_training_data_for_model(
        df_initial_train_slice, initial_train_stations_in_period, all_feature_cols, min_completeness_percent_train
    )
    if df_initial_train_masked is None or not cols_to_keep_initial:
        print("Error: Initial training data preparation failed or resulted in no valid columns. Cannot train initial model.")
        return None

    # Build matrices only for the columns kept for initial training
    dist_matrix_initial = build_distance_matrix(df_coords, cols_to_keep_initial)
    if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
        conn_matrix_initial = build_connectivity_matrix(df_contrib_filtered, cols_to_keep_initial, station_name_to_vcode)
        model_type_key = "full_model"
    else:
        print("No valid contribution data provided or found. Building a zero connectivity matrix for initial training.")
        conn_matrix_initial = pd.DataFrame(0, index=cols_to_keep_initial, columns=cols_to_keep_initial, dtype=int)
        model_type_key = "no_contributor_model"

    # Hashing needs to be robust to file path differences
    input_hash_string = f"{os.path.basename(discharge_path or '')}-{os.path.basename(lat_long_path or '')}-{os.path.basename(contrib_path or '')}-{best_initial_train_start}-{best_initial_train_end}-{min_completeness_percent_train}-{model_type_key}"
    model_hash_initial = hashlib.md5(input_hash_string.encode()).hexdigest()
    model_filepath_initial = os.path.join(MODEL_CACHE_DIR, f'model_{model_hash_initial}.pkl')

    trained_model_initial = None
    if use_model_cache and os.path.exists(model_filepath_initial):
        try:
            print(f"Loading initial model from cache: {model_filepath_initial}")
            with open(model_filepath_initial, 'rb') as f:
                trained_model_initial = pickle.load(f)
            # Verify the loaded model has the expected columns
            if not set(trained_model_initial.discharge_columns) == set(cols_to_keep_initial):
                 print("Warning: Cached model columns do not match current training columns. Retraining.")
                 trained_model_initial = None # Force retraining
        except Exception as e:
            print(f"Warning: Failed to load model from cache. Retraining. Error: {e}")
            trained_model_initial = None

    if trained_model_initial is None: # Train if not loaded or if cache invalid
        if not use_model_cache:
             print("Training initial model from scratch (cache disabled)...")
        else:
             print("Training initial model from scratch (cache not found or invalid)...")

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

        if trained_model_initial and use_model_cache:
            try:
                with open(model_filepath_initial, 'wb') as f:
                    pickle.dump(trained_model_initial, f)
                print(f"Initial model saved to cache: {model_filepath_initial}")
            except Exception as e:
                print(f"Warning: Failed to save model to cache. Error: {e}")

    if trained_model_initial is None:
        print("Error: Initial model training failed. Exiting pipeline.")
        return None
    # Use discharge columns from the actually trained/loaded model
    model_discharge_cols_initial = trained_model_initial.discharge_columns

    print(f"\nPerforming initial forward imputation...")
    impute_forward_start = best_initial_train_end + 1
    impute_forward_end = min(overall_max_year, impute_forward_start + imputation_chunk_size_years - 1)

    print(f"Imputing the initial training chunk ({best_initial_train_start}-{best_initial_train_end}) itself...")
    # Prepare the target slice using the model's expected columns
    df_initial_train_to_impute = _prepare_imputation_target_data(
        df_initial_train_slice, model_discharge_cols_initial, all_feature_cols, trained_model_initial
    )
    if not df_initial_train_to_impute.empty:
        imputed_initial_train_chunk = trained_model_initial.transform(df_initial_train_to_impute)
        # Only update columns that were actually imputed by this model
        imputed_initial_discharge = imputed_initial_train_chunk[model_discharge_cols_initial]
        df_master_imputed.loc[imputed_initial_discharge.index, imputed_initial_discharge.columns] = imputed_initial_discharge.values
        last_imputed_forward_chunk = imputed_initial_discharge.copy()
        last_imputed_backward_chunk = imputed_initial_discharge.copy()
    else:
        print("Warning: Initial training chunk was empty after preparation for transform. Cannot initialize imputed chunks.")
        # Fallback: Initialize with potentially empty/NaN slice if needed, or handle error
        slice_cols = [c for c in model_discharge_cols_initial if c in df_initial_train_slice.columns]
        last_imputed_forward_chunk = df_initial_train_slice[slice_cols].copy()
        last_imputed_backward_chunk = df_initial_train_slice[slice_cols].copy()


    if impute_forward_start <= impute_forward_end:
        df_impute_slice_raw = df_master_imputed.loc[f"{impute_forward_start}":f"{impute_forward_end}"]
        df_to_impute = _prepare_imputation_target_data(
            df_impute_slice_raw, model_discharge_cols_initial, all_feature_cols, trained_model_initial
        )
        if not df_to_impute.empty:
            imputed_chunk = trained_model_initial.transform(df_to_impute)
            imputed_discharge = imputed_chunk[model_discharge_cols_initial]
            df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
            print(f"Initial forward imputation complete for {impute_forward_start}-{impute_forward_end}.")
            last_imputed_forward_chunk = imputed_discharge.copy()
        else:
             print(f"Warning: Slice for initial forward imputation ({impute_forward_start}-{impute_forward_end}) was empty after preparation.")


    print(f"\nPerforming initial backward imputation...")
    impute_backward_end = best_initial_train_start - 1
    impute_backward_start = max(overall_min_year, impute_backward_end - imputation_chunk_size_years + 1)

    if impute_backward_start <= impute_backward_end:
        df_impute_slice_raw = df_master_imputed.loc[f"{impute_backward_start}":f"{impute_backward_end}"]
        df_to_impute = _prepare_imputation_target_data(
            df_impute_slice_raw, model_discharge_cols_initial, all_feature_cols, trained_model_initial
        )
        if not df_to_impute.empty:
            imputed_chunk = trained_model_initial.transform(df_to_impute)
            imputed_discharge = imputed_chunk[model_discharge_cols_initial]
            df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
            print(f"Initial backward imputation complete for {impute_backward_start}-{impute_backward_end}.")
            last_imputed_backward_chunk = imputed_discharge.copy()
        else:
             print(f"Warning: Slice for initial backward imputation ({impute_backward_start}-{impute_backward_end}) was empty after preparation.")


    print("\n--- Starting Forward Bursting Imputation Loop ---")
    current_forward_start = impute_forward_end + 1
    while current_forward_start <= overall_max_year:
        current_forward_end = min(overall_max_year, current_forward_start + imputation_chunk_size_years - 1)
        print(f"Forward Bursting: Training on last imputed chunk ending {last_imputed_forward_chunk.index.max().year if not last_imputed_forward_chunk.empty else 'N/A'}, Imputing {current_forward_start}-{current_forward_end}")

        # Ensure last_imputed chunk is not empty before merging
        if last_imputed_forward_chunk.empty:
            print("Warning: Last imputed forward chunk is empty. Cannot train model. Stopping forward loop.")
            break

        # Merge features; handle cases where features might be missing
        df_train_slice = last_imputed_forward_chunk.copy()
        for feat in all_feature_cols:
             if feat in df_full_data_with_features.columns:
                  df_train_slice = df_train_slice.merge(df_full_data_with_features[[feat]], left_index=True, right_index=True, how='left')
             else:
                  print(f"Warning: Feature column '{feat}' not found in df_full_data_with_features.")

        # Train on the columns present in the last imputed chunk
        df_train_masked, cols_to_keep = _prepare_training_data_for_model(
            df_train_slice, last_imputed_forward_chunk.columns.tolist(), all_feature_cols, min_completeness_percent_train
        )
        if df_train_masked is None or not cols_to_keep:
             print(f"Warning: Training data prep failed for forward step {current_forward_start}-{current_forward_end}. Skipping chunk.")
             current_forward_start = current_forward_end + 1
             continue

        dist_matrix = build_distance_matrix(df_coords, cols_to_keep)
        if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
            conn_matrix = build_connectivity_matrix(df_contrib_filtered, cols_to_keep, station_name_to_vcode)
            current_model_type_key = "full_model"
        else:
            conn_matrix = pd.DataFrame(0, index=cols_to_keep, columns=cols_to_keep, dtype=int)
            current_model_type_key = "no_contributor_model"

        current_hash_string = f"{os.path.basename(discharge_path or '')}-{os.path.basename(lat_long_path or '')}-{os.path.basename(contrib_path or '')}-Fwd-{current_forward_start}-{current_forward_end}-{min_completeness_percent_train}-{current_model_type_key}"
        current_model_hash = hashlib.md5(current_hash_string.encode()).hexdigest()
        current_model_filepath = os.path.join(MODEL_CACHE_DIR, f'model_{current_model_hash}.pkl')

        trained_model = None
        if use_model_cache and os.path.exists(current_model_filepath):
            try:
                print(f"Loading model from cache: {current_model_filepath}")
                with open(current_model_filepath, 'rb') as f:
                    trained_model = pickle.load(f)
                if not set(trained_model.discharge_columns) == set(cols_to_keep):
                     print("Warning: Cached model columns mismatch. Retraining.")
                     trained_model = None
            except Exception as e:
                print(f"Warning: Failed to load model from cache. Retraining. Error: {e}")
                trained_model = None

        if trained_model is None:
            if not use_model_cache: print(f"--- Training model (cache disabled) for Fwd {current_forward_start}-{current_forward_end} ---")
            else: print(f"--- Training model (cache not found or invalid) for Fwd {current_forward_start}-{current_forward_end} ---")

            if current_model_type_key == "full_model":
                trained_model = train_full_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
            elif current_model_type_key == "no_contributor_model":
                trained_model = train_no_contributor_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)

            if trained_model and use_model_cache:
                try:
                    with open(current_model_filepath, 'wb') as f: pickle.dump(trained_model, f)
                    print(f"Model saved to cache: {current_model_filepath}")
                except Exception as e: print(f"Warning: Failed to save model to cache. Error: {e}")

        if trained_model is None:
             print(f"Error: Model training failed for forward step {current_forward_start}-{current_forward_end}. Stopping forward loop.")
             break

        # Impute the *next* chunk
        df_impute_slice_raw = df_master_imputed.loc[f"{current_forward_start}":f"{current_forward_end}"]
        df_to_impute = _prepare_imputation_target_data(df_impute_slice_raw, trained_model.discharge_columns, all_feature_cols, trained_model)
        if df_to_impute.empty:
            print(f"Warning: Slice for forward imputation ({current_forward_start}-{current_forward_end}) was empty after preparation. Skipping chunk.")
            current_forward_start = current_forward_end + 1
            continue

        imputed_chunk = trained_model.transform(df_to_impute)
        imputed_discharge = imputed_chunk[trained_model.discharge_columns]
        df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
        print(f"Forward imputation complete for {current_forward_start}-{current_forward_end}.")
        last_imputed_forward_chunk = imputed_discharge.copy()
        current_forward_start = current_forward_end + 1

    # --- Backward Loop ---
    print("\n--- Starting Backward Bursting Imputation Loop ---")
    current_backward_end = impute_backward_start - 1
    while current_backward_end >= overall_min_year:
        current_backward_start = max(overall_min_year, current_backward_end - imputation_chunk_size_years + 1)
        print(f"Backward Bursting: Training on last imputed chunk ending {last_imputed_backward_chunk.index.min().year if not last_imputed_backward_chunk.empty else 'N/A'}, Imputing {current_backward_start}-{current_backward_end}")

        if last_imputed_backward_chunk.empty:
            print("Warning: Last imputed backward chunk is empty. Cannot train model. Stopping backward loop.")
            break

        df_train_slice = last_imputed_backward_chunk.copy()
        for feat in all_feature_cols:
             if feat in df_full_data_with_features.columns:
                  df_train_slice = df_train_slice.merge(df_full_data_with_features[[feat]], left_index=True, right_index=True, how='left')
             else:
                  print(f"Warning: Feature column '{feat}' not found in df_full_data_with_features.")

        df_train_masked, cols_to_keep = _prepare_training_data_for_model(
            df_train_slice, last_imputed_backward_chunk.columns.tolist(), all_feature_cols, min_completeness_percent_train
        )
        if df_train_masked is None or not cols_to_keep:
            print(f"Warning: Training data prep failed for backward step {current_backward_start}-{current_backward_end}. Skipping chunk.")
            current_backward_end = current_backward_start - 1
            continue

        dist_matrix = build_distance_matrix(df_coords, cols_to_keep)
        if contrib_path and df_contrib_filtered is not None and not df_contrib_filtered.empty:
            conn_matrix = build_connectivity_matrix(df_contrib_filtered, cols_to_keep, station_name_to_vcode)
            current_model_type_key = "full_model"
        else:
            conn_matrix = pd.DataFrame(0, index=cols_to_keep, columns=cols_to_keep, dtype=int)
            current_model_type_key = "no_contributor_model"

        current_hash_string = f"{os.path.basename(discharge_path or '')}-{os.path.basename(lat_long_path or '')}-{os.path.basename(contrib_path or '')}-Bwd-{current_backward_start}-{current_backward_end}-{min_completeness_percent_train}-{current_model_type_key}"
        current_model_hash = hashlib.md5(current_hash_string.encode()).hexdigest()
        current_model_filepath = os.path.join(MODEL_CACHE_DIR, f'model_{current_model_hash}.pkl')

        trained_model = None
        if use_model_cache and os.path.exists(current_model_filepath):
             try:
                 print(f"Loading model from cache: {current_model_filepath}")
                 with open(current_model_filepath, 'rb') as f: trained_model = pickle.load(f)
                 if not set(trained_model.discharge_columns) == set(cols_to_keep):
                     print("Warning: Cached model columns mismatch. Retraining.")
                     trained_model = None
             except Exception as e:
                 print(f"Warning: Failed to load model from cache. Retraining. Error: {e}")
                 trained_model = None

        if trained_model is None:
            if not use_model_cache: print(f"--- Training model (cache disabled) for Bwd {current_backward_start}-{current_backward_end} ---")
            else: print(f"--- Training model (cache not found or invalid) for Bwd {current_backward_start}-{current_backward_end} ---")

            if current_model_type_key == "full_model":
                trained_model = train_full_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)
            elif current_model_type_key == "no_contributor_model":
                trained_model = train_no_contributor_model(df_train_masked, dist_matrix, conn_matrix, all_feature_cols)

            if trained_model and use_model_cache:
                try:
                    with open(current_model_filepath, 'wb') as f: pickle.dump(trained_model, f)
                    print(f"Model saved to cache: {current_model_filepath}")
                except Exception as e: print(f"Warning: Failed to save model to cache. Error: {e}")

        if trained_model is None:
             print(f"Error: Model training failed for backward step {current_backward_start}-{current_backward_end}. Stopping backward loop.")
             break

        # Impute the *previous* chunk
        df_impute_slice_raw = df_master_imputed.loc[f"{current_backward_start}":f"{current_backward_end}"]
        df_to_impute = _prepare_imputation_target_data(df_impute_slice_raw, trained_model.discharge_columns, all_feature_cols, trained_model)
        if df_to_impute.empty:
            print(f"Warning: Slice for backward imputation ({current_backward_start}-{current_backward_end}) was empty after preparation. Skipping chunk.")
            current_backward_end = current_backward_start - 1
            continue

        imputed_chunk = trained_model.transform(df_to_impute)
        imputed_discharge = imputed_chunk[trained_model.discharge_columns]
        df_master_imputed.loc[imputed_discharge.index, imputed_discharge.columns] = imputed_discharge.values
        print(f"Backward imputation complete for {current_backward_start}-{current_backward_end}.")
        last_imputed_backward_chunk = imputed_discharge.copy()
        current_backward_end = current_backward_start - 1

    # Final Output - only include original discharge columns
    output_cols = [c for c in all_discharge_cols_overall if c in df_master_imputed.columns]
    output_filename = os.path.join(output_dir, "final_bursting_imputed_data.csv")
    df_master_imputed[output_cols].to_csv(output_filename)
    print(f"\nBursting Imputation complete. Final imputed data saved to: {output_filename}")
    print(f"Final NaNs remaining in master imputed discharge data: {df_master_imputed[output_cols].isna().sum().sum()}")
    return df_master_imputed[output_cols] # Return only original discharge columns


# --- EVALUATION FUNCTION ---
def evaluate_bursting_pipeline(discharge_path, lat_long_path, contrib_path=None,
                               overall_min_year=1976, overall_max_year=2016,
                               min_completeness_percent_train=70.0,
                               initial_train_window_size=3,
                               imputation_chunk_size_years=3,
                               evaluation_dir="evaluation_results",
                               use_model_cache=True,
                               target_overall_gap_percentage=10.0):
    """
    Evaluates the accuracy of the Bursting imputation pipeline across
    multiple continuous gap lengths (3, 7, 30, 100 days), aiming for a
    consistent target overall percentage of missing data introduced.
    """
    os.makedirs(evaluation_dir, exist_ok=True)
    temporal_features = ['day_of_year_sin', 'day_of_year_cos']
    gap_lengths_to_test = [3, 7, 30, 100]
    all_results = {}

    print("--- Starting Evaluation of Bursting Imputation Method ---")
    print(f"Targeting ~{target_overall_gap_percentage:.1f}% OVERALL missing data for gap evaluation.")

    # --- Phase 1: Load and Prepare Original Data (Done ONCE) ---
    try:
        df_discharge_raw, df_contrib_filtered, df_coords, vcode_to_station_name, station_name_to_vcode = \
            load_and_preprocess_data(discharge_path, lat_long_path, contrib_path) # Corrected order
        if df_discharge_raw is None or df_discharge_raw.empty:
            print("Error: Discharge data is empty or failed to load. Exiting evaluation.")
            return None
    except Exception as e:
        print(f"FATAL ERROR during initial data loading/preprocessing for evaluation: {e}")
        return None
    df_full_data_with_features = add_temporal_features(df_discharge_raw)
    df_full_data_with_features = df_full_data_with_features.loc[
        (df_full_data_with_features.index.year >= overall_min_year) &
        (df_full_data_with_features.index.year <= overall_max_year)
    ].copy()
    print(f"Dataset trimmed to range: {overall_min_year}-{overall_max_year}. Shape: {df_full_data_with_features.shape}")
    if df_full_data_with_features.empty:
        print("Error: Dataset is empty after trimming to specified year range. Exiting evaluation.")
        return None
    # Get the list of discharge columns AFTER preprocessing and BEFORE adding features if possible
    # This relies on load_and_preprocess returning the correct columns
    all_discharge_cols_overall = [col for col in df_discharge_raw.columns if col not in temporal_features]
    if not all_discharge_cols_overall:
         # Fallback if raw didn't have discharge cols properly identified
         all_discharge_cols_overall = [col for col in df_full_data_with_features.columns if col not in temporal_features]

    if not all_discharge_cols_overall:
        print("Error: No discharge columns identified after preprocessing. Exiting evaluation.")
        return None

    # --- START EVALUATION LOOP FOR EACH GAP LENGTH ---
    for gap_length in gap_lengths_to_test:
        print(f"\n{'-'*20} EVALUATING FOR {gap_length}-DAY GAPS {'-'*20}")

        # --- Phase 2: Create Masked Data (Gaps) for Evaluation ---
        print(f"Creating {gap_length}-day continuous gaps (target overall ~{target_overall_gap_percentage:.1f}%)...")
        df_original_truth_copy = df_full_data_with_features.copy()
        gap_info = create_contiguous_segment_gaps(
            df_original_truth_copy,
            all_discharge_cols_overall,
            gap_lengths=[gap_length],
            target_overall_gap_percentage=target_overall_gap_percentage
        )
        if gap_length not in gap_info or 'gapped_data' not in gap_info[gap_length]:
            print(f"Warning: Failed to create gaps for length {gap_length}. Skipping.")
            all_results[f"{gap_length}-day_gap"] = {'RMSE': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            continue
        df_to_impute_for_pipelines = gap_info[gap_length]['gapped_data']
        print(f"Total NaNs in data sent to pipeline (gap length {gap_length}): {df_to_impute_for_pipelines[all_discharge_cols_overall].isna().sum().sum()}")

        # --- Phase 3: Run Bursting Imputation Pipeline ---
        print("\n--- Running Bursting Imputation ---")
        run_output_dir = os.path.join(evaluation_dir, f"bursting_results_gap_{gap_length}")
        imputed_data_bursting = run_bursting_imputation_pipeline(
            df_initial_masked=df_to_impute_for_pipelines,
            df_full_data_with_features_param=df_full_data_with_features,
            all_discharge_cols_overall_param=all_discharge_cols_overall,
            df_contrib_filtered_param=df_contrib_filtered,
            df_coords_param=df_coords,
            station_name_to_vcode_param=station_name_to_vcode,
            discharge_path=discharge_path, # Pass paths for hashing
            lat_long_path=lat_long_path,
            contrib_path=contrib_path,
            overall_min_year=overall_min_year,
            overall_max_year=overall_max_year,
            initial_train_window_size=initial_train_window_size,
            imputation_chunk_size_years=imputation_chunk_size_years,
            min_completeness_percent_train=min_completeness_percent_train,
            output_dir=run_output_dir,
            use_model_cache=use_model_cache
        )
        if imputed_data_bursting is None:
            print(f"Error: Bursting imputation pipeline failed for gap length {gap_length}. Skipping.")
            all_results[f"{gap_length}-day_gap"] = {'RMSE': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            continue

        # --- Phase 4: Compare Accuracy ---
        print(f"\n--- Comparing Imputation Accuracy for {gap_length}-day gaps ---")
        # Ensure alignment, using only the original discharge columns for evaluation
        imputed_data_aligned = imputed_data_bursting.reindex(
            index=df_full_data_with_features.index,
            columns=all_discharge_cols_overall # Use the definitive list
        )
        gap_data_filename = os.path.join(evaluation_dir, f"gap_true_vs_predicted_{gap_length}days.csv")
        metrics = _evaluate_at_gap_locations(
            df_full_data_with_features, # Original data including features for index alignment
            df_to_impute_for_pipelines, # Gapped data
            imputed_data_aligned,       # Imputed data aligned to original
            all_discharge_cols_overall, # Evaluate only on discharge columns
            output_csv_path=gap_data_filename
        )
        all_results[f"{gap_length}-day_gap"] = metrics
        print(f"Metrics for {gap_length}-day gap: {metrics}")

    # --- FINAL SUMMARY ---
    print("\n" + "="*50)
    print("--- FINAL EVALUATION SUMMARY TABLE ---")
    print("="*50)
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    results_df.columns.name = "Metric"
    results_df.index.name = "Gap Scenario"
    print(results_df.to_string(float_format="%.4f"))
    results_file = os.path.join(evaluation_dir, "imputation_gap_comparison_metrics.csv")
    results_df.to_csv(results_file)
    print(f"\nFinal summary table saved to {results_file}")
    return results_df

# --- Example of how to run the evaluation ---
if __name__ == "__main__":
    # Use raw strings (r"...") for Windows paths
    DISCHARGE_DATA_PATH = r"C:\Users\ethan\Downloads\discharge_data_cleaned.csv"
    LAT_LONG_DATA_PATH = r"C:\Users\ethan\Downloads\lat_long_discharge.csv"
    CONTRIBUTOR_DATA_PATH = None

    EVAL_OVERALL_MIN_YEAR = 1980
    EVAL_OVERALL_MAX_YEAR = 1990
    EVAL_MIN_COMPLETENESS_TRAIN = 70.0
    EVAL_INITIAL_TRAIN_WINDOW_SIZE = 3
    EVAL_IMPUTATION_CHUNK_SIZE = 3
    EVAL_OUTPUT_DIR = "evaluation_results"
    USE_MODEL_CACHE = False # Set to True to reuse trained models if available

    TARGET_OVERALL_GAP_PERCENTAGE = 10.0 # Aim for 10% overall missing data

    evaluation_results_df = evaluate_bursting_pipeline(
        discharge_path=DISCHARGE_DATA_PATH,
        lat_long_path=LAT_LONG_DATA_PATH,
        contrib_path=CONTRIBUTOR_DATA_PATH,
        overall_min_year=EVAL_OVERALL_MIN_YEAR,
        overall_max_year=EVAL_OVERALL_MAX_YEAR,
        min_completeness_percent_train=EVAL_MIN_COMPLETENESS_TRAIN,
        initial_train_window_size=EVAL_INITIAL_TRAIN_WINDOW_SIZE,
        imputation_chunk_size_years=EVAL_IMPUTATION_CHUNK_SIZE,
        evaluation_dir=EVAL_OUTPUT_DIR,
        use_model_cache=USE_MODEL_CACHE,
        target_overall_gap_percentage=TARGET_OVERALL_GAP_PERCENTAGE # Pass the overall percentage
    )

    if evaluation_results_df is not None:
        print("\n--- Final Evaluation Summary (from main block) ---")
        print(evaluation_results_df.to_string(float_format="%.4f"))
    else:
        print("\nEvaluation process did not complete successfully.")
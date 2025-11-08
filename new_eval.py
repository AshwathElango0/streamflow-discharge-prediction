# evaluation_1980s.py
#
# This script implements a "seed-and-fill" *evaluation* strategy
# focused *only* on the 1980-1990 period.
#
# It benchmarks CustomMissForest against MICE, KNN, Kalman, and other
# methods using this "seed-and-fill" process.

import matplotlib
# FIX: Set a non-interactive backend for matplotlib to avoid Tkinter errors
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import warnings

# Import utility functions from your existing project
from simplified_utils import (
    load_and_preprocess_data,
    add_temporal_features,
    create_contiguous_segment_gaps,
    evaluate_metrics,  # Note: This is the metric calculator
    build_distance_matrix,
    build_connectivity_matrix,
    historical_mean_imputation,
    find_best_data_window # <-- Import this new function
)
# Import your custom model
from custom_missforest import CustomMissForest

# Import sklearn models
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge

# Import statsmodels for Kalman Filter
import statsmodels.api as sm


# ---
# BENCHMARK IMPUTATION FUNCTIONS (LOCAL)
# ---

def linear_interpolation_imputation(df_gapped, discharge_cols):
    """
    Imputes missing values using time-based linear interpolation.
    (This is a simple wrapper, doesn't need a seed)
    """
    print(f"\n--- Running Linear Interpolation ---")
    df_imputed = df_gapped.copy()
    
    for col in discharge_cols:
        if col in df_imputed.columns:
            # 'time' method interpolates based on index
            df_imputed[col] = df_imputed[col].interpolate(method='time')
            
            # Fill any remaining NaNs (at start/end) with backfill/ffill
            df_imputed[col] = df_imputed[col].fillna(method='ffill').fillna(method='bfill')
            
    print("✓ Linear interpolation complete.")
    return df_imputed

def spline_interpolation_imputation(df_gapped, discharge_cols):
    """
    Imputes missing values using spline interpolation.
    (This is a simple wrapper, doesn't need a seed)
    """
    print(f"\n--- Running Spline Interpolation (order 3) ---")
    df_imputed = df_gapped.copy()
    
    for col in discharge_cols:
        if col in df_imputed.columns:
            # 'spline' method with order 3
            df_imputed[col] = df_imputed[col].interpolate(method='spline', order=3)
            
            # Fill any remaining NaNs (at start/end) with backfill/ffill
            df_imputed[col] = df_imputed[col].fillna(method='ffill').fillna(method='bfill')
            
    print("✓ Spline interpolation complete.")
    return df_imputed

def historical_mean_imputation_eval(df_gapped, discharge_cols):
    """
    Wrapper for historical_mean_imputation.
    (This is a simple wrapper, doesn't need a seed, uses historical_mean_imputation from utils)
    """
    print(f"\n--- Running Historical Mean Imputation ---")
    df_imputed = historical_mean_imputation(df_gapped, discharge_cols, min_years_for_mean=1)
    print("✓ Historical mean imputation complete.")
    return df_imputed


def scale_and_impute_sklearn(df_seed_gapped, df_eval_gapped, imputer_class, discharge_cols, temporal_features):
    """
    Helper function to scale, fit on seed, and transform on eval for sklearn imputers.
    
    FIXED: This function now correctly handles all-NA columns.
    """
    print("  Scaling data...")
    
    # 1. Align columns and identify data types
    all_cols = df_seed_gapped.columns
    df_seed = df_seed_gapped.copy()
    df_eval = df_eval_gapped.copy()

    # 2. Handle All-NA columns before scaling
    # Find columns that are all-NA in *either* set, as they break scaler/imputer
    seed_all_na = df_seed[all_cols].isnull().all()
    eval_all_na = df_eval[all_cols].isnull().all()
    all_na_cols = all_cols[seed_all_na | eval_all_na].tolist()
    
    if all_na_cols:
        print(f"  Warning: Found all-NA columns: {all_na_cols}. Filling with 0 before scaling/imputation.")
        df_seed[all_na_cols] = df_seed[all_na_cols].fillna(0)
        df_eval[all_na_cols] = df_eval[all_na_cols].fillna(0)

    # 3. Fit scaler ONLY on seed data
    # We fill NaNs *before* fitting the scaler (e.g., with mean)
    seed_means = df_seed.mean().fillna(0) # Use 0 for any all-NA cols
    df_seed_filled_for_scaler = df_seed.fillna(seed_means)
    
    scaler = StandardScaler()
    scaler.fit(df_seed_filled_for_scaler)

    # 4. Transform both seed and eval
    # The scaler transform will return NaNs where data was missing
    df_seed_scaled_values = scaler.transform(df_seed)
    df_eval_scaled_values = scaler.transform(df_eval)
    
    df_seed_scaled = pd.DataFrame(df_seed_scaled_values, index=df_seed.index, columns=all_cols)
    df_eval_scaled = pd.DataFrame(df_eval_scaled_values, index=df_eval.index, columns=all_cols)
    
    # 5. Fit imputer on *scaled seed*
    print(f"  Fitting imputer ({imputer_class.__class__.__name__}) on scaled seed block...")
    imputer_class.fit(df_seed_scaled)

    # 6. Transform *scaled eval*
    print("  Transforming scaled evaluation block...")
    imputed_eval_scaled_values = imputer_class.transform(df_eval_scaled)
    
    # Reconstruct DataFrame (imputer returns numpy array)
    df_eval_imputed_scaled = pd.DataFrame(imputed_eval_scaled_values, index=df_eval.index, columns=all_cols)

    # 7. Inverse transform
    print("  Inverse transforming scaled data...")
    imputed_eval_values = scaler.inverse_transform(df_eval_imputed_scaled)
    
    df_eval_imputed = pd.DataFrame(imputed_eval_values, index=df_eval.index, columns=all_cols)
    
    # 8. Re-apply non-missing values from original gapped data
    # We only want to fill the NaNs we created
    df_eval_imputed = df_eval_gapped.combine_first(df_eval_imputed)
    
    return df_eval_imputed


def kalman_imputation(df_seed_gapped, df_eval_gapped, discharge_cols, temporal_features):
    """
    Imputes missing values using SARIMAX with a Kalman Filter.
    
    FIXED: This function now combines seed+eval and fits SARIMAX once,
    which is the correct way to impute with it.
    """
    print(f"\n--- Running Kalman (SARIMAX) Imputation ---")
    
    # Combine seed and eval. SARIMAX handles NaNs natively.
    df_combined = pd.concat([df_seed_gapped, df_eval_gapped])
    df_combined_imputed = df_combined.copy()

    # Define exogenous variables (temporal features)
    exog_vars = df_combined[temporal_features] if temporal_features else None

    for col in discharge_cols:
        print(f"  Kalman: Processing {col}...")
        endog = df_combined[col]
        
        # Skip if all data is missing
        if endog.isnull().all():
            print(f"    Skipping {col} (all missing).")
            continue
            
        try:
            # A simple, robust SARIMAX model. (1,1,1) non-seasonal, (1,1,1,12) seasonal
            # This is a common setup for monthly-like seasonal data.
            model = sm.tsa.SARIMAX(
                endog,
                exog=exog_vars,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Fit the model. The model internally uses the Kalman filter
            # to handle missing values (NaNs) in 'endog'.
            results = model.fit(disp=False)
            
            # 'fittedvalues' contains the one-step-ahead predictions.
            # For NaNs, this value is the Kalman filter's estimate.
            imputed_series = results.fittedvalues
            
            # Fill *only* the missing values in our copy
            df_combined_imputed[col] = df_combined_imputed[col].fillna(imputed_series)
            
        except Exception as e:
            print(f"    ERROR processing {col} with SARIMAX: {e}")
            # Fallback: fill with seed mean if Kalman fails
            seed_mean = df_seed_gapped[col].mean()
            if pd.isna(seed_mean): seed_mean = 0
            df_combined_imputed[col] = df_combined_imputed[col].fillna(seed_mean)

    # Return only the imputed *evaluation* block
    df_eval_imputed = df_combined_imputed.loc[df_eval_gapped.index]
    
    # Final ffill/bfill to catch any remaining NaNs (e.g., at start)
    df_eval_imputed[discharge_cols] = df_eval_imputed[discharge_cols].fillna(method='ffill').fillna(method='bfill')
    
    print("✓ Kalman (SARIMAX) imputation complete.")
    return df_eval_imputed


# ---
# EVALUATION HELPER FUNCTIONS
# ---

def evaluate_imputation_performance(df_original, df_gapped, df_imputed, discharge_cols):
    """
    Evaluate imputation performance by comparing true values with imputed values
    at the locations where artificial gaps were created.
    
    (This is the local definition that returns 3 values)
    """
    y_true_eval, y_pred_eval = [], []
    for station in discharge_cols:
        if station not in df_imputed.columns or station not in df_original.columns:
            continue
        # Get mask of *only* the artificial gaps
        gap_mask = df_gapped[station].isnull()
        
        if gap_mask.sum() > 0:
            predicted_vals = df_imputed.loc[gap_mask, station].values
            true_vals = df_original.loc[gap_mask, station].values
            
            y_pred_eval.extend(predicted_vals)
            y_true_eval.extend(true_vals)
            
    if y_true_eval:
        # This function relies on evaluate_metrics, which is imported from simplified_utils
        metrics = evaluate_metrics(np.array(y_true_eval), np.array(y_pred_eval))
        return metrics, y_true_eval, y_pred_eval
    else:
        # No gaps found or no data
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}, [], []


def plot_imputation_results(df_original, df_gapped, df_imputed, y_true, y_pred, output_dir, method_name):
    """Generates and saves plots to visualize imputation results for a specific method."""
    print(f"\n--- Generating visualizations for {method_name} ---")
    
    method_plot_dir = os.path.join(output_dir, f"plots_{method_name}")
    os.makedirs(method_plot_dir, exist_ok=True)
    
    try:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.3, edgecolors='none', label='Imputed Values', s=10)
        if y_true and y_pred:
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line (Perfect Fit)')
        plt.title(f'True vs. Predicted Discharge - {method_name}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.legend()
        plt.xlim(left=min(min_val, 0)) # Start axis at 0 or min_val
        plt.ylim(bottom=min(min_val, 0))
        scatter_path = os.path.join(method_plot_dir, f"{method_name}_true_vs_predicted_scatter.png")
        plt.savefig(scatter_path)
        plt.close()
        print(f"✓ Scatter plot saved to: {scatter_path}")
    except Exception as e:
        print(f"  WARNING: Could not generate scatter plot for {method_name}. Error: {e}")

    try:
        # Get stations that had artificial gaps
        stations_with_gaps = df_gapped.columns[df_gapped.isnull().any()].tolist()
        stations_in_original = [s for s in stations_with_gaps if s in df_original.columns]
        
        if not stations_in_original:
            print("  WARNING: No stations with gaps found for time series plots.")
            return
            
        stations_to_plot = random.sample(stations_in_original, min(len(stations_in_original), 3))
        
        for station in stations_to_plot:
            if station not in df_imputed.columns or station not in df_original.columns:
                continue
                
            plt.figure(figsize=(15, 6))
            
            # Plot original data (ground truth)
            plt.plot(df_original.index, df_original[station], color='cornflowerblue', label='Original Data (Truth)', zorder=1)
            
            # Find the artificial gap locations
            gap_mask = df_gapped[station].isnull()
            
            if gap_mask.sum() > 0:
                # Plot the imputed values at those gap locations
                plt.scatter(
                    df_imputed.index[gap_mask], 
                    df_imputed[station][gap_mask], 
                    color='red', 
                    marker='o',
                    s=20, # Smaller dots
                    label=f'Imputed Values ({method_name})', 
                    zorder=2
                )
                
            plt.title(f'Imputation Results for Station: {station} (1980s Eval Block)')
            plt.xlabel('Date')
            plt.ylabel('Discharge')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plot_path = os.path.join(method_plot_dir, f"{station}_{method_name}_imputation_plot.png")
            plt.savefig(plot_path)
            plt.close()
        print(f"✓ Time series plots saved for {method_name}.")
    except Exception as e:
        print(f"  WARNING: Could not generate time series plots for {method_name}. Error: {e}")


# ---
# MAIN EVALUATION SCRIPT
# ---

def run_focused_evaluation(
    discharge_path='discharge_data_cleaned.csv',
    lat_long_path='lat_long_discharge.csv',
    contrib_path='mahanadi_contribs.csv',
    output_dir_base="evaluation_1980s_output",
    seed_duration_years=5,
    gap_lengths=[30],
    num_gaps_per_column=10
):
    """
    Runs the "seed-and-fill" evaluation focused on the 1980-1990 period.
    """
    
    print("--- Starting Focused 1980-1990 Imputation Evaluation ---")
    
    # 1. Load and prepare all data
    print("\n--- 1. Loading and preparing all data ---")
    df_original_all, df_contrib, df_coords, _, station_to_vcode = \
        load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
    if df_original_all is None:
        print("Data loading failed. Exiting.")
        return
        
    df_with_features = add_temporal_features(df_original_all)
    
    # 2. Isolate 1980-1990 period
    print("\n--- 2. Isolating 1980-1990 data ---")
    try:
        df_1980s = df_with_features.loc['1980-01-01':'1990-12-31'].copy()
        if df_1980s.empty:
            print("ERROR: No data found in 1980-1990 period. Exiting.")
            return
    except Exception as e:
        print(f"ERROR: Could not slice 1980-1990 period: {e}")
        return
        
    print(f"Data 1980-1990 shape: {df_1980s.shape}")
    
    # Set up output directory
    start_str = df_1980s.index.min().strftime('%Y-%m-%d')
    end_str = df_1980s.index.max().strftime('%Y-%m-%d')
    output_dir = os.path.join(output_dir_base, f"evaluation_results_{start_str}_to_{end_str}")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Define columns and build matrices
    print("\n--- 3. Defining columns and building matrices ---")
    all_cols_in_data = df_1980s.columns.tolist()
    discharge_cols = [col for col in all_cols_in_data if not (
                        col.startswith('day_of_year_') or 
                        col.startswith('month_') or 
                        col.startswith('week_of_year_'))]
    temporal_features = [col for col in all_cols_in_data if col not in discharge_cols]
    
    print(f"Found {len(discharge_cols)} discharge columns (stations).")
    print(f"Found {len(temporal_features)} temporal features.")

    # Build matrices for MissForest
    all_stations = sorted(discharge_cols) 
    distance_matrix = build_distance_matrix(df_coords, all_stations)
    distance_matrix = distance_matrix.loc[all_stations, all_stations]
    connectivity_matrix = build_connectivity_matrix(df_contrib, all_stations, station_to_vcode)
    connectivity_matrix = connectivity_matrix.loc[all_stations, all_stations]
    
    # 4. Find best "seed" block and define "evaluation" block
    print("\n--- 4. Finding best 'seed' block and 'evaluation' block ---")
    
    # Calculate window size in days (e.g., 5 years = 1826 days)
    # This accounts for leap years, matching the docstring example
    window_size_days = (seed_duration_years * 365) + (seed_duration_years // 4)
    
    seed_start_date, seed_end_date = find_best_data_window(
        df=df_1980s,
        discharge_cols=discharge_cols,
        start_date_str='1980-01-01',
        end_date_str='1990-12-31',
        window_size_days=window_size_days
    )
    
    if seed_start_date is None:
        print("ERROR: Could not find a suitable seed window. Exiting.")
        return

    print(f"Found best seed block: {seed_start_date.date()} to {seed_end_date.date()}")
    
    # df_seed_original contains the *original* data (with real gaps)
    df_seed_original = df_1980s.loc[seed_start_date:seed_end_date].copy()
    
    # df_eval_original is the *rest* of the 1980-1990 data (also with real gaps)
    eval_mask = ~df_1980s.index.isin(df_seed_original.index)
    df_eval_original = df_1980s.loc[eval_mask].copy()
    
    if df_eval_original.empty:
        print("ERROR: Evaluation block is empty. Check seed/period logic. Exiting.")
        return
        
    print(f"Seed block shape: {df_seed_original.shape}")
    print(f"Evaluation block shape: {df_eval_original.shape}")
    
    # 5. Create *artificial* gaps in the evaluation block
    print("\n--- 5. Creating artificial gaps in 'evaluation' block ---")
    
    # df_eval_gapped has *both* real gaps AND new artificial gaps
    gap_info = create_contiguous_segment_gaps(
        df_eval_original, 
        discharge_cols, 
        gap_lengths=gap_lengths,
        num_intervals_per_column=num_gaps_per_column
    )
    df_eval_gapped = gap_info[gap_lengths[0]]['gapped_data']

    all_results = {}
    
    # --- 6a. Testing CustomMissForest (Your Model) ---
    print("\n--- 6a. Testing CustomMissForest_hist_mean ---")
    try:
        model = CustomMissForest(
            distance_matrix=distance_matrix,
            connectivity=connectivity_matrix,
            max_iter=10,
            n_estimators=100,
            random_state=42,
            distance_weighting_type='inverse',
            temporal_feature_columns=temporal_features,
            initialization_method='historical_mean'
        )
        
        print(f"  Training MissForest on seed block ({df_seed_original.shape})...")
        # Train on the seed data (will impute its *real* gaps)
        model.fit(df_seed_original) 
        print("  ✓ Seed training complete.")
        
        print(f"  Imputing evaluation block ({df_eval_gapped.shape})...")
        # Impute the gapped evaluation block
        df_imputed = model.transform(df_eval_gapped) 
        print("  ✓ Evaluation imputation complete.")
        
        # Evaluate *only* against the artificial gaps
        metrics, y_true, y_pred = evaluate_imputation_performance(
            df_eval_original, df_eval_gapped, df_imputed, discharge_cols
        )
        method_name = f'CustomMissForest_hist_mean'
        all_results[method_name] = metrics
        print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
        
        plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name)

    except Exception as e:
        print(f"  CustomMissForest failed: {e}")
        all_results[f'CustomMissForest_hist_mean'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}


    # --- 6b. Testing Benchmark_MICE_RandomForest ---
    print("\n--- 6b. Testing Benchmark_MICE_RandomForest ---")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=10, random_state=42), 
                max_iter=10, 
                random_state=42, 
                imputation_order='ascending'
            )
            df_imputed = scale_and_impute_sklearn(
                df_seed_original, df_eval_gapped, imputer, discharge_cols, temporal_features
            )
            
        metrics, y_true, y_pred = evaluate_imputation_performance(
            df_eval_original, df_eval_gapped, df_imputed, discharge_cols
        )
        method_name = 'Benchmark_MICE_RandomForest'
        all_results[method_name] = metrics
        print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
        plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name)
    except Exception as e:
        print(f"  MICE (RF) failed: {e}")
        all_results['Benchmark_MICE_RandomForest'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}


    # --- 6c. Testing Benchmark_KNN_k5 ---
    print("\n--- 6c. Testing Benchmark_KNN_k5 ---")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputer = KNNImputer(n_neighbors=5)
            df_imputed = scale_and_impute_sklearn(
                df_seed_original, df_eval_gapped, imputer, discharge_cols, temporal_features
            )
            
        metrics, y_true, y_pred = evaluate_imputation_performance(
            df_eval_original, df_eval_gapped, df_imputed, discharge_cols
        )
        method_name = 'Benchmark_KNN_k5'
        all_results[method_name] = metrics
        print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
        plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name)
    except Exception as e:
        print(f"  KNN failed: {e}")
        all_results['Benchmark_KNN_k5'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}


    # --- 6d. Testing Benchmark_Kalman_SARIMAX ---
    print("\n--- 6d. Testing Benchmark_Kalman_SARIMAX ---")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_imputed = kalman_imputation(
                df_seed_original, df_eval_gapped, discharge_cols, temporal_features
            )
            
        metrics, y_true, y_pred = evaluate_imputation_performance(
            df_eval_original, df_eval_gapped, df_imputed, discharge_cols
        )
        method_name = 'Benchmark_Kalman_SARIMAX'
        all_results[method_name] = metrics
        print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
        plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name)
    except Exception as e:
        print(f"  Kalman (SARIMAX) failed: {e}")
        all_results['Benchmark_Kalman_SARIMAX'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}


    # --- 6e. Testing Benchmark_Linear_Interp ---
    print("\n--- 6e. Testing Benchmark_Linear_Interp ---")
    try:
        df_imputed = linear_interpolation_imputation(df_eval_gapped, discharge_cols)
        metrics, y_true, y_pred = evaluate_imputation_performance(
            df_eval_original, df_eval_gapped, df_imputed, discharge_cols
        )
        method_name = 'Benchmark_Linear_Interp'
        all_results[method_name] = metrics
        print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
        plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name)
    except Exception as e:
        print(f"  Linear Interp failed: {e}")
        all_results['Benchmark_Linear_Interp'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    # --- 6f. Testing Baseline_Historical_Mean ---
    print("\n--- 6f. Testing Baseline_Historical_Mean ---")
    try:
        # Note: This imputes based *only* on the gapped data itself
        df_imputed = historical_mean_imputation_eval(df_eval_gapped, discharge_cols)
        metrics, y_true, y_pred = evaluate_imputation_performance(
            df_eval_original, df_eval_gapped, df_imputed, discharge_cols
        )
        method_name = 'Baseline_Historical_Mean'
        all_results[method_name] = metrics
        print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
        plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name)
    except Exception as e:
        print(f"  Historical Mean failed: {e}")
        all_results['Baseline_Historical_Mean'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}


    # --- 7. Save comprehensive results ---
    print("\n" + "="*50)
    print("FOCUSED EVALUATION COMPLETE")
    print("="*50)
    
    results_df = pd.DataFrame(all_results).T
    results_csv = os.path.join(output_dir, f"evaluation_results_{start_str}_to_{end_str}.csv")
    results_df.to_csv(results_csv)
    
    # Print summary table
    print("\n--- Final Results Summary ---")
    print(results_df.round(4))
    
    print(f"\nResults saved to: {results_csv}")
    
    return results_df


if __name__ == '__main__':
    # Run the comprehensive evaluation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_focused_evaluation()


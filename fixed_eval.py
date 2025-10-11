#!/usr/bin/env python3
"""
Fixed evaluation script that properly handles empty columns.
"""

import pandas as pd
import numpy as np
import os
from simplified_utils import (
    load_and_preprocess_data,
    add_temporal_features,
    create_contiguous_segment_gaps,
    evaluate_metrics,
    build_distance_matrix,
    build_connectivity_matrix,
    initialize_for_missforest
)
from simplified_model_config import create_full_model
from missforest_imputer import ModifiedMissForest

def evaluate_missforest_with_initialization(
    discharge_path='discharge_data_cleaned.csv',
    lat_long_path='lat_long_discharge.csv',
    contrib_path='mahanadi_contribs.csv',
    output_dir="fixed_missforest_evaluation"
):
    """
    Evaluate MissForest with different initialization methods, properly handling empty columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Starting Fixed MissForest Evaluation ---")
    
    # 1. Load and prepare the original data
    print("\n--- Loading and preparing data ---")
    df_original, df_contrib, df_coords, _, station_to_vcode = \
        load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
        
    if df_original is None:
        print("Data loading failed. Exiting.")
        return

    df_with_features = add_temporal_features(df_original)
    
    # Filter data to the relevant period (2010 onwards)
    df_filtered = df_with_features.loc['2010-01-01':].copy()
    print(f"Data filtered from 2010 onwards. Shape: {df_filtered.shape}")

    # 2. Define Training and Testing sets
    df_train = df_filtered.loc['2010-01-01':'2014-12-31']
    df_test_original = df_filtered.loc['2014-01-01':]
    
    print(f"Training data shape (2010-2014): {df_train.shape}")
    print(f"Testing data shape (2014-onwards): {df_test_original.shape}")

    # 3. Identify discharge columns and remove completely empty ones
    discharge_cols = [col for col in df_original.columns if not col.startswith('day_of_year_')]
    
    # Check for completely empty columns in training data and remove them
    empty_cols = []
    for col in discharge_cols:
        if col in df_train.columns:
            if df_train[col].isnull().sum() == len(df_train):
                empty_cols.append(col)
                print(f"Warning: Column {col} is completely empty in training data")
    
    if empty_cols:
        print(f"Removing {len(empty_cols)} completely empty columns from analysis: {empty_cols}")
        discharge_cols = [col for col in discharge_cols if col not in empty_cols]
    
    print(f"Using {len(discharge_cols)} columns for analysis: {discharge_cols}")

    # 4. Create artificial gaps in the test set for evaluation
    print("\n--- Creating artificial gaps in the test set ---")
    gap_info = create_contiguous_segment_gaps(
        df_test_original, 
        discharge_cols, 
        gap_lengths=[30], # Using 30-day gaps for evaluation
        num_intervals_per_column=15
    )
    df_test_gapped = gap_info[30]['gapped_data']

    # 5. Build matrices for MissForest
    print("\n--- Building helper matrices ---")
    all_stations = sorted(discharge_cols)
    temporal_features = ['day_of_year_sin', 'day_of_year_cos']
    
    distance_matrix = build_distance_matrix(df_coords, all_stations)
    connectivity_matrix = build_connectivity_matrix(df_contrib, all_stations, station_to_vcode)

    # 6. Test different initialization methods
    initialization_methods = ['column_mean', 'historical_mean', 'seasonal_mean']
    all_results = {}
    
    for init_method in initialization_methods:
        print(f"\n--- Testing MissForest with {init_method} initialization ---")
        try:
            # Initialize training data with the specified method
            print(f"Initializing training data with {init_method} method...")
            df_train_initialized = initialize_for_missforest(
                df_train, 
                discharge_cols, 
                init_method
            )
            
            # Verify initialization is complete
            total_nans = df_train_initialized[discharge_cols].isnull().sum().sum()
            if total_nans > 0:
                print(f"ERROR: {total_nans} NaN values remain in training data!")
                continue
            
            print(f"Training data initialization complete. NaN values: {total_nans}")
            
            # Create and train MissForest model
            model = create_full_model(distance_matrix, connectivity_matrix, temporal_features)
            
            print(f"Training MissForest with {init_method} initialization...")
            model.fit(df_train_initialized)
            print("✓ Model training complete.")
            
            # Initialize test data with the same method
            print(f"Initializing test data with {init_method} method...")
            df_test_initialized = initialize_for_missforest(
                df_test_gapped, 
                discharge_cols, 
                init_method
            )
            
            # Impute the test data
            df_imputed = model.transform(df_test_initialized)
            print("✓ MissForest imputation complete.")
            
            # Evaluate performance
            metrics = evaluate_imputation_performance(
                df_test_original, df_test_gapped, df_imputed, discharge_cols
            )
            all_results[f'MissForest_{init_method}'] = metrics
            print(f"MissForest ({init_method}) - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, KGE: {metrics['KGE']:.4f}")
            
        except Exception as e:
            print(f"MissForest with {init_method} initialization failed: {e}")
            all_results[f'MissForest_{init_method}'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    # 7. Also test baseline methods for comparison
    print(f"\n--- Testing baseline methods for comparison ---")
    
    # Column Mean baseline
    df_column_mean = initialize_for_missforest(df_test_gapped, discharge_cols, 'column_mean')
    metrics_column_mean = evaluate_imputation_performance(
        df_test_original, df_test_gapped, df_column_mean, discharge_cols
    )
    all_results['Baseline_Column_Mean'] = metrics_column_mean
    print(f"Baseline Column Mean - RMSE: {metrics_column_mean['RMSE']:.4f}, MAE: {metrics_column_mean['MAE']:.4f}, KGE: {metrics_column_mean['KGE']:.4f}")

    # Historical Mean baseline
    df_historical_mean = initialize_for_missforest(df_test_gapped, discharge_cols, 'historical_mean')
    metrics_historical_mean = evaluate_imputation_performance(
        df_test_original, df_test_gapped, df_historical_mean, discharge_cols
    )
    all_results['Baseline_Historical_Mean'] = metrics_historical_mean
    print(f"Baseline Historical Mean - RMSE: {metrics_historical_mean['RMSE']:.4f}, MAE: {metrics_historical_mean['MAE']:.4f}, KGE: {metrics_historical_mean['KGE']:.4f}")

    # 8. Save comprehensive results
    print("\n--- Saving Results ---")
    results_df = pd.DataFrame(all_results).T
    results_csv = os.path.join(output_dir, "fixed_missforest_results.csv")
    results_df.to_csv(results_csv)
    print(f"Results saved to: {results_csv}")
    
    # Print summary table
    print("\n--- Fixed MissForest Evaluation Summary ---")
    print(results_df.round(4))
    
    # Find best method for each metric
    print("\n--- Best Methods by Metric ---")
    for metric in ['RMSE', 'MAE', 'KGE']:
        if metric in results_df.columns:
            if metric == 'RMSE' or metric == 'MAE':
                best_method = results_df[metric].idxmin()
            else:  # KGE, R2, NSE - higher is better
                best_method = results_df[metric].idxmax()
            print(f"{metric}: {best_method} ({results_df.loc[best_method, metric]:.4f})")
    
    return results_df

def evaluate_imputation_performance(df_original, df_gapped, df_imputed, discharge_cols):
    """
    Evaluate imputation performance by comparing true values with imputed values
    at the locations where artificial gaps were created.
    """
    y_true_eval, y_pred_eval = [], []

    for station in discharge_cols:
        if station not in df_imputed.columns:
            continue
            
        # Find where the artificial gaps were created
        gap_mask = df_gapped[station].isnull()
        
        if gap_mask.sum() > 0:
            # Get the predicted values from our model at these specific gap locations
            predicted_vals = df_imputed[station][gap_mask].values
            
            # Get the ground truth values from the original, non-gapped test data
            true_vals = df_original[station][gap_mask].values
            
            y_pred_eval.extend(predicted_vals)
            y_true_eval.extend(true_vals)

    if y_true_eval:
        metrics = evaluate_metrics(np.array(y_true_eval), np.array(y_pred_eval))
        return metrics
    else:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

if __name__ == '__main__':
    print("Running fixed MissForest evaluation...")
    results = evaluate_missforest_with_initialization()
    
    if results is not None:
        print("\n" + "="*60)
        print("FIXED MISSFOREST EVALUATION COMPLETE!")
        print("="*60)
    else:
        print("Evaluation failed.")

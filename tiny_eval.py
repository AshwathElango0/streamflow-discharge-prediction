# evaluate_missforest.py - Evaluation script for a static ModifiedMissForest model

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# Import utility functions from your existing project
from simplified_utils import (
    load_and_preprocess_data,
    add_temporal_features,
    create_contiguous_segment_gaps,
    evaluate_metrics,
    build_distance_matrix,
    build_connectivity_matrix,
    historical_mean_imputation,
    seasonal_mean_imputation,
    simple_column_mean_imputation
)
from simplified_model_config import create_full_model
from missforest_imputer import ModifiedMissForest
from custom_missforest import CustomMissForest

def plot_imputation_results(df_original, df_gapped, df_imputed, y_true, y_pred, output_dir):
    """Generates and saves plots to visualize imputation results."""
    print("\n--- Generating visualizations ---")
    
    # --- 1. True vs. Predicted Scatter Plot ---
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', label='Imputed Values')
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line (Perfect Fit)')
    plt.title('True vs. Predicted Discharge')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.legend()
    scatter_path = os.path.join(output_dir, "true_vs_predicted_scatter.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"✓ Scatter plot saved to: {scatter_path}")

    # --- 2. Time Series Plots for specific stations ---
    stations_with_gaps = df_gapped.columns[df_gapped.isnull().any()].tolist()
    stations_to_plot = random.sample(stations_with_gaps, min(len(stations_with_gaps), 3))

    for station in stations_to_plot:
        plt.figure(figsize=(15, 6))
        
        # Plot original data
        plt.plot(df_original.index, df_original[station], color='cornflowerblue', label='Original Data', zorder=1)
        
        # Highlight imputed points
        gap_mask = df_gapped[station].isnull()
        plt.scatter(
            df_imputed.index[gap_mask], 
            df_imputed[station][gap_mask], 
            color='red', 
            marker='o',
            s=50,
            label='Imputed Values', 
            zorder=2
        )
        
        plt.title(f'Imputation Results for Station: {station}')
        plt.xlabel('Date')
        plt.ylabel('Discharge')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_path = os.path.join(output_dir, f"{station}_imputation_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✓ Time series plot for '{station}' saved to: {plot_path}")


def compare_missforest_initializations(
    discharge_path='discharge_data_cleaned.csv',
    lat_long_path='lat_long_discharge.csv',
    contrib_path='mahanadi_contribs.csv',
    output_dir="missforest_initialization_comparison"
):
    """
    Compare MissForest performance with different initialization methods:
    - Column Mean (default)
    - Historical Mean 
    - Seasonal Mean
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Starting MissForest Initialization Methods Comparison ---")
    
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

    # 3. Create artificial gaps in the test set for evaluation
    print("\n--- Creating artificial gaps in the test set ---")
    discharge_cols = [col for col in df_original.columns if not col.startswith('day_of_year_')]
    gap_info = create_contiguous_segment_gaps(
        df_test_original, 
        discharge_cols, 
        gap_lengths=[30], # Using 30-day gaps for evaluation
        num_intervals_per_column=15
    )
    df_test_gapped = gap_info[30]['gapped_data']

    # 4. Build matrices for MissForest
    print("\n--- Building helper matrices ---")
    all_stations = sorted(discharge_cols)
    temporal_features = ['day_of_year_sin', 'day_of_year_cos']
    
    distance_matrix = build_distance_matrix(df_coords, all_stations)
    connectivity_matrix = build_connectivity_matrix(df_contrib, all_stations, station_to_vcode)

    # 5. Test different initialization methods
    initialization_methods = ['column_mean', 'historical_mean', 'seasonal_mean']
    all_results = {}
    
    for init_method in initialization_methods:
        print(f"\n--- Testing MissForest with {init_method} initialization ---")
        try:
            # Create CustomMissForest with specific initialization
            model = CustomMissForest(
                distance_matrix=distance_matrix,
                connectivity=connectivity_matrix,
                max_iter=10,
                n_estimators=100,
                random_state=42,
                distance_weighting_type='inverse',
                temporal_feature_columns=temporal_features,
                initialization_method=init_method
            )
            
            # Train the model
            print(f"Training MissForest with {init_method} initialization...")
            model.fit(df_train)
            print("✓ Model training complete.")
            
            # Impute the gapped test data
            df_imputed = model.transform(df_test_gapped)
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

    # 6. Also test baseline methods for comparison
    print(f"\n--- Testing baseline methods for comparison ---")
    
    # Column Mean baseline
    df_column_mean = simple_column_mean_imputation(df_test_gapped, discharge_cols)
    metrics_column_mean = evaluate_imputation_performance(
        df_test_original, df_test_gapped, df_column_mean, discharge_cols
    )
    all_results['Baseline_Column_Mean'] = metrics_column_mean
    print(f"Baseline Column Mean - RMSE: {metrics_column_mean['RMSE']:.4f}, MAE: {metrics_column_mean['MAE']:.4f}, KGE: {metrics_column_mean['KGE']:.4f}")

    # Historical Mean baseline
    df_historical_mean = historical_mean_imputation(df_test_gapped, discharge_cols)
    metrics_historical_mean = evaluate_imputation_performance(
        df_test_original, df_test_gapped, df_historical_mean, discharge_cols
    )
    all_results['Baseline_Historical_Mean'] = metrics_historical_mean
    print(f"Baseline Historical Mean - RMSE: {metrics_historical_mean['RMSE']:.4f}, MAE: {metrics_historical_mean['MAE']:.4f}, KGE: {metrics_historical_mean['KGE']:.4f}")

    # 7. Save comprehensive results
    print("\n--- Saving Results ---")
    results_df = pd.DataFrame(all_results).T
    results_csv = os.path.join(output_dir, "missforest_initialization_comparison.csv")
    results_df.to_csv(results_csv)
    print(f"Results saved to: {results_csv}")
    
    # Print summary table
    print("\n--- MissForest Initialization Methods Comparison Summary ---")
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
    
    # Compare MissForest improvements over baselines
    print("\n--- MissForest Improvement Analysis ---")
    for init_method in initialization_methods:
        method_key = f'MissForest_{init_method}'
        baseline_key = f'Baseline_{init_method.replace("_", "_").title()}'
        
        if method_key in results_df.index and baseline_key in results_df.index:
            rmse_improvement = (results_df.loc[baseline_key, 'RMSE'] - results_df.loc[method_key, 'RMSE']) / results_df.loc[baseline_key, 'RMSE'] * 100
            kge_improvement = (results_df.loc[method_key, 'KGE'] - results_df.loc[baseline_key, 'KGE']) / abs(results_df.loc[baseline_key, 'KGE']) * 100
            
            print(f"{init_method}:")
            print(f"  RMSE improvement: {rmse_improvement:.2f}%")
            print(f"  KGE improvement: {kge_improvement:.2f}%")
    
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

def evaluate_missforest_static(
    discharge_path='discharge_data_cleaned.csv',
    lat_long_path='lat_long_discharge.csv',
    contrib_path='mahanadi_contribs.csv',
    output_dir="missforest_evaluation_results"
):
    """
    Evaluates a single ModifiedMissForest model trained on a fixed period.
    - Trains on: 2010-2014
    - Evaluates on: 2014 onwards
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Starting Static MissForest Evaluation ---")
    
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

    # 3. Create artificial gaps in the test set for evaluation
    print("\n--- Creating artificial gaps in the test set ---")
    discharge_cols = [col for col in df_original.columns if not col.startswith('day_of_year_')]
    gap_info = create_contiguous_segment_gaps(
        df_test_original, 
        discharge_cols, 
        gap_lengths=[30], # Using 30-day gaps for evaluation
        num_intervals_per_column=15
    )
    df_test_gapped = gap_info[30]['gapped_data']

    # 4. Build matrices and model
    print("\n--- Building helper matrices and initializing model ---")
    all_stations = sorted(discharge_cols)
    temporal_features = ['day_of_year_sin', 'day_of_year_cos']
    
    distance_matrix = build_distance_matrix(df_coords, all_stations)
    # Note: build_connectivity_matrix expects a station_to_vcode mapping
    connectivity_matrix = build_connectivity_matrix(df_contrib, all_stations, station_to_vcode)
    
    model = create_full_model(distance_matrix, connectivity_matrix, temporal_features)

    # 5. Train the model
    print("\n--- Training ModifiedMissForest model on 2010-2014 data ---")
    model.fit(df_train)
    print("✓ Model training complete.")

    # 6. Impute the gapped test data
    print("\n--- Imputing data on the test set (2014 onwards) ---")
    df_imputed = model.transform(df_test_gapped)
    print("✓ Imputation complete.")
    
    # Save the imputed data to a CSV file
    imputed_csv_path = os.path.join(output_dir, "imputed_data_test_set.csv")
    df_imputed.to_csv(imputed_csv_path)
    print(f"✓ Imputed test data saved to: {imputed_csv_path}")
    
    # 7. Evaluate the results
    print("\n--- Evaluating imputation performance ---")
    metrics = evaluate_imputation_performance(df_test_original, df_test_gapped, df_imputed, discharge_cols)
    
    if not all(np.isnan(list(metrics.values()))):
        print("\n--- MissForest Imputation Performance on Artificial Gaps ---")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
            
        # Save results
        results_df = pd.DataFrame([metrics])
        results_csv = os.path.join(output_dir, "missforest_static_results.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to: {results_csv}")

        # Generate and save plots
        y_true_eval, y_pred_eval = [], []
        for station in discharge_cols:
            if station in df_imputed.columns:
                gap_mask = df_test_gapped[station].isnull()
                if gap_mask.sum() > 0:
                    predicted_vals = df_imputed[station][gap_mask].values
                    true_vals = df_test_original[station][gap_mask].values
                    y_pred_eval.extend(predicted_vals)
                    y_true_eval.extend(true_vals)
        
        if y_true_eval:
            plot_imputation_results(
                df_test_original, 
                df_test_gapped, 
                df_imputed, 
                y_true_eval, 
                y_pred_eval, 
                output_dir
            )

    else:
        print("Could not find any artificial gaps in the test period to evaluate.")

if __name__ == '__main__':
    # Run the MissForest initialization methods comparison
    print("Running MissForest initialization methods comparison...")
    results = compare_missforest_initializations()
    
    if results is not None:
        print("\n" + "="*60)
        print("MISSFOREST INITIALIZATION COMPARISON COMPLETE!")
        print("="*60)
        print("\nThis comparison tests how different initialization methods")
        print("(column mean, historical mean, seasonal mean) affect the")
        print("final performance of MissForest imputation.")
        print("\nTo run only standard MissForest evaluation, uncomment the line below:")
        print("# evaluate_missforest_static()")
    else:
        print("Comparison failed. Running standard MissForest evaluation...")
        evaluate_missforest_static()



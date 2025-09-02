# run_all_evaluations.py - Global Evaluation Orchestrator for Multi-Period Analysis
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import warnings

# Import refactored evaluation function
from eval import run_evaluation

# Import model training configurations
from model_configurations import (
    train_full_model,
    train_no_temporal_model,
    train_no_contributor_model,
    train_no_spatial_temporal_model
)

# Import necessary functions from utils.py
from utils import (
    load_and_preprocess_data,
    add_temporal_features,
    build_distance_matrix,
    build_connectivity_matrix
)

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

def main():
    base_output_dir = "multi_period_evaluation_results"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Path for the overall summary text file
    summary_file_path = os.path.join(base_output_dir, "multi_period_model_comparison_summary.txt")
    # Clear content of summary file from previous runs
    with open(summary_file_path, 'w') as f:
        f.write("--- Comprehensive Multi-Period Model Evaluation Summary ---\n")
        f.write("Generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("="*80 + "\n")

    print("--- Starting Hydrological Data Imputation Multi-Model, Multi-Period Evaluation ---")

    # 1. Load and Preprocess Data (load once as raw data)
    # NOTE: The paths are assumed to be accessible in the environment.
    discharge_path = 'discharge_data_cleaned.csv'
    lat_long_path = 'lat_long_discharge.csv'
    df_data_full_period, df_lat_long = load_and_preprocess_data(discharge_path, lat_long_path)

    if df_data_full_period is None:
        print("Data loading failed. Exiting.")
        return

    # Get a list of all discharge columns to use throughout the evaluation
    discharge_cols = [col for col in df_data_full_period.columns if not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))]

    # 2. Add temporal features for all data. These will be subset later for each period.
    df_data_full_period = add_temporal_features(df_data_full_period)
    temporal_features = ['day_of_year_sin', 'day_of_year_cos']

    # 3. Build static spatial and connectivity matrices for the entire dataset
    # These will be subsetted for each specific training period later.
    distance_matrix_overall = build_distance_matrix(df_lat_long, discharge_cols)
    connectivity_matrix_overall = build_connectivity_matrix(df_lat_long, discharge_cols) # assuming contrib_path is now part of lat_long file or we need a new file

    # 4. Define evaluation windows (rolling periods)
    # The first training period is defined explicitly, then we'll roll forward
    training_period_size = 5 # years
    evaluation_period_size = 5 # years
    
    # Example: Start a rolling evaluation from a specific year.
    # Here, we'll just run one large evaluation for demonstration, but the loop would go here.
    train_start_year = 1980
    train_end_year = train_start_year + training_period_size - 1
    
    eval_start_year = train_end_year + 1
    eval_end_year = eval_start_year + evaluation_period_size - 1
    
    # 5. Loop through each training and evaluation period
    # For now, we'll do just one period as an example.
    
    # Subsetting the data for the current period
    df_train_period = df_data_full_period.loc[str(train_start_year):str(train_end_year)].copy()
    df_test_period = df_data_full_period.loc[str(eval_start_year):str(eval_end_year)].copy()
    
    # Check for empty dataframes
    if df_train_period.empty or df_test_period.empty:
        print(f"Warning: Training or testing data for period {train_start_year}-{eval_end_year} is empty. Skipping.")
        return

    # Filter spatial matrices to only include stations present in the current training period
    present_stations = [col for col in discharge_cols if not df_train_period[col].isnull().all()]
    distance_matrix_filtered = distance_matrix_overall.loc[present_stations, present_stations]
    connectivity_matrix_filtered = connectivity_matrix_overall.loc[present_stations, present_stations]

    # --- Prepare a masked training set for model fitting ---
    # We want to train the model on data that has simulated missingness
    # First, handle existing missingness by a simple mean imputation for training
    df_train_imputed = df_train_period[present_stations].copy()
    for col in df_train_imputed.columns:
        df_train_imputed[col] = df_train_imputed[col].fillna(df_train_imputed[col].mean())

    # Then, simulate 10% random missingness for model training robustness
    np.random.seed(42)
    train_mask_simulated = np.random.rand(*df_train_imputed.shape) < 0.1
    df_train_masked = df_train_period.copy()
    df_train_masked[present_stations] = df_train_imputed.mask(train_mask_simulated)

    # 6. Train the different models
    print(f"\n--- Training Models for Period {train_start_year}-{train_end_year} ---")

    # Model 1: Full Model
    full_model = train_full_model(df_train_masked, distance_matrix_filtered, connectivity_matrix_filtered, temporal_features)

    # Model 2: No Temporal Features
    no_temporal_model = train_no_temporal_model(df_train_masked, distance_matrix_filtered, connectivity_matrix_filtered, temporal_features)

    # Model 3: No Contributor Info
    no_contributor_model = train_no_contributor_model(df_train_masked, distance_matrix_filtered, connectivity_matrix_filtered, temporal_features)

    # Model 4: No Spatial or Temporal Info (Baseline)
    no_spatial_temporal_model = train_no_spatial_temporal_model(df_train_masked, distance_matrix_filtered, connectivity_matrix_filtered, temporal_features)

    # Store trained models and their types in a dictionary for easy iteration
    models_to_evaluate = {
        'Full Model': full_model,
        'No Temporal': no_temporal_model,
        'No Contributor': no_contributor_model,
        'No Spatial/Temporal': no_spatial_temporal_model
    }

    # 7. Evaluate each model on the test period
    print(f"\n--- Evaluating Models on Test Period {eval_start_year}-{eval_end_year} ---")
    
    # Gap lengths to test (in days)
    gap_lengths = [1, 3, 5, 7, 10, 14, 21, 30] 

    for model_name, model in models_to_evaluate.items():
        print(f"\nEvaluating {model_name}...")
        model_output_dir = os.path.join(base_output_dir, model_name.replace(' ', '_').replace('/', '_'))
        run_evaluation(model, df_test_period, gap_lengths, model_name, model_output_dir, summary_file_path)

    # 8. Generate comparative plots and summary tables
    print("\n--- Generating Comparative Plots ---")
    plot_comparisons(base_output_dir, models_to_evaluate.keys(), [f'Test Period {eval_start_year}-{eval_end_year}'])
    
    print("\nEvaluation pipeline finished.")

def plot_comparisons(base_output_dir, model_names, period_labels):
    """
    Loads evaluation results from CSV files and generates comparative plots.
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    line_styles = ['-', '--', '-.', ':']

    for period_label in period_labels:
        period_label_safe = period_label.replace(' ', '_').replace('/', '_')
        current_period_results_for_plotting = {}
        for model_name in model_names:
            model_safe_name = model_name.replace(' ', '_').replace('/', '_')
            csv_path = os.path.join(base_output_dir, model_safe_name, "evaluation_results.csv")
            if os.path.exists(csv_path):
                results_df = pd.read_csv(csv_path, index_col=0)
                current_period_results_for_plotting[model_name] = results_df
            else:
                print(f"Warning: CSV file not found for {model_name} in {period_label}. Skipping plot for this model.")
        
        if not current_period_results_for_plotting:
            print(f"No results found for period '{period_label}'. Skipping plot generation.")
            continue

        metrics_to_plot = ['R2', 'RMSE', 'MAE']
        
        plt.figure(figsize=(20, 5))
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(1, len(metrics_to_plot), i + 1)
            for j, (model_name, results_df) in enumerate(current_period_results_for_plotting.items()):
                if metric in results_df.columns:
                    plt.plot(results_df.index, results_df[metric], 
                             marker=markers[j % len(markers)], 
                             linestyle=line_styles[j % len(line_styles)], 
                             label=model_name, 
                             color=colors[j % len(colors)])
                
            plt.xlabel('Gap Length (days)')
            plt.ylabel(metric)
            plt.xticks(results_df.index)
            plt.title(f'{metric} Comparison for {period_label}')
            plt.grid(True, linestyle=':', alpha=0.7)
            if i == 0:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plot_filename = os.path.join(base_output_dir, period_label_safe, f"comparison_metrics_{period_label_safe}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Comparative plot for period '{period_label}' saved to: {plot_filename}")

if __name__ == '__main__':
    main()

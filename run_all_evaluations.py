# run_all_evaluations.py - Global Evaluation Orchestrator for Multi-Period Analysis
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import warnings
import tempfile # Added for temporary file handling
import atexit # Added for cleanup of temporary files

# Import refactored evaluation function
from eval import evaluate_metrics # Only need evaluate_metrics here

# Import burst pipeline
from burst_pipeline import run_rolling_imputation_pipeline

# Import gap generation and utility functions
from utils import (
    load_and_preprocess_data,
    add_temporal_features,
    build_distance_matrix,
    build_connectivity_matrix,
    create_contiguous_segment_gaps,
    create_single_point_gaps
)

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

def main():
    base_output_dir = "multi_period_evaluation_results"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Path for the overall summary text file
    summary_file_path = os.path.join(base_output_dir, "multi_period_model_comparison_summary.txt")
    # Clear content of summary file from previous runs
    with open(summary_file_path, 'w') as f:
        f.write("--- Comprehensive Burst Pipeline Evaluation Summary ---\n") # Updated title
        f.write("Generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("="*80 + "\n")

    print("--- Starting Bursting Imputation Pipeline Evaluation ---") # Updated title

    # Define paths to your data files
    DISCHARGE_DATA_PATH = 'discharge_data_cleaned.csv'
    LAT_LONG_DATA_PATH = 'lat_long_discharge.csv'
    CONTRIBUTOR_DATA_PATH = 'mahanadi_contribs.csv'
    
    # 1. Load and Preprocess Data to get the original, complete data for gap generation
    print("\n--- Loading original data for gap generation ---")
    df_original_raw, df_contrib, df_coords, vcode_to_station_name, station_name_to_vcode = \
        load_and_preprocess_data(DISCHARGE_DATA_PATH, LAT_LONG_DATA_PATH, CONTRIBUTOR_DATA_PATH)
    
    if df_original_raw is None:
        print("FATAL ERROR: Failed to load original data. Exiting.")
        return

    # Add temporal features to the original raw data for consistency with pipeline input
    df_original_with_features = add_temporal_features(df_original_raw)

    # Identify all discharge columns from the original data
    discharge_cols_overall = [col for col in df_original_raw.columns if not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))]

    print("\n--- Starting Evaluation of Bursting Pipeline Output on Artificial Gaps ---")

    # Define gap lengths to test
    gap_lengths_contiguous = [7, 14, 21, 30, 60, 180, 365]
    gap_lengths_single_point = [30, 60, 90, 120, 180, 365]

    all_evaluation_metrics = []

    # List to keep track of temporary files for cleanup
    temp_files_to_clean = []

    # Register cleanup function to run on exit
    def cleanup_temp_files():
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                os.remove(f_path)
                print(f"Cleaned up temporary file: {f_path}")
    atexit.register(cleanup_temp_files)

    # --- Evaluate Contiguous Gaps ---
    if gap_lengths_contiguous:
        print("\n--- Evaluating Contiguous Segment Gaps ---")
        for length in gap_lengths_contiguous:
            print(f"\n  Processing {length}-day contiguous gaps...")
            
            # 1. Create a gapped dataset for this specific scenario
            results_gaps = create_contiguous_segment_gaps(
                df_original_with_features, discharge_cols_overall, [length], random_seed=42
            ) # Pass [length] as create_contiguous_segment_gaps expects a list
            current_gapped_data = results_gaps[length]['gapped_data']
            current_true_values_dict = results_gaps[length]['true_values']
            current_mask_dict = results_gaps[length]['mask']

            # 2. Save the gapped data to a temporary CSV file
            temp_gapped_discharge_file = tempfile.NamedTemporaryFile(mode='w', suffix='_gapped.csv', delete=False)
            current_gapped_data.to_csv(temp_gapped_discharge_file.name, index=True, date_format='%Y-%m-%d')
            temp_gapped_discharge_file.close()
            temp_files_to_clean.append(temp_gapped_discharge_file.name)
            print(f"  Temporary gapped data saved to: {temp_gapped_discharge_file.name}")

            # 3. Run the Bursting Imputation Pipeline with the temporarily gapped data
            print(f"  Running Burst Pipeline on gapped data for {length}-day contiguous gaps...")
            df_imputed_by_pipeline = run_rolling_imputation_pipeline(
                discharge_path=temp_gapped_discharge_file.name,
                lat_long_path=LAT_LONG_DATA_PATH,
                contrib_path=CONTRIBUTOR_DATA_PATH,
                initial_train_window_size=5,
                imputation_chunk_size_years=5,
                overall_min_year=df_original_raw.index.year.min(), # Use actual min year of data
                overall_max_year=df_original_raw.index.year.max(), # Use actual max year of data
                min_completeness_percent_train=70.0,
                output_dir=os.path.join(base_output_dir, f"burst_output_contiguous_{length}") # Separate output for each run
            )

            if df_imputed_by_pipeline is None or df_imputed_by_pipeline.empty:
                print(f"Warning: Pipeline failed for {length}-day contiguous gaps. Skipping metrics.")
                continue

            # 4. Extract true and imputed values and calculate global metrics
            y_true_all_gaps = []
            y_pred_all_gaps = []

            for col in discharge_cols_overall:
                y_true_all_gaps.extend(current_true_values_dict[col])
                gapped_indices_for_col = df_original_with_features.index[current_mask_dict[col]]
                
                if col in df_imputed_by_pipeline.columns:
                    y_pred_col = df_imputed_by_pipeline.loc[gapped_indices_for_col, col].values
                    y_pred_all_gaps.extend(y_pred_col)
                else:
                    print(f"Warning: Column {col} not found in imputed pipeline output for {length}-day contiguous gaps.")
                    y_pred_all_gaps.extend([np.nan] * len(current_true_values_dict[col]))

            y_true_global = np.array(y_true_all_gaps)
            y_pred_global = np.array(y_pred_all_gaps)
                
            metrics = evaluate_metrics(y_true_global, y_pred_global)
            metrics['Gap Type'] = 'Contiguous'
            metrics['Gap Length'] = length
            metrics['Model'] = 'Burst Pipeline'
            all_evaluation_metrics.append(metrics)
            print(f"  Global Metrics for {length}-day contiguous gaps (Burst Pipeline): {metrics}")

    # --- Evaluate Single Point Gaps ---
    if gap_lengths_single_point:
        print("\n--- Evaluating Single Data Point Gaps ---")
        for length in gap_lengths_single_point:
            print(f"\n  Processing {length}-day equivalent single point gaps...")
            
            # 1. Create a gapped dataset for this specific scenario
            single_point_gaps_config = {length: length}
            results_gaps = create_single_point_gaps(
                df_original_with_features, discharge_cols_overall, single_point_gaps_config, random_seed=42
            )
            current_gapped_data = results_gaps[length]['gapped_data']
            current_true_values_dict = results_gaps[length]['true_values']
            current_mask_dict = results_gaps[length]['mask']

            # 2. Save the gapped data to a temporary CSV file
            temp_gapped_discharge_file = tempfile.NamedTemporaryFile(mode='w', suffix='_gapped.csv', delete=False)
            current_gapped_data.to_csv(temp_gapped_discharge_file.name, index=True, date_format='%Y-%m-%d')
            temp_gapped_discharge_file.close()
            temp_files_to_clean.append(temp_gapped_discharge_file.name)
            print(f"  Temporary gapped data saved to: {temp_gapped_discharge_file.name}")

            # 3. Run the Bursting Imputation Pipeline with the temporarily gapped data
            print(f"  Running Burst Pipeline on gapped data for {length}-day equivalent single point gaps...")
            df_imputed_by_pipeline = run_rolling_imputation_pipeline(
                discharge_path=temp_gapped_discharge_file.name,
                lat_long_path=LAT_LONG_DATA_PATH,
                contrib_path=CONTRIBUTOR_DATA_PATH,
                initial_train_window_size=5,
                imputation_chunk_size_years=5,
                overall_min_year=df_original_raw.index.year.min(), # Use actual min year of data
                overall_max_year=df_original_raw.index.year.max(), # Use actual max year of data
                min_completeness_percent_train=70.0,
                output_dir=os.path.join(base_output_dir, f"burst_output_single_point_{length}") # Separate output for each run
            )

            if df_imputed_by_pipeline is None or df_imputed_by_pipeline.empty:
                print(f"Warning: Pipeline failed for {length}-day single point gaps. Skipping metrics.")
                continue

            # 4. Extract true and imputed values and calculate global metrics
            y_true_all_gaps = []
            y_pred_all_gaps = []

            for col in discharge_cols_overall:
                y_true_all_gaps.extend(current_true_values_dict[col])
                gapped_indices_for_col = df_original_with_features.index[current_mask_dict[col]]

                if col in df_imputed_by_pipeline.columns:
                    y_pred_col = df_imputed_by_pipeline.loc[gapped_indices_for_col, col].values
                    y_pred_all_gaps.extend(y_pred_col)
                else:
                    print(f"Warning: Column {col} not found in imputed pipeline output for {length}-day single point gaps.")
                    y_pred_all_gaps.extend([np.nan] * len(current_true_values_dict[col]))
            
            y_true_global = np.array(y_true_all_gaps)
            y_pred_global = np.array(y_pred_all_gaps)

            metrics = evaluate_metrics(y_true_global, y_pred_global)
            metrics['Gap Type'] = 'Single Point'
            metrics['Gap Length'] = length
            metrics['Model'] = 'Burst Pipeline'
            all_evaluation_metrics.append(metrics)

            print(f"  Global Metrics for {length}-day equivalent single point gaps (Burst Pipeline): {metrics}")

    if not all_evaluation_metrics:
        print("No evaluation metrics were generated.")
        return

    # Convert all collected metrics to a DataFrame
    results_df = pd.DataFrame(all_evaluation_metrics)
    results_csv_path = os.path.join(base_output_dir, "burst_pipeline_evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nAll evaluation results for Burst Pipeline saved to: {results_csv_path}")
    print(results_df.to_string(float_format="%.4f"))

    # Append results to the summary file
    with open(summary_file_path, 'a') as f:
        f.write(f"\n--- Evaluation Results for: Burst Pipeline ---\n")
        f.write(results_df.to_string(float_format="%.4f"))
        f.write("\n" + "="*80 + "\n") # Separator

    print("\n--- Generating Comparative Plots ---")
    plot_comparisons(base_output_dir, ['Burst Pipeline'], results_df=results_df) # Pass results_df directly
    
    print("\nEvaluation pipeline finished.")

def plot_comparisons(base_output_dir, model_names, results_df=None):
    """
    Loads evaluation results from CSV files or uses a provided DataFrame and generates comparative plots.
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    line_styles = ['-', '--', '-.', ':']

    all_global_results = {}

    if results_df is not None:
        # If a DataFrame is provided, use it directly (assuming it's already global results)
        for model_name in model_names:
            # Filter for the specific model if results_df contains multiple (e.g., if you expanded later)
            # For now, assuming results_df contains results for the single 'Burst Pipeline' model
            all_global_results[model_name] = results_df[results_df['Model'] == model_name].set_index('Gap Length')
    else:
        for model_name in model_names:
            model_safe_name = model_name.replace(' ', '_').replace('/', '_')
            csv_path = os.path.join(base_output_dir, "burst_pipeline_evaluation_results.csv") # Corrected path

            if os.path.exists(csv_path):
                results_df_from_file = pd.read_csv(csv_path)
                global_results_df = results_df_from_file[results_df_from_file['Station'] == 'Global'].set_index('Gap Length')
                if not global_results_df.empty:
                    all_global_results[model_name] = global_results_df
                else:
                    print(f"Warning: No global results found for {model_name} in {csv_path}.")
            else:
                print(f"Warning: CSV file not found for {model_name} at {csv_path}. Skipping plot for this model.")

    if not all_global_results:
        print(f"No global results found across all models. Skipping comparative plot generation.")
        return

    metrics_to_plot = ['R2', 'RMSE', 'MAE', 'NSE']
    
    sample_model_name = list(all_global_results.keys())[0]
    gap_lengths_for_plot = all_global_results[sample_model_name].index.unique().sort_values()

    # Plotting for each Gap Type separately
    gap_types = all_global_results[sample_model_name]['Gap Type'].unique()

    for gap_type in gap_types:
        plt.figure(figsize=(20, 10)) # Increased height for two rows of plots if needed
        plt.suptitle(f'Burst Pipeline Evaluation - {gap_type} Gaps', fontsize=16)

        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, len(metrics_to_plot) // 2 if len(metrics_to_plot) > 2 else 1, i + 1)
            for j, (model_name, results_df_model) in enumerate(all_global_results.items()):
                # Filter by current gap type
                df_to_plot = results_df_model[results_df_model['Gap Type'] == gap_type]
                
                if metric in df_to_plot.columns and not df_to_plot.empty:
                    plt.plot(df_to_plot.index, df_to_plot[metric], 
                             marker=markers[j % len(markers)], 
                             linestyle=line_styles[j % len(line_styles)], 
                             label=model_name, 
                             color=colors[j % len(colors)])
            
            plt.xlabel('Gap Length (days)')
            plt.ylabel(metric)
            plt.xticks(gap_lengths_for_plot)
            plt.title(f'{metric} Comparison')
            plt.grid(True, linestyle=':', alpha=0.7)
            if i == 0:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plot_filename = os.path.join(base_output_dir, f"comparison_metrics_global_{gap_type.replace(' ', '_')}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Global comparative plot for {gap_type} gaps saved to: {plot_filename}")

if __name__ == '__main__':
    main()

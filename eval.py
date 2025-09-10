# eval.py - Refactored to be an evaluation function for a single model run
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import create_contiguous_segment_gaps, create_single_point_gaps, evaluate_metrics, plot_results

def run_evaluation(model, test_data, gap_lengths_contiguous, gap_lengths_single_point, model_type, output_dir, summary_file_path):
    """
    Evaluates a trained imputation model on test data with various gap lengths
    for both contiguous segments and single data points.
    Generates performance metrics, plots, and appends results to a summary file.

    Args:
        model (object): An instantiated imputation model with a `transform` method.
        test_data (pd.DataFrame): The original, complete test data (without any missingness).
        gap_lengths_contiguous (list): List of gap lengths (in days) for contiguous segments.
        gap_lengths_single_point (list): List of gap lengths (in days) for single data points.
        model_type (str): Descriptor for the model type (e.g., 'Inverse Weighting', 'No Temporal Info').
        output_dir (str): Directory to save evaluation results and plots for this specific model.
        summary_file_path (str): Path to a text file to append comprehensive summary results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.plot_output_dir = output_dir 

    print(f"\n--- Starting Evaluation for {model_type} Model ---")
    print(f"Saving specific model results to: {output_dir}")

    discharge_columns = [col for col in test_data.columns if not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))]
    if not discharge_columns:
        print("Error: No discharge columns found in test data. Cannot evaluate.")
        return pd.DataFrame() 

    all_evaluation_metrics = []

    # --- Evaluate Contiguous Gaps ---
    if gap_lengths_contiguous:
        print("\n--- Evaluating Contiguous Segment Gaps ---")
        results_contiguous_gaps = create_contiguous_segment_gaps(test_data, discharge_columns, gap_lengths_contiguous, random_seed=42)
        for length, gap_dict in results_contiguous_gaps.items():
            print(f"\n  Evaluating for {length}-day contiguous gaps with {model_type}...")
            imputed_data = model.transform(gap_dict['gapped_data'])
            
            y_true_all_gaps = []
            y_pred_all_gaps = []

            for col in discharge_columns:
                # Collect true values for this column and gap
                y_true_col = gap_dict['true_values'][col]
                y_true_all_gaps.extend(y_true_col)

                # Collect predicted values for this column and gap
                # Use the boolean mask for indexing rows and the specific column name
                gapped_indices_for_col = test_data.index[gap_dict['mask'][col]]
                y_pred_col = imputed_data.loc[gapped_indices_for_col, col].values
                y_pred_all_gaps.extend(y_pred_col)

            y_true_global = np.array(y_true_all_gaps)
            y_pred_global = np.array(y_pred_all_gaps)
                
            metrics = evaluate_metrics(y_true_global, y_pred_global)
            metrics['Gap Type'] = 'Contiguous'
            metrics['Gap Length'] = length
            metrics['Station'] = 'Global' # Indicate global metrics
            all_evaluation_metrics.append(metrics)
            
            print(f"  Global Metrics for {length}-day contiguous gaps: {metrics}")
            # Removed per-station plotting

    # --- Evaluate Single Point Gaps ---
    if gap_lengths_single_point:
        print("\n--- Evaluating Single Data Point Gaps ---")
        single_point_gaps_config = {length: length for length in gap_lengths_single_point}
        results_single_point_gaps = create_single_point_gaps(test_data, discharge_columns, single_point_gaps_config, random_seed=42)
        for length, gap_dict in results_single_point_gaps.items():
            print(f"\n  Evaluating for {length}-day equivalent single point gaps with {model_type}...")
            imputed_data = model.transform(gap_dict['gapped_data'])

            y_true_all_gaps = []
            y_pred_all_gaps = []

            for col in discharge_columns:
                # Collect true values for this column and gap
                y_true_col = gap_dict['true_values'][col]
                y_true_all_gaps.extend(y_true_col)

                # Collect predicted values for this column and gap
                gapped_indices_for_col = test_data.index[gap_dict['mask'][col]]
                y_pred_col = imputed_data.loc[gapped_indices_for_col, col].values
                y_pred_all_gaps.extend(y_pred_col)
            
            y_true_global = np.array(y_true_all_gaps)
            y_pred_global = np.array(y_pred_all_gaps)

            metrics = evaluate_metrics(y_true_global, y_pred_global)
            metrics['Gap Type'] = 'Single Point'
            metrics['Gap Length'] = length
            metrics['Station'] = 'Global' # Indicate global metrics
            all_evaluation_metrics.append(metrics)

            print(f"  Global Metrics for {length}-day equivalent single point gaps: {metrics}")
            # Removed per-station plotting

    if not all_evaluation_metrics:
        print("No evaluation metrics were generated.")
        return pd.DataFrame()

    # Convert all collected metrics to a DataFrame
    results_df = pd.DataFrame(all_evaluation_metrics)
    
    results_csv_path = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nAll evaluation results for {model_type} saved to: {results_csv_path}")
    print(results_df.to_string(float_format="%.4f"))

    # Append results to the summary file
    with open(summary_file_path, 'a') as f:
        f.write(f"\n--- Evaluation Results for: {model_type} ---\n")
        f.write(results_df.to_string(float_format="%.4f"))
        f.write("\n" + "="*80 + "\n") # Separator

    print(f"--- Evaluation for {model_type} Model Complete ---")
    return results_df


# eval.py - Refactored to be an evaluation function for a single model run
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import create_continuous_gaps, evaluate_metrics, plot_results

def run_evaluation(model, test_data, gap_lengths, model_type, output_dir, summary_file_path):
    """
    Evaluates a trained imputation model on test data with various gap lengths.
    Generates performance metrics, plots, and appends results to a summary file.

    Args:
        model (object): An instantiated imputation model with a `transform` method.
        test_data (pd.DataFrame): The original, complete test data (without any missingness).
        gap_lengths (list): List of gap lengths (in days) to evaluate.
        model_type (str): Descriptor for the model type (e.g., 'Inverse Weighting', 'No Temporal Info').
        output_dir (str): Directory to save evaluation results and plots for this specific model.
        summary_file_path (str): Path to a text file to append comprehensive summary results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the plot output directory for the model object (used by plot_results in utils)
    # This is a bit of a workaround for the plot_results function which expects this attribute.
    # In a more advanced refactor, plot_results could take plot_dir directly.
    model.plot_output_dir = output_dir 

    print(f"\n--- Starting Evaluation for {model_type} Model ---")
    print(f"Saving specific model results to: {output_dir}")

    # Identify true discharge columns (excluding temporal features)
    discharge_columns = [col for col in test_data.columns if not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))]
    if not discharge_columns:
        print("Error: No discharge columns found in test data. Cannot evaluate.")
        return pd.DataFrame() # Return empty DataFrame if no discharge columns

    results_gaps = create_continuous_gaps(test_data, discharge_columns, gap_lengths, random_seed=42)
    
    evaluation_metrics_by_gap_length = {}

    for length, gap_dict in results_gaps.items():
        print(f"\n  Evaluating for {length}-day gaps with {model_type}...")
        
        # Ensure that the data passed to transform has the correct temporal features
        # The create_continuous_gaps ensures temporal features are not gapped,
        # but the model itself expects them if trained with them.
        imputed_data = model.transform(gap_dict['gapped_data'])
        
        y_true = gap_dict['true_values']
        # Filter imputed_data to only include discharge columns before indexing with the mask
        # Use the same discharge_columns identified earlier for consistency
        y_pred = imputed_data[discharge_columns].values[gap_dict['mask']]
        
        metrics = evaluate_metrics(y_true, y_pred)
        evaluation_metrics_by_gap_length[length] = metrics
        
        print(f"  Metrics for {length}-day gaps: {metrics}")
        
        # Only plot if there are valid true values (i.e., actual gaps were introduced and values exist)
        if not np.all(np.isnan(y_true)):
            plot_results(y_true, y_pred, length, plot_dir=output_dir)
        else:
            print(f"  Skipping plot for {length}-day gaps: no valid true values to plot.")
            
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(evaluation_metrics_by_gap_length).T
    results_csv_path = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(results_csv_path)
    print(f"\nEvaluation results for {model_type} saved to: {results_csv_path}")
    print(results_df.to_string(float_format="%.4f"))

    # Append results to the summary file
    with open(summary_file_path, 'a') as f:
        f.write(f"\n--- Evaluation Results for: {model_type} ---\n")
        f.write(results_df.to_string(float_format="%.4f"))
        f.write("\n" + "="*80 + "\n") # Separator

    print(f"--- Evaluation for {model_type} Model Complete ---")
    return results_df # Return dataframe for comparative plotting


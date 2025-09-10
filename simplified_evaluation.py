# simplified_evaluation.py - Streamlined evaluation for burst pipeline
import pandas as pd
import numpy as np
import os
import tempfile
import atexit
from eval import evaluate_metrics
from burst_pipeline import run_rolling_imputation_pipeline
from utils import (
    load_and_preprocess_data,
    add_temporal_features,
    create_contiguous_segment_gaps,
    create_single_point_gaps
)

def evaluate_burst_pipeline(
    discharge_path='discharge_data_cleaned.csv',
    lat_long_path='lat_long_discharge.csv',
    contrib_path='mahanadi_contribs.csv',
    output_dir="evaluation_results",
    gap_lengths_contiguous=[7, 14, 21, 30, 60, 180, 365],
    gap_lengths_single_point=[30, 60, 90, 120, 180, 365],
    initial_train_window_size=5,
    imputation_chunk_size_years=5,
    min_completeness_percent_train=70.0
):
    """
    Simplified evaluation of the burst pipeline on artificial gaps.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Starting Burst Pipeline Evaluation ---")
    
    # Load and preprocess data
    print("Loading data...")
    df_original, df_contrib, df_coords, vcode_to_station, station_to_vcode = \
        load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
    
    if df_original is None:
        print("ERROR: Failed to load data. Exiting.")
        return None
    
    # Add temporal features
    df_with_features = add_temporal_features(df_original)
    discharge_cols = [col for col in df_original.columns if not col.startswith('day_of_year_')]
    
    # Track temporary files for cleanup
    temp_files = []
    atexit.register(lambda: [os.remove(f) for f in temp_files if os.path.exists(f)])
    
    all_results = []
    
    # Evaluate contiguous gaps
    if gap_lengths_contiguous:
        print("\n--- Evaluating Contiguous Gaps ---")
        for length in gap_lengths_contiguous:
            print(f"Processing {length}-day contiguous gaps...")
            
            # Create gaps
            gap_results = create_contiguous_segment_gaps(
                df_with_features, discharge_cols, [length], random_seed=42
            )
            gapped_data = gap_results[length]['gapped_data']
            true_values = gap_results[length]['true_values']
            mask = gap_results[length]['mask']
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='_gapped.csv', delete=False)
            gapped_data.to_csv(temp_file.name, index=True, date_format='%Y-%m-%d')
            temp_file.close()
            temp_files.append(temp_file.name)
            
            # Run burst pipeline
            imputed_data = run_rolling_imputation_pipeline(
                discharge_path=temp_file.name,
                lat_long_path=lat_long_path,
                contrib_path=contrib_path,
                initial_train_window_size=initial_train_window_size,
                imputation_chunk_size_years=imputation_chunk_size_years,
                overall_min_year=df_original.index.year.min(),
                overall_max_year=df_original.index.year.max(),
                min_completeness_percent_train=min_completeness_percent_train,
                output_dir=os.path.join(output_dir, f"burst_contiguous_{length}")
            )
            
            if imputed_data is not None and not imputed_data.empty:
                # Calculate metrics
                y_true, y_pred = [], []
                for col in discharge_cols:
                    y_true.extend(true_values[col])
                    gapped_indices = df_with_features.index[mask[col]]
                    if col in imputed_data.columns:
                        y_pred.extend(imputed_data.loc[gapped_indices, col].values)
                    else:
                        y_pred.extend([np.nan] * len(true_values[col]))
                
                metrics = evaluate_metrics(np.array(y_true), np.array(y_pred))
                metrics.update({
                    'Gap Type': 'Contiguous',
                    'Gap Length': length,
                    'Model': 'Burst Pipeline'
                })
                all_results.append(metrics)
                print(f"  Metrics: {metrics}")
    
    # Evaluate single point gaps
    if gap_lengths_single_point:
        print("\n--- Evaluating Single Point Gaps ---")
        for length in gap_lengths_single_point:
            print(f"Processing {length}-day equivalent single point gaps...")
            
            # Create gaps
            gap_config = {length: length}
            gap_results = create_single_point_gaps(
                df_with_features, discharge_cols, gap_config, random_seed=42
            )
            gapped_data = gap_results[length]['gapped_data']
            true_values = gap_results[length]['true_values']
            mask = gap_results[length]['mask']
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='_gapped.csv', delete=False)
            gapped_data.to_csv(temp_file.name, index=True, date_format='%Y-%m-%d')
            temp_file.close()
            temp_files.append(temp_file.name)
            
            # Run burst pipeline
            imputed_data = run_rolling_imputation_pipeline(
                discharge_path=temp_file.name,
                lat_long_path=lat_long_path,
                contrib_path=contrib_path,
                initial_train_window_size=initial_train_window_size,
                imputation_chunk_size_years=imputation_chunk_size_years,
                overall_min_year=df_original.index.year.min(),
                overall_max_year=df_original.index.year.max(),
                min_completeness_percent_train=min_completeness_percent_train,
                output_dir=os.path.join(output_dir, f"burst_single_{length}")
            )
            
            if imputed_data is not None and not imputed_data.empty:
                # Calculate metrics
                y_true, y_pred = [], []
                for col in discharge_cols:
                    y_true.extend(true_values[col])
                    gapped_indices = df_with_features.index[mask[col]]
                    if col in imputed_data.columns:
                        y_pred.extend(imputed_data.loc[gapped_indices, col].values)
                    else:
                        y_pred.extend([np.nan] * len(true_values[col]))
                
                metrics = evaluate_metrics(np.array(y_true), np.array(y_pred))
                metrics.update({
                    'Gap Type': 'Single Point',
                    'Gap Length': length,
                    'Model': 'Burst Pipeline'
                })
                all_results.append(metrics)
                print(f"  Metrics: {metrics}")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv = os.path.join(output_dir, "burst_pipeline_results.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to: {results_csv}")
        print(results_df.to_string(float_format="%.4f"))
        return results_df
    else:
        print("No results generated.")
        return None

if __name__ == '__main__':
    results = evaluate_burst_pipeline()
    if results is not None:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed.")

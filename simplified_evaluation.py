# simplified_evaluation.py - Streamlined evaluation for burst pipeline
import pandas as pd
import numpy as np
import os
import tempfile
import atexit
from eval import evaluate_metrics
from simplified_burst_pipeline import BurstImputer
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
    This version loads the correct cached model for each gapped period.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Starting Burst Pipeline Evaluation ---")
    
    # 1. Load and prepare the original data
    print("Loading and preparing data...")
    df_original, df_contrib, df_coords, vcode_to_station, station_to_vcode = \
        load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
    df_with_features = add_temporal_features(df_original)
    
    # 2. Instantiate imputer and pre-train all rolling models
    imputer = BurstImputer(
        initial_train_window_size=initial_train_window_size,
        min_completeness_percent_train=min_completeness_percent_train
    )
    
    print("\nPre-training and caching models for all rolling periods...")
    imputer.impute_rolling_windows(
        df_with_features=df_with_features,
        df_coords=df_coords,
        df_contrib=df_contrib,
        imputation_chunk_size_years=imputation_chunk_size_years
    )
    
    all_results = []
    
    # 3. Evaluate with contiguous gaps using the cached models
    print("\n--- Evaluating Contiguous Gaps ---")
    for length in gap_lengths_contiguous:
        print(f"Creating and evaluating gaps of length: {length}")
        gaps_info = create_contiguous_segment_gaps(df_with_features, length=length)
        
        for num_gaps_key, gap_data in gaps_info.items():
            df_gapped_features = gap_data['gapped_data']
            true_values = gap_data['true_values']
            
            # Identify the year of the gapped data to select the correct model
            gap_year = df_gapped_features.index.year[0]
            start_year = int(np.floor(gap_year / imputation_chunk_size_years) * imputation_chunk_size_years)
            end_year = start_year + initial_train_window_size
            
            # Load the correct cached model
            cache_key = imputer._create_cache_key(df_gapped_features, start_year, end_year)
            model = imputer.load_model_from_cache(cache_key)
            
            if model is None:
                print(f"Warning: No model found for period {start_year}-{end_year}. Skipping.")
                continue
                
            # Use the pre-trained model to impute
            imputed_data = imputer.impute(model, df_gapped_features)
            
            # Calculate metrics
            y_true, y_pred = [], []
            discharge_cols = [col for col in df_original.columns if col in df_gapped_features.columns]
            for col in discharge_cols:
                y_true.extend(true_values.get(col, []))
                if col in imputed_data.columns:
                    y_pred.extend(imputed_data.loc[df_gapped_features.index, col].values)
            
            metrics = evaluate_metrics(np.array(y_true), np.array(y_pred))
            metrics.update({
                'Gap Type': 'Contiguous Segment',
                'Gap Length': length,
                'Model': 'Burst Pipeline'
            })
            all_results.append(metrics)
            print(f"  Metrics: {metrics}")

    # 4. Evaluate with single-point gaps using the same trained models
    print("\n--- Evaluating Single-Point Gaps ---")
    for length in gap_lengths_single_point:
        print(f"Creating and evaluating gaps with {length} single points")
        gaps_info = create_single_point_gaps(df_with_features, num_gaps=length)
        
        for num_gaps_key, gap_data in gaps_info.items():
            df_gapped_features = gap_data['gapped_data']
            true_values = gap_data['true_values']
            
            # Use the same logic to load the correct cached model
            gap_year = df_gapped_features.index.year[0]
            start_year = int(np.floor(gap_year / imputation_chunk_size_years) * imputation_chunk_size_years)
            end_year = start_year + initial_train_window_size
            
            cache_key = imputer._create_cache_key(df_gapped_features, start_year, end_year)
            model = imputer.load_model_from_cache(cache_key)
            
            if model is None:
                print(f"Warning: No model found for period {start_year}-{end_year}. Skipping.")
                continue
                
            # Use the pre-trained model to impute
            imputed_data = imputer.impute(model, df_gapped_features)
            
            # Calculate metrics
            y_true, y_pred = [], []
            discharge_cols = [col for col in df_original.columns if col in df_gapped_features.columns]
            for col in discharge_cols:
                y_true.extend(true_values.get(col, []))
                if col in imputed_data.columns:
                    y_pred.extend(imputed_data.loc[df_gapped_features.index, col].values)

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

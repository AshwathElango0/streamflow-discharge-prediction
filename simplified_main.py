# simplified_main.py - Main script demonstrating the simplified burst imputation pipeline
import os
from simplified_evaluation import evaluate_burst_pipeline
from simplified_burst_pipeline import run_rolling_imputation_pipeline

def main():
    """Main function demonstrating the simplified burst imputation pipeline."""
    
    print("=== Simplified Burst Imputation Pipeline ===\n")
    
    # Configuration
    config = {
        'discharge_path': 'discharge_data_cleaned.csv',
        'lat_long_path': 'lat_long_discharge.csv', 
        'contrib_path': 'mahanadi_contribs.csv',
        'output_dir': 'simplified_results',
        'initial_train_window_size': 5,
        'imputation_chunk_size_years': 5,
        'min_completeness_percent_train': 70.0
    }
    
    print("1. Running Burst Imputation Pipeline...")
    print("-" * 50)
    
    # Run the burst imputation pipeline
    imputed_data = run_rolling_imputation_pipeline(
        discharge_path=config['discharge_path'],
        lat_long_path=config['lat_long_path'],
        contrib_path=config['contrib_path'],
        initial_train_window_size=config['initial_train_window_size'],
        imputation_chunk_size_years=config['imputation_chunk_size_years'],
        overall_min_year=1970,
        overall_max_year=2010,
        min_completeness_percent_train=config['min_completeness_percent_train'],
        output_dir=os.path.join(config['output_dir'], 'imputation')
    )
    
    if imputed_data is not None:
        print(f"✓ Imputation completed successfully!")
        print(f"  Final shape: {imputed_data.shape}")
        print(f"  Stations: {len(imputed_data.columns)}")
        print(f"  Time period: {imputed_data.index.min()} to {imputed_data.index.max()}")
    else:
        print("✗ Imputation failed!")
        return
    
    print("\n2. Running Evaluation on Artificial Gaps...")
    print("-" * 50)
    
    # Run evaluation on artificial gaps
    evaluation_results = evaluate_burst_pipeline(
        discharge_path=config['discharge_path'],
        lat_long_path=config['lat_long_path'],
        contrib_path=config['contrib_path'],
        output_dir=os.path.join(config['output_dir'], 'evaluation'),
        gap_lengths_contiguous=[7, 14, 30, 60, 180],
        gap_lengths_single_point=[30, 60, 90, 180],
        initial_train_window_size=config['initial_train_window_size'],
        imputation_chunk_size_years=config['imputation_chunk_size_years'],
        min_completeness_percent_train=config['min_completeness_percent_train']
    )
    
    if evaluation_results is not None:
        print(f"✓ Evaluation completed successfully!")
        print(f"  Results shape: {evaluation_results.shape}")
        print(f"  Gap types evaluated: {evaluation_results['Gap Type'].unique()}")
        print(f"  Gap lengths: {sorted(evaluation_results['Gap Length'].unique())}")
        
        # Show summary of results
        print("\n3. Evaluation Results Summary:")
        print("-" * 50)
        for gap_type in evaluation_results['Gap Type'].unique():
            type_results = evaluation_results[evaluation_results['Gap Type'] == gap_type]
            print(f"\n{gap_type} Gaps:")
            for metric in ['R2', 'RMSE', 'MAE', 'NSE']:
                if metric in type_results.columns:
                    mean_val = type_results[metric].mean()
                    print(f"  {metric}: {mean_val:.4f}")
    else:
        print("✗ Evaluation failed!")
    
    print(f"\n=== Pipeline Complete ===")
    print(f"Results saved in: {config['output_dir']}/")

if __name__ == '__main__':
    main()

# simplified_burst_pipeline.py - Streamlined burst imputation pipeline with rolling cache
import pandas as pd
import numpy as np
import os
import pickle
import hashlib
from missforest_imputer import ModifiedMissForest
from utils import (
    load_and_preprocess_data, 
    add_temporal_features, 
    build_distance_matrix, 
    build_connectivity_matrix
)
from simplified_model_config import create_full_model

MODEL_CACHE_DIR = "trained_models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class BurstImputer:
    """
    Simplified burst imputation pipeline with consolidated logic and
    a rolling window approach to training and caching models.
    """
    
    def __init__(self, 
                 initial_train_window_size=5,
                 min_completeness_percent_train=70.0):
        self.initial_train_window_size = initial_train_window_size
        self.min_completeness_percent_train = min_completeness_percent_train
        self.temporal_features = ['day_of_year_sin', 'day_of_year_cos']
        
    def find_best_training_period(self, df_data, min_year, max_year):
        """Find the best training period with highest completeness."""
        discharge_cols = [col for col in df_data.columns 
                         if not col.startswith('day_of_year_')]
        
        best_period = (None, None)
        best_stations = []
        max_completeness = -1.0
        
        print(f"Searching for the best {self.initial_train_window_size}-year training period...")
        
        for start_year in range(min_year, max_year - self.initial_train_window_size + 1):
            end_year = start_year + self.initial_train_window_size
            df_slice = df_data.loc[str(start_year):str(end_year)]
            
            # Count stations with sufficient data completeness
            station_completeness = (1.0 - df_slice[discharge_cols].isnull().sum() / len(df_slice)) * 100
            complete_stations = station_completeness[station_completeness >= self.min_completeness_percent_train].index.tolist()
            
            if len(complete_stations) > max_completeness:
                max_completeness = len(complete_stations)
                best_period = (start_year, end_year)
                best_stations = complete_stations
                
        if not best_stations:
            print("Warning: No suitable training period found.")
            return None, None, None
        
        print(f"Found best training period: {best_period[0]}-{best_period[1]} with {len(best_stations)} stations.")
        return df_data.loc[str(best_period[0]):str(best_period[1]), best_stations], best_stations, best_period
    
    def _create_cache_key(self, df_data, start_year, end_year):
        """Creates a unique hash for the dataframe and time period for caching."""
        # A simple string key is sufficient for a specific time period
        return f"{start_year}_{end_year}"
        
    def save_model_to_cache(self, model, cache_key):
        """Saves a trained model to the cache directory."""
        cache_path = os.path.join(MODEL_CACHE_DIR, f"model_{cache_key}.pkl")
        print(f"Saving model to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f)
            
    def load_model_from_cache(self, cache_key):
        """Loads a model from the cache directory if it exists."""
        cache_path = os.path.join(MODEL_CACHE_DIR, f"model_{cache_key}.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached model from: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def train_and_cache_model_for_period(self, df_train_period, df_coords, df_contrib, start_year, end_year):
        """Trains a model for a specific period and saves it to the cache."""
        cache_key = self._create_cache_key(df_train_period, start_year, end_year)
        cached_model = self.load_model_from_cache(cache_key)
        if cached_model:
            return cached_model

        print(f"No cached model for period {start_year}-{end_year}. Training a new model...")
        stations_to_impute = df_train_period.columns.tolist()
        distance_matrix = build_distance_matrix(df_coords, stations_to_impute)
        connectivity_matrix = build_connectivity_matrix(df_contrib, stations_to_impute)
        
        model = create_full_model(distance_matrix, connectivity_matrix, self.temporal_features)
        model.fit(df_train_period)
        print(f"Model for {start_year}-{end_year} trained successfully.")
        self.save_model_to_cache(model, cache_key)
        return model

    def impute_rolling_windows(self, df_with_features, df_coords, df_contrib, imputation_chunk_size_years):
        """
        Performs imputation using a rolling window approach, training/loading
        a model for each chunk.
        """
        imputed_dfs = []
        df_imputed = df_with_features.copy()
        
        all_years = sorted(df_with_features.index.year.unique())
        
        for year in range(all_years[0], all_years[-1], imputation_chunk_size_years):
            start_year = year
            end_year = start_year + self.initial_train_window_size
            
            # Find a suitable training period
            df_train_period, stations_to_impute, _ = self.find_best_training_period(
                df_with_features, start_year, start_year + imputation_chunk_size_years
            )
            
            if df_train_period is None:
                continue

            # Train and cache model for this specific period
            model = self.train_and_cache_model_for_period(
                df_train_period, df_coords, df_contrib, start_year, end_year
            )
            
            if model:
                # Identify the chunk to be imputed (the same 5-year period)
                imputation_chunk = df_with_features.loc[str(start_year):str(end_year)]
                
                # Perform imputation using the trained model
                imputed_chunk = model.predict(imputation_chunk)
                imputed_dfs.append(imputed_chunk)

        # Combine all imputed chunks
        if imputed_dfs:
            combined_df = pd.concat(imputed_dfs).sort_index()
            # The imputation is done on the full df now, so this is just to show how to use the model
            return combined_df
        
        return None

    def run_pipeline(self, discharge_path, lat_long_path, contrib_path, overall_min_year, overall_max_year, imputation_chunk_size_years, output_dir):
        """Main pipeline for training and imputation with rolling windows."""
        df_original, df_contrib, df_coords, _, _ = load_and_preprocess_data(
            discharge_path, lat_long_path, contrib_path
        )
        df_with_features = add_temporal_features(df_original)

        # This part will be handled by the imputation function itself now
        df_imputed = self.impute_rolling_windows(
            df_with_features=df_with_features,
            df_coords=df_coords,
            df_contrib=df_contrib,
            imputation_chunk_size_years=imputation_chunk_size_years
        )
        
        if df_imputed is None:
            print("Imputation pipeline failed.")
            return None

        # Save results
        output_path = os.path.join(output_dir, "imputed_discharge.csv")
        df_imputed.to_csv(output_path)
        print(f"Pipeline complete. Imputed data saved to: {output_path}")
        return df_imputed

def run_rolling_imputation_pipeline(
    discharge_path, lat_long_path, contrib_path,
    initial_train_window_size=5,
    imputation_chunk_size_years=5,
    overall_min_year=1976, overall_max_year=2016,
    min_completeness_percent_train=70.0,
    output_dir="bursting_imputed_results"):
    """Wrapper function for backward compatibility."""
    imputer = BurstImputer(
        initial_train_window_size=initial_train_window_size,
        min_completeness_percent_train=min_completeness_percent_train
    )
    
    return imputer.run_pipeline(
        discharge_path=discharge_path,
        lat_long_path=lat_long_path,
        contrib_path=contrib_path,
        overall_min_year=overall_min_year,
        overall_max_year=overall_max_year,
        imputation_chunk_size_years=imputation_chunk_size_years,
        output_dir=output_dir
    )

if __name__ == "__main__":
    # Example usage
    result = run_rolling_imputation_pipeline(
        discharge_path="discharge_data_cleaned.csv",
        lat_long_path="lat_long_discharge.csv",
        contrib_path="mahanadi_contribs.csv",
        initial_train_window_size=5,
        imputation_chunk_size_years=5,
        overall_min_year=1970,
        overall_max_year=2010,
        min_completeness_percent_train=70.0
    )
    if result is not None:
        print("\nPipeline run successfully!")
    else:
        print("\nPipeline failed.")

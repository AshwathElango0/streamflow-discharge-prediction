# simplified_burst_pipeline.py - Streamlined burst imputation pipeline
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

MODEL_CACHE_DIR = "trained_models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class BurstImputer:
    """Simplified burst imputation pipeline with consolidated logic."""
    
    def __init__(self, 
                 initial_train_window_size=5,
                 imputation_chunk_size_years=5,
                 min_completeness_percent_train=70.0):
        self.initial_train_window_size = initial_train_window_size
        self.imputation_chunk_size_years = imputation_chunk_size_years
        self.min_completeness_percent_train = min_completeness_percent_train
        self.temporal_features = ['day_of_year_sin', 'day_of_year_cos']
        
    def find_best_training_period(self, df_data, min_year, max_year):
        """Find the best training period with highest completeness."""
        discharge_cols = [col for col in df_data.columns 
                         if not col.startswith('day_of_year_')]
        
        best_period = (None, None)
        best_stations = []
        max_completeness = -1.0
        
        print(f"Finding best {self.initial_train_window_size}-year training period...")
        
        for start_year in range(min_year, max_year - self.initial_train_window_size + 2):
            end_year = start_year + self.initial_train_window_size - 1
            
            if start_year > df_data.index.year.max() or end_year < df_data.index.year.min():
                continue
                
            period_data = df_data[(df_data.index.year >= start_year) & 
                                (df_data.index.year <= end_year)]
            if period_data.empty:
                continue
                
            period_discharge = period_data[discharge_cols]
            total_cells = period_discharge.shape[0] * period_discharge.shape[1]
            
            if total_cells == 0:
                completeness = 0.0
            else:
                non_na_cells = period_discharge.notna().sum().sum()
                completeness = (non_na_cells / total_cells) * 100
            
            print(f"  Period {start_year}-{end_year}: {completeness:.2f}% completeness")
            
            if completeness >= self.min_completeness_percent_train:
                if completeness > max_completeness:
                    max_completeness = completeness
                    best_period = (start_year, end_year)
                    min_non_na_count = int(self.min_completeness_percent_train / 100 * len(period_data))
                    best_stations = period_discharge.columns[
                        period_discharge.notna().sum() >= min_non_na_count
                    ].tolist()
        
        if best_period == (None, None):
            print(f"ERROR: No suitable training period found.")
            return None, None, []
        
        print(f"Best period: {best_period[0]}-{best_period[1]} ({max_completeness:.2f}% completeness)")
        return best_period[0], best_period[1], best_stations
    
    def prepare_training_data(self, df_train_slice, discharge_cols):
        """Prepare training data with simulated missingness."""
        min_non_na_count = int(self.min_completeness_percent_train / 100 * len(df_train_slice))
        
        # Filter discharge columns with sufficient data
        valid_discharge_cols = df_train_slice[discharge_cols].columns[
            df_train_slice[discharge_cols].notna().sum() >= min_non_na_count
        ].tolist()
        
        if not valid_discharge_cols:
            return None, None
        
        # Combine discharge and feature columns
        all_cols = valid_discharge_cols + self.temporal_features
        df_train = df_train_slice[all_cols].copy()
        
        # Handle all-NaN columns
        for col in valid_discharge_cols:
            if df_train[col].isnull().all():
                df_train[col] = 0.0
            else:
                df_train[col] = df_train[col].fillna(df_train[col].mean())
        
        # Simulate 10% missingness for training
        np.random.seed(42)
        train_mask = np.random.rand(*df_train[valid_discharge_cols].shape) < 0.1
        df_train[valid_discharge_cols] = df_train[valid_discharge_cols].mask(train_mask)
        
        return df_train, valid_discharge_cols
    
    def train_model(self, df_train, discharge_cols, dist_matrix, conn_matrix, 
                   model_type="full_model", cache_key=""):
        """Train a model with caching."""
        model_file = os.path.join(MODEL_CACHE_DIR, f'model_{cache_key}.pkl')
        
        # Try to load from cache
        if os.path.exists(model_file):
            print(f"Loading model from cache: {model_file}")
            with open(model_file, 'rb') as f:
                return pickle.load(f)
        
        # Train new model
        print(f"Training {model_type}...")
        
        if model_type == "full_model":
            imputer = ModifiedMissForest(
                distance_matrix=dist_matrix,
                connectivity=conn_matrix,
                max_iter=10,
                n_estimators=100,
                random_state=42,
                distance_weighting_type='inverse',
                temporal_feature_columns=self.temporal_features
            )
        else:  # no_contributor_model
            conn_zero = pd.DataFrame(0.0, index=conn_matrix.index, columns=conn_matrix.columns)
            imputer = ModifiedMissForest(
                distance_matrix=dist_matrix,
                connectivity=conn_zero,
                max_iter=10,
                n_estimators=100,
                random_state=42,
                distance_weighting_type='inverse',
                temporal_feature_columns=self.temporal_features
            )
        
        try:
            imputer.fit(df_train)
            # Cache the model
            with open(model_file, 'wb') as f:
                pickle.dump(imputer, f)
            print(f"Model saved to cache: {model_file}")
            return imputer
        except Exception as e:
            print(f"ERROR: Model training failed: {e}")
            return None
    
    def impute_chunk(self, model, df_chunk, discharge_cols):
        """Impute a chunk of data using the trained model."""
        all_cols = discharge_cols + self.temporal_features
        df_to_impute = df_chunk.reindex(columns=all_cols).copy()
        
        # Handle missing columns
        for col in discharge_cols:
            if col not in df_to_impute.columns:
                df_to_impute[col] = np.nan
            if df_to_impute[col].isnull().all():
                df_to_impute[col] = 0.0
        
        try:
            imputed_chunk = model.transform(df_to_impute)
            return imputed_chunk[discharge_cols]
        except Exception as e:
            print(f"ERROR: Imputation failed: {e}")
            return None
    
    def run_pipeline(self, discharge_path, lat_long_path, contrib_path=None,
                    overall_min_year=1976, overall_max_year=2016,
                    output_dir="bursting_imputed_results"):
        """Run the complete burst imputation pipeline."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("--- Starting Burst Imputation Pipeline ---")
        
        # Load data
        df_discharge, df_contrib, df_coords, vcode_to_station, station_to_vcode = \
            load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
        
        if df_discharge is None:
            print("ERROR: Failed to load data.")
            return None
        
        # Add temporal features
        df_with_features = add_temporal_features(df_discharge)
        all_discharge_cols = [col for col in df_discharge.columns 
                             if not col.startswith('day_of_year_')]
        
        # Initialize master imputed dataframe
        df_imputed = df_with_features.copy()
        
        # Find best initial training period
        train_start, train_end, train_stations = self.find_best_training_period(
            df_with_features, overall_min_year, overall_max_year
        )
        
        if train_start is None:
            return None
        
        # Prepare initial training data
        df_train_slice = df_imputed.loc[
            (df_imputed.index.year >= train_start) & 
            (df_imputed.index.year <= train_end)
        ]
        df_train, valid_discharge_cols = self.prepare_training_data(
            df_train_slice, train_stations
        )
        
        if df_train is None:
            print("ERROR: Failed to prepare training data.")
            return None
        
        # Build matrices for initial training
        dist_matrix = build_distance_matrix(df_coords, valid_discharge_cols)
        if contrib_path and df_contrib is not None and not df_contrib.empty:
            conn_matrix = build_connectivity_matrix(df_contrib, valid_discharge_cols, station_to_vcode)
            model_type = "full_model"
        else:
            conn_matrix = pd.DataFrame(0, index=valid_discharge_cols, 
                                     columns=valid_discharge_cols, dtype=int)
            model_type = "no_contributor_model"
        
        # Train initial model
        cache_key = hashlib.md5(f"{discharge_path}-{lat_long_path}-{contrib_path}-{train_start}-{train_end}-{self.min_completeness_percent_train}-{model_type}".encode()).hexdigest()
        initial_model = self.train_model(df_train, valid_discharge_cols, 
                                       dist_matrix, conn_matrix, model_type, cache_key)
        
        if initial_model is None:
            print("ERROR: Initial model training failed.")
            return None
        
        # Initial imputation (forward and backward)
        print("\nPerforming initial imputation...")
        
        # Forward
        forward_start = train_end + 1
        forward_end = min(overall_max_year, forward_start + self.imputation_chunk_size_years - 1)
        if forward_start <= forward_end:
            df_chunk = df_with_features.loc[f"{forward_start}":f"{forward_end}"]
            imputed_chunk = self.impute_chunk(initial_model, df_chunk, valid_discharge_cols)
            if imputed_chunk is not None:
                df_imputed.loc[imputed_chunk.index, imputed_chunk.columns] = imputed_chunk.values
                print(f"Forward imputation: {forward_start}-{forward_end}")
        
        # Backward
        backward_end = train_start - 1
        backward_start = max(overall_min_year, backward_end - self.imputation_chunk_size_years + 1)
        if backward_start <= backward_end:
            df_chunk = df_with_features.loc[f"{backward_start}":f"{backward_end}"]
            imputed_chunk = self.impute_chunk(initial_model, df_chunk, valid_discharge_cols)
            if imputed_chunk is not None:
                df_imputed.loc[imputed_chunk.index, imputed_chunk.columns] = imputed_chunk.values
                print(f"Backward imputation: {backward_start}-{backward_end}")
        
        # Bursting loops (forward and backward)
        print("\n--- Starting Bursting Loops ---")
        
        # Forward bursting
        current_start = forward_end + 1
        last_imputed = df_imputed.loc[
            (df_imputed.index.year >= train_start) & 
            (df_imputed.index.year <= train_end),
            valid_discharge_cols
        ].copy()
        
        while current_start <= overall_max_year:
            current_end = min(overall_max_year, current_start + self.imputation_chunk_size_years - 1)
            print(f"Forward bursting: {current_start}-{current_end}")
            
            # Prepare training data from last imputed chunk
            df_train_slice = last_imputed.merge(
                df_with_features[self.temporal_features], 
                left_index=True, right_index=True
            )
            df_train, cols_to_keep = self.prepare_training_data(
                df_train_slice, last_imputed.columns.tolist()
            )
            
            if df_train is None:
                break
            
            # Build matrices and train model
            dist_matrix = build_distance_matrix(df_coords, cols_to_keep)
            if contrib_path and df_contrib is not None and not df_contrib.empty:
                conn_matrix = build_connectivity_matrix(df_contrib, cols_to_keep, station_to_vcode)
                model_type = "full_model"
            else:
                conn_matrix = pd.DataFrame(0, index=cols_to_keep, 
                                         columns=cols_to_keep, dtype=int)
                model_type = "no_contributor_model"
            
            cache_key = hashlib.md5(f"{discharge_path}-{lat_long_path}-{contrib_path}-{current_start}-{current_end}-{self.min_completeness_percent_train}-{model_type}".encode()).hexdigest()
            model = self.train_model(df_train, cols_to_keep, dist_matrix, conn_matrix, model_type, cache_key)
            
            if model is None:
                break
            
            # Impute next chunk
            df_chunk = df_with_features.loc[f"{current_start}":f"{current_end}"]
            imputed_chunk = self.impute_chunk(model, df_chunk, model.discharge_columns)
            
            if imputed_chunk is not None:
                df_imputed.loc[imputed_chunk.index, imputed_chunk.columns] = imputed_chunk.values
                last_imputed = imputed_chunk.copy()
                print(f"Forward imputation complete: {current_start}-{current_end}")
            
            current_start = current_end + 1
        
        # Backward bursting
        current_end = backward_start - 1
        last_imputed = df_imputed.loc[
            (df_imputed.index.year >= train_start) & 
            (df_imputed.index.year <= train_end),
            valid_discharge_cols
        ].copy()
        
        while current_end >= overall_min_year:
            current_start = max(overall_min_year, current_end - self.imputation_chunk_size_years + 1)
            print(f"Backward bursting: {current_start}-{current_end}")
            
            # Prepare training data from last imputed chunk
            df_train_slice = last_imputed.merge(
                df_with_features[self.temporal_features], 
                left_index=True, right_index=True
            )
            df_train, cols_to_keep = self.prepare_training_data(
                df_train_slice, last_imputed.columns.tolist()
            )
            
            if df_train is None:
                break
            
            # Build matrices and train model
            dist_matrix = build_distance_matrix(df_coords, cols_to_keep)
            if contrib_path and df_contrib is not None and not df_contrib.empty:
                conn_matrix = build_connectivity_matrix(df_contrib, cols_to_keep, station_to_vcode)
                model_type = "full_model"
            else:
                conn_matrix = pd.DataFrame(0, index=cols_to_keep, 
                                         columns=cols_to_keep, dtype=int)
                model_type = "no_contributor_model"
            
            cache_key = hashlib.md5(f"{discharge_path}-{lat_long_path}-{contrib_path}-{current_start}-{current_end}-{self.min_completeness_percent_train}-{model_type}".encode()).hexdigest()
            model = self.train_model(df_train, cols_to_keep, dist_matrix, conn_matrix, model_type, cache_key)
            
            if model is None:
                break
            
            # Impute next chunk
            df_chunk = df_with_features.loc[f"{current_start}":f"{current_end}"]
            imputed_chunk = self.impute_chunk(model, df_chunk, model.discharge_columns)
            
            if imputed_chunk is not None:
                df_imputed.loc[imputed_chunk.index, imputed_chunk.columns] = imputed_chunk.values
                last_imputed = imputed_chunk.copy()
                print(f"Backward imputation complete: {current_start}-{current_end}")
            
            current_end = current_start - 1
        
        # Save final results
        final_stations = [col for col in all_discharge_cols 
                         if col in df_imputed.columns and not df_imputed[col].isna().all()]
        
        output_file = os.path.join(output_dir, "final_bursting_imputed_data.csv")
        df_imputed[final_stations].to_csv(output_file)
        
        print(f"\nBurst imputation complete!")
        print(f"Final stations: {len(final_stations)} out of {len(all_discharge_cols)}")
        print(f"Output saved to: {output_file}")
        
        return df_imputed[final_stations]

def run_rolling_imputation_pipeline(discharge_path, lat_long_path, contrib_path=None,
                                  initial_train_window_size=5,
                                  imputation_chunk_size_years=5,
                                  overall_min_year=1976, overall_max_year=2016,
                                  min_completeness_percent_train=70.0,
                                  output_dir="bursting_imputed_results"):
    """Wrapper function for backward compatibility."""
    imputer = BurstImputer(
        initial_train_window_size=initial_train_window_size,
        imputation_chunk_size_years=imputation_chunk_size_years,
        min_completeness_percent_train=min_completeness_percent_train
    )
    
    return imputer.run_pipeline(
        discharge_path=discharge_path,
        lat_long_path=lat_long_path,
        contrib_path=contrib_path,
        overall_min_year=overall_min_year,
        overall_max_year=overall_max_year,
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
        min_completeness_percent_train=70.0,
        output_dir="bursting_imputed_results"
    )
    
    if result is not None:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed.")

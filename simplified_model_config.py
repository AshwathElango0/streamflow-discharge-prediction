# simplified_model_config.py - Streamlined model configuration
import pandas as pd
import numpy as np
from missforest_imputer import ModifiedMissForest

def create_full_model(distance_matrix, connectivity_matrix, temporal_features):
    """
    Creates a ModifiedMissForest model with all features (spatial + temporal).
    This is the main model configuration used throughout the pipeline.
    """
    return ModifiedMissForest(
        distance_matrix=distance_matrix,
        connectivity=connectivity_matrix,
        max_iter=10,
        n_estimators=100,
        random_state=42,
        distance_weighting_type='inverse',
        temporal_feature_columns=temporal_features
    )

def create_no_contributor_model(distance_matrix, temporal_features):
    """
    Creates a ModifiedMissForest model without contributor connectivity.
    Used as fallback when contributor data is not available.
    """
    # Create zero connectivity matrix to nullify contributor effects
    connectivity_zero = pd.DataFrame(
        0.0, 
        index=distance_matrix.index, 
        columns=distance_matrix.columns
    )
    
    return ModifiedMissForest(
        distance_matrix=distance_matrix,
        connectivity=connectivity_zero,
        max_iter=10,
        n_estimators=100,
        random_state=42,
        distance_weighting_type='inverse',
        temporal_feature_columns=temporal_features
    )

def train_model(df_train, discharge_cols, distance_matrix, connectivity_matrix, 
                temporal_features, model_type="full_model"):
    """
    Train a model with the specified configuration.
    
    Args:
        df_train: Training data
        discharge_cols: List of discharge column names
        distance_matrix: Spatial distance matrix
        connectivity_matrix: Hydrological connectivity matrix
        temporal_features: List of temporal feature names
        model_type: Either "full_model" or "no_contributor_model"
    
    Returns:
        Trained ModifiedMissForest model or None if training fails
    """
    print(f"Training {model_type}...")
    
    try:
        if model_type == "full_model":
            model = create_full_model(distance_matrix, connectivity_matrix, temporal_features)
        elif model_type == "no_contributor_model":
            model = create_no_contributor_model(distance_matrix, temporal_features)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(df_train)
        print(f"{model_type} trained successfully.")
        return model
        
    except Exception as e:
        print(f"ERROR: {model_type} training failed: {e}")
        return None

# Backward compatibility functions
def train_full_model(df_train_masked, distance_matrix, connectivity_matrix, all_feature_cols):
    """Backward compatibility wrapper for full model training."""
    discharge_cols = [col for col in df_train_masked.columns 
                     if not col.startswith('day_of_year_')]
    return train_model(df_train_masked, discharge_cols, distance_matrix, 
                      connectivity_matrix, all_feature_cols, "full_model")

def train_no_contributor_model(df_train_masked, distance_matrix, connectivity_matrix, all_feature_cols):
    """Backward compatibility wrapper for no-contributor model training."""
    discharge_cols = [col for col in df_train_masked.columns 
                     if not col.startswith('day_of_year_')]
    return train_model(df_train_masked, discharge_cols, distance_matrix, 
                      connectivity_matrix, all_feature_cols, "no_contributor_model")
